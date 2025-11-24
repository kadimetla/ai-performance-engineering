// optimized_flash_attn_tma_micro_pipeline.cu
//
// FlashAttention-shaped micro-pipeline using cuda::pipeline (maps to cp.async/TMA
// with mbarriers on Hopper/Blackwell). Demonstrates double-buffered PREFETCH
// (K/V tiles) overlapped with COMPUTE (QK^T + apply V).

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda/barrier>
#include <cooperative_groups.h>

#include "../common/headers/tma_helpers.cuh"

#include <cstdio>
#include <cstdlib>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t _status = (call);                                        \
        if (_status != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(_status));   \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)
#endif

#if CUDART_VERSION < 12000
int main() {
    std::printf("SKIP: cuda::pipeline requires CUDA 12.x+ (found %d)\n", CUDART_VERSION);
    return 0;
}
#else

constexpr int SEQ_LEN = 2048;
constexpr int D_HEAD  = 64;
constexpr int TILE_KV = 40;
constexpr int THREADS = 128;
constexpr int STAGES  = 2;  // double buffer

namespace cg = cooperative_groups;
namespace cde = cuda::device::experimental;
using namespace cuda::device::experimental;

using block_barrier = cuda::barrier<cuda::thread_scope_block>;

__global__ void flash_attn_tma_microkernel(
    const __grid_constant__ CUtensorMap k_desc,
    const __grid_constant__ CUtensorMap v_desc,
    const float* __restrict__ q,
    float* __restrict__ o,
    int seq_len,
    int d_head) {
    const int q_idx = blockIdx.x;
    if (q_idx >= seq_len) return;

    extern __shared__ float smem[];
    __shared__ alignas(128) unsigned char stage_barrier_storage[STAGES][sizeof(block_barrier)];
    float* smem_k[STAGES];
    float* smem_v[STAGES];
    const int tile_elems = TILE_KV * d_head;
    smem_k[0] = smem;
    smem_v[0] = smem_k[0] + tile_elems;
    smem_k[1] = smem_v[0] + tile_elems;
    smem_v[1] = smem_k[1] + tile_elems;

    const int participants = blockDim.x;
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[s]);
            init(bar_ptr, participants);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    __shared__ float score_smem[THREADS];
    const int tid = threadIdx.x;

    // Load Q row into registers once.
    float q_reg[D_HEAD];
    for (int d = tid; d < d_head; d += blockDim.x) {
        q_reg[d] = q[q_idx * d_head + d];
    }
    __syncthreads();

    float o_reg[D_HEAD];
    for (int d = 0; d < d_head; ++d) o_reg[d] = 0.f;

    const int num_tiles = (seq_len + TILE_KV - 1) / TILE_KV;
    block_barrier::arrival_token stage_tokens[STAGES];

    auto issue_tile = [&](int tile_idx) {
        if (tile_idx >= num_tiles) return;
        const int stage = tile_idx & 1;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
        const int row_base = tile_idx * TILE_KV;
        const int rows = min(TILE_KV, seq_len - row_base);
        const std::size_t bytes = std::size_t(rows) * d_head * sizeof(float);

        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                smem_k[stage],
                &k_desc,
                /*coord_row=*/row_base,
                /*coord_col=*/0,
                *bar_ptr);
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                smem_v[stage],
                &v_desc,
                /*coord_row=*/row_base,
                /*coord_col=*/0,
                *bar_ptr);
            stage_tokens[stage] = cuda::device::barrier_arrive_tx(*bar_ptr, 1, bytes * 2);
        } else {
            stage_tokens[stage] = bar_ptr->arrive();
        }
    };

    const int preload = min(num_tiles, STAGES);
    for (int t = 0; t < preload; ++t) {
        issue_tile(t);
    }

    int stage = 0;
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        stage = tile_idx & 1;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
        bar_ptr->wait(std::move(stage_tokens[stage]));
        __syncthreads();

        const int row_base = tile_idx * TILE_KV;
        const int rows_this_tile = min(TILE_KV, seq_len - row_base);

        for (int r = 0; r < rows_this_tile; ++r) {
            const float* k_row = smem_k[stage] + r * d_head;
            const float* v_row = smem_v[stage] + r * d_head;

            float score = 0.f;
            for (int d = tid; d < d_head; d += blockDim.x) {
                score += q_reg[d] * k_row[d];
            }

            score_smem[tid] = score;
            __syncthreads();
            if (tid < 64) score_smem[tid] += score_smem[tid + 64];
            __syncthreads();
            if (tid < 32) score_smem[tid] += score_smem[tid + 32];
            __syncthreads();

            if (tid == 0) {
                float s = score_smem[0];
                s = fminf(fmaxf(s, -10.f), 10.f);
                score_smem[0] = __expf(s) * 1e-3f;
            }
            __syncthreads();

            float weight = score_smem[0];
            for (int d = tid; d < d_head; d += blockDim.x) {
                o_reg[d] += weight * v_row[d];
            }
            __syncthreads();
        }

        const int next_tile = tile_idx + 1;
        if (next_tile < num_tiles) {
            issue_tile(next_tile);
        }
    }

    for (int d = tid; d < d_head; d += blockDim.x) {
        o[q_idx * d_head + d] = o_reg[d];
    }
}

bool supports_device_pipeline() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    const int sm = prop.major * 10 + prop.minor;
    // Enforce no-fallback behavior: require SM90+ and TMA capability.
    #ifdef cudaDevAttrTensorMapAccessSupported
    int attr = 0;
    if (cudaDeviceGetAttribute(&attr, cudaDevAttrTensorMapAccessSupported, 0) != cudaSuccess) {
        return false;
    }
    const bool tma_available = (attr != 0);
    const int d_head_bytes = D_HEAD * sizeof(float);
    const bool alignment_ok = (d_head_bytes % 16) == 0;  // TMA leading dimension alignment
    if (sm < 90 || !tma_available || !alignment_ok) return false;
    return true;
    #else
    return false;
    #endif
}

int main() {
    if (!supports_device_pipeline()) {
        std::printf("SKIP: No capable GPU for pipeline/TMA path.\nelapsed_ms=0.0 ms\n");
        return 0;
    }

    const int seq_len = SEQ_LEN;
    const int d_head = D_HEAD;
    const size_t bytes = size_t(seq_len) * d_head * sizeof(float);

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    CHECK_CUDA(cudaMalloc(&d_q, bytes));
    CHECK_CUDA(cudaMalloc(&d_k, bytes));
    CHECK_CUDA(cudaMalloc(&d_v, bytes));
    CHECK_CUDA(cudaMalloc(&d_o, bytes));

    CHECK_CUDA(cudaMemset(d_q, 0, bytes));
    CHECK_CUDA(cudaMemset(d_k, 0, bytes));
    CHECK_CUDA(cudaMemset(d_v, 0, bytes));
    CHECK_CUDA(cudaMemset(d_o, 0, bytes));

    // Build TensorMap descriptors for TMA copies.
    CUtensorMap k_desc{};
    CUtensorMap v_desc{};
    auto encode = cuda_tma::load_cuTensorMapEncodeTiled();
    if (!encode) {
        std::printf("SKIP: cuTensorMapEncodeTiled unavailable.\nelapsed_ms=0.0 ms\n");
        return 0;
    }
    const int box_h = TILE_KV;
    const int box_w = d_head;
    if (!cuda_tma::make_2d_tensor_map(
            k_desc, encode, d_k, d_head, seq_len, d_head, box_w, box_h, CU_TENSOR_MAP_SWIZZLE_128B) ||
        !cuda_tma::make_2d_tensor_map(
            v_desc, encode, d_v, d_head, seq_len, d_head, box_w, box_h, CU_TENSOR_MAP_SWIZZLE_128B)) {
        std::printf("SKIP: Failed to encode TMA descriptors.\nelapsed_ms=0.0 ms\n");
        return 0;
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const dim3 block(THREADS);
    const dim3 grid(seq_len);
    const size_t shmem_bytes = STAGES * 2 * TILE_KV * d_head * sizeof(float);  // K + V per stage

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    flash_attn_tma_microkernel<<<grid, block, shmem_bytes, stream>>>(
        k_desc, v_desc, d_q, d_o, seq_len, d_head);
    CHECK_CUDA(cudaEventRecord(stop, stream));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_o));

    std::printf("elapsed_ms=%.3f ms\n", ms);
    return 0;
}

#endif  // CUDART_VERSION
