#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <algorithm>
#include <stdexcept>

// Avoid macro collisions with CUTLASS prefetch helpers.
#ifdef prefetch
#undef prefetch
#endif

#include <cute/algorithm/copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/tensor.hpp>
using cute::Copy_Atom;
namespace {

__device__ inline float dot_tile_fallback(const float* q, const float* k, int head_dim) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        acc += q[d] * k[d];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

constexpr bool kTmemAvailable =
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    true;
#else
    false;
#endif

constexpr int TILE_M = 32;
constexpr int TILE_N = 64;
static_assert(TILE_N <= cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns,
              "TILE_N exceeds TMEM column capacity");

__global__ void persistent_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int batch,
    int seq_len,
    int head_dim
) {
    constexpr int MAX_HEAD_DIM = 128;
    const int seq_id = blockIdx.x;
    if (seq_id >= batch) {
        return;
    }
    constexpr int MAX_SEQ_LEN = 64;
    if (seq_len > MAX_SEQ_LEN) {
        return;
    }

    // shared layout: K0|K1|V0|V1|reduce|tmep_stage
    extern __shared__ float smem_f[];
    float* smem_k0 = smem_f;
    float* smem_k1 = smem_k0 + MAX_HEAD_DIM;
    float* smem_v0 = smem_k1 + MAX_HEAD_DIM;
    float* smem_v1 = smem_v0 + MAX_HEAD_DIM;
    float* red = smem_v1 + MAX_HEAD_DIM;
    float* tmep_stage = red + blockDim.x;  // up to MAX_SEQ_LEN x head_dim

    __shared__ alignas(128) float tmem_tile[TILE_M][TILE_N];
    __shared__ uint32_t tmem_base_ptr;

    for (int t = 0; t < seq_len; ++t) {
        const float* q_ptr = q + (seq_id * seq_len + t) * head_dim;
        const float* k_ptr = k + (seq_id * seq_len + t) * head_dim;
        const float* v_ptr = v + (seq_id * seq_len + t) * head_dim;

        // fallback dot + scale
        float dot = dot_tile_fallback(q_ptr, k_ptr, head_dim);
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            tmep_stage[t * head_dim + d] = v_ptr[d] * dot;
        }
        __syncthreads();
        // stash V into shared for possible future cp.async re-enable
        if (head_dim <= MAX_HEAD_DIM) {
            float* v_smem_curr = (t & 1) ? smem_v1 : smem_v0;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                v_smem_curr[d] = v_ptr[d];
            }
        }
        __syncthreads();
    }
    // TMEM epilogue: tiles of 32x64, no fallback path.
    if (!kTmemAvailable) {
        return;
    }

    // Allocate full TMEM slice once per CTA.
    if (threadIdx.x == 0) {
        cute::TMEM::Allocator1Sm allocator{};
        allocator.allocate(cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns, &tmem_base_ptr);
    }
    __syncthreads();

    for (int row = 0; row < seq_len; row += TILE_M) {
        for (int col = 0; col < head_dim; col += TILE_N) {
            const int rows_this = TILE_M;
            const int cols_this = TILE_N;

            // Load tile from tmep_stage into shared tile (full tiles only).
            for (int idx = threadIdx.x; idx < rows_this * cols_this; idx += blockDim.x) {
                int r = idx / cols_this;
                int c = idx - r * cols_this;
                tmem_tile[r][c] = tmep_stage[(row + r) * head_dim + (col + c)];
            }
            __syncthreads();

            // TMEM store and load via CUTLASS copy atoms.
            auto tmem_tensor = cute::make_tensor(
                cute::make_tmem_ptr<float>(tmem_base_ptr),
                cute::make_layout(
                    cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_N>{}),
                    cute::make_stride(cute::TMEM::DP<float>{}, cute::Int<1>{})));

            auto smem_tensor = cute::make_tensor(
                cute::make_smem_ptr(&tmem_tile[0][0]),
                cute::make_layout(
                    cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_N>{}),
                    cute::make_stride(cute::Int<TILE_N>{}, cute::Int<1>{})));

            auto gmem_tensor = cute::make_tensor(
                cute::make_gmem_ptr(out + (seq_id * seq_len + row) * head_dim + col),
                cute::make_layout(
                    cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_N>{}),
                    cute::make_stride(cute::Int<TILE_N>{}, cute::Int<1>{})));

            auto tmem_store = cute::make_tmem_copy(Copy_Atom<cute::SM100_TMEM_STORE_32dp32b4x, float>{}, tmem_tensor);
            auto tmem_load = cute::make_tmem_copy(Copy_Atom<cute::SM100_TMEM_LOAD_32dp32b4x, float>{}, tmem_tensor);

            if (threadIdx.x < 32) {
                auto store_thr = tmem_store.get_slice(threadIdx.x);
                auto src = store_thr.partition_S(smem_tensor);
                auto dst = store_thr.partition_D(tmem_tensor);
                cute::copy(tmem_store, src, dst);
            }
            __syncthreads();
            if (threadIdx.x < 32) {
                auto load_thr = tmem_load.get_slice(threadIdx.x);
                auto src = load_thr.partition_S(tmem_tensor);
                auto dst = load_thr.partition_D(gmem_tensor);
                cute::copy(tmem_load, src, dst);
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        cute::TMEM::Allocator1Sm allocator{};
        allocator.release_allocation_lock();
        allocator.free(tmem_base_ptr, cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns);
    }
}

} // namespace

void persistent_decode_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out, int blocks) {
    if (!q.is_cuda()) {
        throw std::runtime_error("q must be CUDA");
    }
    if (q.scalar_type() != torch::kFloat) {
        throw std::runtime_error("q must be float32");
    }
    if (!(q.sizes() == k.sizes() && q.sizes() == v.sizes())) {
        throw std::runtime_error("q/k/v shapes must match");
    }
    if (out.sizes() != q.sizes()) {
        throw std::runtime_error("out shape mismatch");
    }
    if (q.size(2) > 128) {
        throw std::runtime_error("head_dim exceeds MAX_HEAD_DIM=128");
    }

    const int batch = static_cast<int>(q.size(0));
    const int seq_len = static_cast<int>(q.size(1));
    const int head_dim = static_cast<int>(q.size(2));
    const int threads = 64;

    constexpr int MAX_HEAD_DIM = 128;
    constexpr int MAX_SEQ_LEN = 48;
    if (seq_len % TILE_M != 0 || head_dim != TILE_N) {
        throw std::runtime_error("seq_len must be a multiple of 32 and head_dim must equal 64 for TMEM path");
    }
    if (!kTmemAvailable) {
        throw std::runtime_error("TMEM required but not available on this build/device");
    }
    const size_t smem_bytes = (4 * MAX_HEAD_DIM + threads + MAX_SEQ_LEN * MAX_HEAD_DIM) * sizeof(float);

    c10::cuda::CUDAGuard guard(q.get_device());
    cudaDeviceProp prop{};
    auto err = cudaGetDeviceProperties(&prop, q.get_device());
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceProperties failed");
    }
    if (prop.major < 10) {
        throw std::runtime_error("persistent_decode TMEM path requires SM100+");
    }

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    const int blocks_per_batch = std::min(blocks, batch);
    persistent_decode_kernel<<<blocks_per_batch, threads, smem_bytes, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        seq_len,
        head_dim);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("persistent_decode_kernel launch failed");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("persistent_decode", &persistent_decode_cuda, "Persistent decode (CUDA)");
}
