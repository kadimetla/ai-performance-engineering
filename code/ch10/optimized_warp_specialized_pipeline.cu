// Chapter 10: Book-aligned warp-specialized pipeline using cuda::pipeline for loader/compute/storer handoff.
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>
#include <numeric>

#include "../core/common/headers/cuda_verify.cuh"

namespace cg = cooperative_groups;

namespace {
constexpr int TILE_SIZE = 64;
constexpr int TILE_ELEMS = TILE_SIZE * TILE_SIZE;
constexpr int PIPELINE_STAGES = 2;
constexpr int WARPS_PER_BLOCK = 3;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

__device__ void compute_full_tile(const float* __restrict__ A_tile,
                                  const float* __restrict__ B_tile,
                                  float* __restrict__ C_tile,
                                  int lane_id) {
    for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
        int row = idx / TILE_SIZE;
        int col = idx % TILE_SIZE;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += A_tile[row * TILE_SIZE + k] * B_tile[k * TILE_SIZE + col];
        }
        C_tile[idx] = acc;
    }
}

__global__ void optimized_warp_specialized_kernel(const float* __restrict__ A_global,
                                                  const float* __restrict__ B_global,
                                                  float* __restrict__ C_global,
                                                  int num_tiles) {
    cg::thread_block cta = cg::this_thread_block();

    extern __shared__ float shared_mem[];
    float* A_stage_base = shared_mem;
    float* B_stage_base = A_stage_base + PIPELINE_STAGES * TILE_ELEMS;
    float* C_stage_base = B_stage_base + PIPELINE_STAGES * TILE_ELEMS;

    using pipeline_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_STAGES>;
    __shared__ alignas(pipeline_state_t) unsigned char state_bytes[sizeof(pipeline_state_t)];
    auto* state = reinterpret_cast<pipeline_state_t*>(state_bytes);
    if (threadIdx.x == 0) {
        new (state) pipeline_state_t();
    }
    __syncthreads();
    auto pipe = cuda::make_pipeline(cta, state);

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    auto warp = cg::tiled_partition<32>(cta);

    int stride = gridDim.x;
    for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
        int tile = blockIdx.x + stage * stride;
        if (tile >= num_tiles) {
            break;
        }
        size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;
        float* stage_a = A_stage_base + stage * TILE_ELEMS;
        float* stage_b = B_stage_base + stage * TILE_ELEMS;

        if (warp_id == 0) {
            pipe.producer_acquire();
            cuda::memcpy_async(
                warp,
                stage_a,
                A_global + offset,
                cuda::aligned_size_t<16>(static_cast<size_t>(TILE_ELEMS) * sizeof(float)),
                pipe);
            cuda::memcpy_async(
                warp,
                stage_b,
                B_global + offset,
                cuda::aligned_size_t<16>(static_cast<size_t>(TILE_ELEMS) * sizeof(float)),
                pipe);
            pipe.producer_commit();
        }
    }
    __syncthreads();

    int tile_iter = 0;
    for (int tile = blockIdx.x; tile < num_tiles; tile += stride, ++tile_iter) {
        int stage = tile_iter % PIPELINE_STAGES;
        float* A_tile = A_stage_base + stage * TILE_ELEMS;
        float* B_tile = B_stage_base + stage * TILE_ELEMS;
        float* C_tile = C_stage_base + stage * TILE_ELEMS;
        size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;

        pipe.consumer_wait();
        __syncthreads();

        if (warp_id == 1) {
            compute_full_tile(A_tile, B_tile, C_tile, lane_id);
        }

        __syncthreads();

        if (warp_id == 2) {
            for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
                C_global[offset + idx] = C_tile[idx];
            }
        }

        __syncthreads();
        pipe.consumer_release();
        __syncthreads();

        int next_tile = tile + PIPELINE_STAGES * stride;
        if (next_tile < num_tiles) {
            int next_stage = (tile_iter + PIPELINE_STAGES) % PIPELINE_STAGES;
            float* next_a = A_stage_base + next_stage * TILE_ELEMS;
            float* next_b = B_stage_base + next_stage * TILE_ELEMS;
            size_t next_offset = static_cast<size_t>(next_tile) * TILE_ELEMS;
            if (warp_id == 0) {
                pipe.producer_acquire();
                cuda::memcpy_async(
                    warp,
                    next_a,
                    A_global + next_offset,
                    cuda::aligned_size_t<16>(static_cast<size_t>(TILE_ELEMS) * sizeof(float)),
                    pipe);
                cuda::memcpy_async(
                    warp,
                    next_b,
                    B_global + next_offset,
                    cuda::aligned_size_t<16>(static_cast<size_t>(TILE_ELEMS) * sizeof(float)),
                    pipe);
                pipe.producer_commit();
            }
        }
        __syncthreads();
    }
}

void run_optimized(int tiles) {
    const size_t bytes = static_cast<size_t>(tiles) * TILE_ELEMS * sizeof(float);
    std::vector<float> h_A(bytes / sizeof(float));
    std::vector<float> h_B(bytes / sizeof(float));
    std::vector<float> h_C(bytes / sizeof(float));
    std::iota(h_A.begin(), h_A.end(), 0.0f);
    std::iota(h_B.begin(), h_B.end(), 1.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(std::min(tiles, 128));
    size_t shared_bytes = 3 * PIPELINE_STAGES * TILE_ELEMS * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    optimized_warp_specialized_kernel<<<grid, block, shared_bytes>>>(d_A, d_B, d_C, tiles);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (float v : h_C) checksum += v;

    printf("optimized_warp_specialized_pipeline: %d tiles, %.3f ms, checksum %.3f\n",
           tiles, ms, checksum / h_C.size());

#ifdef VERIFY
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
}  // namespace

int main() {
    run_optimized(512);
    return 0;
}
