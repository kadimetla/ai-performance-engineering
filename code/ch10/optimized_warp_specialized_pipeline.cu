// Chapter 10: Book-aligned warp-specialized pipeline using cuda::pipeline for loader/compute/storer handoff.
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>
#include <numeric>

namespace cg = cooperative_groups;

namespace {
constexpr int TILE_SIZE = 64;
constexpr int TILE_ELEMS = TILE_SIZE * TILE_SIZE;
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
    float* A_tile = shared_mem;
    float* B_tile = A_tile + TILE_ELEMS;
    float* C_tile = B_tile + TILE_ELEMS;

    using pipeline_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, 1>;
    __shared__ alignas(pipeline_state_t) unsigned char pipe_state_bytes[sizeof(pipeline_state_t)];
    auto* pipe_state = reinterpret_cast<pipeline_state_t*>(pipe_state_bytes);
    if (threadIdx.x == 0) {
        new (pipe_state) pipeline_state_t();
    }
    __syncthreads();
    auto pipe = cuda::make_pipeline(cta, pipe_state);

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    const int warps_per_grid = (gridDim.x * blockDim.x) / warpSize;
    const int global_warp = warp_id + (blockIdx.x * blockDim.x) / warpSize;

    for (int tile = global_warp; tile < num_tiles; tile += warps_per_grid) {
        const size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;

        if (warp_id == 0) {
            pipe.producer_acquire();
            for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
                A_tile[idx] = A_global[offset + idx];
                B_tile[idx] = B_global[offset + idx];
            }
            pipe.producer_commit();
        }

        if (warp_id == 1) {
            pipe.consumer_wait();
            compute_full_tile(A_tile, B_tile, C_tile, lane_id);
            pipe.consumer_release();
        }

        if (warp_id == 2) {
            pipe.consumer_wait();
            for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
                C_global[offset + idx] = C_tile[idx];
            }
            pipe.consumer_release();
        }
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
    dim3 grid(std::min(tiles, 64));
    size_t shared_bytes = 3 * TILE_ELEMS * sizeof(float);

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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
}  // namespace

int main() {
    run_optimized(128);
    return 0;
}
