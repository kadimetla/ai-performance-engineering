// Chapter 10: Book-aligned optimized cluster pipeline using cuda::pipeline + DSMEM broadcast.
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
constexpr int THREADS_PER_BLOCK = 3 * 32;

__device__ void compute_rows(const float* __restrict__ A_tile,
                             const float* __restrict__ B_tile,
                             float* __restrict__ C_tile,
                             int row_begin,
                             int row_end,
                             int lane_id) {
    for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
        for (int col = 0; col < TILE_SIZE; ++col) {
            float acc = 0.0f;
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                acc += A_tile[row * TILE_SIZE + k] * B_tile[k * TILE_SIZE + col];
            }
            C_tile[row * TILE_SIZE + col] = acc;
        }
    }
}

__global__ void optimized_cluster_kernel(const float* __restrict__ A_global,
                                         const float* __restrict__ B_global,
                                         float* __restrict__ C_global,
                                         int num_tiles) {
    cg::thread_block cta = cg::this_thread_block();
    cg::cluster_group cluster = cg::this_cluster();

    extern __shared__ float shared_mem[];
    float* A_tile_local = shared_mem;
    float* B_tile_local = A_tile_local + TILE_ELEMS;
    float* C_tile_local = B_tile_local + TILE_ELEMS;

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
    const int cluster_rank = cluster.block_rank();
    const int blocks_in_cluster = cluster.dim_blocks().x * cluster.dim_blocks().y * cluster.dim_blocks().z;

    const int rows_per_block = (TILE_SIZE + blocks_in_cluster - 1) / blocks_in_cluster;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;

        if (cluster_rank == 0 && warp_id == 0) {
            pipe.producer_acquire();
            for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
                A_tile_local[idx] = A_global[offset + idx];
                B_tile_local[idx] = B_global[offset + idx];
            }
            pipe.producer_commit();
            pipe.consumer_wait();
            pipe.consumer_release();
        }

        cluster.sync();

        const float* A_src = cluster.map_shared_rank(A_tile_local, 0);
        const float* B_src = cluster.map_shared_rank(B_tile_local, 0);

        const int row_begin = min(cluster_rank * rows_per_block, TILE_SIZE);
        const int row_end = min(row_begin + rows_per_block, TILE_SIZE);

        if (warp_id == 1) {
            compute_rows(A_src, B_src, C_tile_local, row_begin, row_end, lane_id);
        }

        cta.sync();

        if (warp_id == 2) {
            for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
                for (int col = 0; col < TILE_SIZE; ++col) {
                    C_global[offset + row * TILE_SIZE + col] = C_tile_local[row * TILE_SIZE + col];
                }
            }
        }

        cluster.sync();
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

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    int cluster_launch = 0;
#ifdef cudaDevAttrClusterLaunch
    cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, 0);
#endif
    if (!cluster_launch && prop.major < 9) {
        printf("optimized_warp_specialized_cluster_pipeline requires cluster-capable GPU.\n");
        return;
    }

    const int cluster_size = prop.major >= 10 ? 8 : 4;

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(cluster_size);
    cfg.blockDim = dim3(THREADS_PER_BLOCK);
    cfg.dynamicSmemBytes = 3 * TILE_ELEMS * sizeof(float);

    cudaLaunchAttribute cluster_attr{};
    cluster_attr.id = cudaLaunchAttributeClusterDimension;
    cluster_attr.val.clusterDim.x = cluster_size;
    cluster_attr.val.clusterDim.y = 1;
    cluster_attr.val.clusterDim.z = 1;
    cfg.attrs = &cluster_attr;
    cfg.numAttrs = 1;

    cudaFuncSetAttribute(optimized_cluster_kernel,
                         cudaFuncAttributeNonPortableClusterSizeAllowed,
                         1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaLaunchKernelEx(&cfg, optimized_cluster_kernel, d_A, d_B, d_C, tiles);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (float v : h_C) checksum += v;

    printf("optimized_warp_specialized_cluster_pipeline: %d tiles, %.3f ms, checksum %.3f\n",
           tiles, ms, checksum / h_C.size());

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
}  // namespace

int main() {
    run_optimized(8);
    return 0;
}
