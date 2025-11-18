#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <algorithm>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;
using std::max;
using std::min;

namespace {
constexpr int BASELINE_BLOCK = 16;
constexpr int PIPE_TILE_M = 64;
constexpr int PIPE_TILE_N = 64;
constexpr int PIPE_TILE_K = 32;
constexpr int PIPE_THREADS = 3 * 32;  // load, compute, store warps
constexpr int CLUSTER_THREADS = 3 * 32;
constexpr int TMA_THREADS = 128;       // use more threads to feed memcpy_async

__global__ void baseline_kernel(const half* __restrict__ A,
                                const half* __restrict__ B,
                                half* __restrict__ C,
                                int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    float lhs = __half2float(A[row * K + k]);
    float rhs = __half2float(B[k * N + col]);
    acc += lhs * rhs;
  }
  C[row * N + col] = __float2half(acc);
}

__device__ inline void zero_tile(float* tile, int elements, int lane_id) {
  for (int idx = lane_id; idx < elements; idx += warpSize) {
    tile[idx] = 0.0f;
  }
}

__device__ inline void store_tile(const float* __restrict__ tile,
                                  half* __restrict__ C,
                                  int ld_c,
                                  int block_row,
                                  int block_col,
                                  int rows,
                                  int cols,
                                  int lane_id) {
  for (int row = lane_id; row < rows; row += warpSize) {
    for (int col = 0; col < cols; ++col) {
      const int global_row = block_row + row;
      const int global_col = block_col + col;
      C[global_row * ld_c + global_col] = __float2half(tile[row * PIPE_TILE_N + col]);
    }
  }
}

__device__ inline void compute_rows(const half* __restrict__ A_tile,
                                    const half* __restrict__ B_tile,
                                    float* __restrict__ accum_tile,
                                    int rows,
                                    int cols,
                                    int k_extent,
                                    int lane_id) {
  for (int row = lane_id; row < rows; row += warpSize) {
    for (int col = 0; col < cols; ++col) {
      float acc = accum_tile[row * PIPE_TILE_N + col];
#pragma unroll 4
      for (int k_it = 0; k_it < k_extent; ++k_it) {
        float lhs = __half2float(A_tile[row * PIPE_TILE_K + k_it]);
        float rhs = __half2float(B_tile[k_it * PIPE_TILE_N + col]);
        acc += lhs * rhs;
      }
      accum_tile[row * PIPE_TILE_N + col] = acc;
    }
  }
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void pipeline_prefetch_kernel(const half* __restrict__ A,
                                         const half* __restrict__ B,
                                         half* __restrict__ C,
                                         int M,
                                         int N,
                                         int K) {
  cg::thread_block cta = cg::this_thread_block();
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;
  auto pipe = cuda::make_pipeline(cta, &pipe_state);

  extern __shared__ unsigned char shared_mem[];
  half* A_tile = reinterpret_cast<half*>(shared_mem);
  half* B_tile = A_tile + TILE_M * TILE_K;
  float* C_tile = reinterpret_cast<float*>(B_tile + TILE_K * TILE_N);

  const int block_row = blockIdx.y * TILE_M;
  const int block_col = blockIdx.x * TILE_N;
  const int rows = max(0, min(TILE_M, M - block_row));
  const int cols = max(0, min(TILE_N, N - block_col));
  if (rows <= 0 || cols <= 0) {
    return;
  }

  const int warp_id = threadIdx.x / warpSize;
  const int lane_id = threadIdx.x % warpSize;

  if (warp_id == 2) {
    zero_tile(C_tile, TILE_M * TILE_N, lane_id);
  }
  cta.sync();

  const int total_k_tiles = (K + TILE_K - 1) / TILE_K;
  for (int tile_idx = 0; tile_idx < total_k_tiles; ++tile_idx) {
    const int global_k = tile_idx * TILE_K;
    const int k_extent = min(TILE_K, K - global_k);

    pipe.producer_acquire();
    if (warp_id == 0) {
      for (int idx = lane_id; idx < TILE_M * TILE_K; idx += warpSize) {
        const int i = idx / TILE_K;
        const int j = idx % TILE_K;
        const int global_i = block_row + i;
        const int global_j = global_k + j;
        half val = __float2half(0.0f);
        if (i < rows && j < k_extent && global_i < M && global_j < K) {
          val = A[global_i * K + global_j];
        }
        A_tile[idx] = val;
      }
      for (int idx = lane_id; idx < TILE_K * TILE_N; idx += warpSize) {
        const int i = idx / TILE_N;
        const int j = idx % TILE_N;
        const int global_i = global_k + i;
        const int global_j = block_col + j;
        half val = __float2half(0.0f);
        if (i < k_extent && j < cols && global_i < K && global_j < N) {
          val = B[global_i * N + global_j];
        }
        B_tile[idx] = val;
      }
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    pipe.consumer_release();

    cta.sync();

    if (warp_id == 1) {
      compute_rows(A_tile, B_tile, C_tile, rows, cols, k_extent, lane_id);
    }

    cta.sync();
  }

  if (warp_id == 2) {
    store_tile(C_tile, C, N, block_row, block_col, rows, cols, lane_id);
  }
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void cluster_kernel(const half* __restrict__ A,
                               const half* __restrict__ B,
                               half* __restrict__ C,
                               int M,
                               int N,
                               int K) {
  cg::thread_block cta = cg::this_thread_block();
  cg::cluster_group cluster = cg::this_cluster();
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;
  auto pipe = cuda::make_pipeline(cta, &pipe_state);

  extern __shared__ unsigned char shared_mem[];
  half* A_tile = reinterpret_cast<half*>(shared_mem);
  half* B_tile = A_tile + TILE_M * TILE_K;
  float* C_tile = reinterpret_cast<float*>(B_tile + TILE_K * TILE_N);

  const int warp_id = threadIdx.x / warpSize;
  const int lane_id = threadIdx.x % warpSize;
  const int cluster_rank = cluster.block_rank();
  const int blocks_in_cluster =
      cluster.dim_blocks().x * cluster.dim_blocks().y * cluster.dim_blocks().z;

  const int cluster_dim_x = cluster.dim_blocks().x;
  const int tile_col = blockIdx.x / cluster_dim_x;
  const int tile_row = blockIdx.y;

  const int block_row = tile_row * TILE_M;
  const int block_col = tile_col * TILE_N;
  const int rows = max(0, min(TILE_M, M - block_row));
  const int cols = max(0, min(TILE_N, N - block_col));
  if (rows <= 0 || cols <= 0) {
    return;
  }

  const int rows_per_block = max(1, (rows + blocks_in_cluster - 1) / blocks_in_cluster);
  const int row_begin = min(cluster_rank * rows_per_block, rows);
  const int row_end = min(row_begin + rows_per_block, rows);

  zero_tile(C_tile, TILE_M * TILE_N, lane_id);
  cta.sync();

  const int total_k_tiles = (K + TILE_K - 1) / TILE_K;
  for (int tile_idx = 0; tile_idx < total_k_tiles; ++tile_idx) {
    const int global_k = tile_idx * TILE_K;
    const int k_extent = min(TILE_K, K - global_k);

    if (cluster_rank == 0) {
      pipe.producer_acquire();
      if (warp_id == 0) {
        for (int idx = lane_id; idx < TILE_M * TILE_K; idx += warpSize) {
          const int i = idx / TILE_K;
          const int j = idx % TILE_K;
          const int global_i = block_row + i;
          const int global_j = global_k + j;
          half val = __float2half(0.0f);
          if (i < rows && j < k_extent && global_i < M && global_j < K) {
            val = A[global_i * K + global_j];
          }
          A_tile[idx] = val;
        }
        for (int idx = lane_id; idx < TILE_K * TILE_N; idx += warpSize) {
          const int i = idx / TILE_N;
          const int j = idx % TILE_N;
          const int global_i = global_k + i;
          const int global_j = block_col + j;
          half val = __float2half(0.0f);
          if (i < k_extent && j < cols && global_i < K && global_j < N) {
            val = B[global_i * N + global_j];
          }
          B_tile[idx] = val;
        }
      }
      pipe.producer_commit();
      pipe.consumer_wait();
      pipe.consumer_release();
    }

    cluster.sync();

    const half* A_src = cluster.map_shared_rank(A_tile, 0);
    const half* B_src = cluster.map_shared_rank(B_tile, 0);

    if (warp_id == 1) {
      for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
        for (int col = 0; col < cols; ++col) {
          float acc = C_tile[row * TILE_N + col];
#pragma unroll 4
          for (int k_it = 0; k_it < k_extent; ++k_it) {
            float lhs = __half2float(A_src[row * TILE_K + k_it]);
            float rhs = __half2float(B_src[k_it * TILE_N + col]);
            acc += lhs * rhs;
          }
          C_tile[row * TILE_N + col] = acc;
        }
      }
    }

    cta.sync();
    cluster.sync();
  }

  if (warp_id == 2) {
    for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
      for (int col = 0; col < cols; ++col) {
        const int global_row = block_row + row;
        const int global_col = block_col + col;
        C[global_row * N + global_col] = __float2half(C_tile[row * TILE_N + col]);
      }
    }
  }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 100
// Real TMA path (requires hardware support).
__global__ void tma_prefetch_kernel(const half* __restrict__ A,
                                    const half* __restrict__ B,
                                    half* __restrict__ C,
                                    int M,
                                    int N,
                                    int K) {
  cg::thread_block cta = cg::this_thread_block();
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_thread, 2> pipe_state;
  auto pipe = cuda::make_pipeline(cuda::thread_scope_thread{}, &pipe_state);

  extern __shared__ unsigned char shared_mem[];
  half* A_tile = reinterpret_cast<half*>(shared_mem);
  half* B_tile = A_tile + PIPE_TILE_M * PIPE_TILE_K;
  float* C_tile = reinterpret_cast<float*>(B_tile + PIPE_TILE_K * PIPE_TILE_N);

  const int block_row = blockIdx.y * PIPE_TILE_M;
  const int block_col = blockIdx.x * PIPE_TILE_N;
  const int rows = max(0, min(PIPE_TILE_M, M - block_row));
  const int cols = max(0, min(PIPE_TILE_N, N - block_col));
  if (rows <= 0 || cols <= 0) {
    return;
  }

  if (threadIdx.x == 0) {
    zero_tile(C_tile, PIPE_TILE_M * PIPE_TILE_N, 0);
  }
  cta.sync();

  const int total_k_tiles = (K + PIPE_TILE_K - 1) / PIPE_TILE_K;
  for (int tile_idx = 0; tile_idx < total_k_tiles; ++tile_idx) {
    const int global_k = tile_idx * PIPE_TILE_K;
    const int k_extent = min(PIPE_TILE_K, K - global_k);

    // Producer threads issue memcpy_async; use thread-level pipeline for TMA.
    if (threadIdx.x < 32) {
      pipe.producer_acquire();
      cuda::memcpy_async(cta, A_tile, A + block_row * K + global_k,
                         rows * k_extent * sizeof(half), pipe);
      cuda::memcpy_async(cta, B_tile, B + global_k * N + block_col,
                         k_extent * cols * sizeof(half), pipe);
      pipe.producer_commit();
    }

    pipe.consumer_wait();
    cta.sync();

    // Compute
    const int lane_id = threadIdx.x % warpSize;
    compute_rows(A_tile, B_tile, C_tile, rows, cols, k_extent, lane_id);

    cta.sync();
    pipe.consumer_release();
  }

  // Store
  const int lane_id = threadIdx.x % warpSize;
  store_tile(C_tile, C, N, block_row, block_col, rows, cols, lane_id);
}
#endif

inline bool cluster_launch_supported_impl() {
  int device = at::cuda::current_device();
  int value = 0;
#ifdef cudaDevAttrClusterLaunch
  cudaDeviceGetAttribute(&value, cudaDevAttrClusterLaunch, device);
#endif
  return value != 0;
}

inline bool tma_supported_impl() {
  int device = at::cuda::current_device();
#ifdef cudaDevAttrTensorMemoryAccessSupported
  int value = 0;
  if (cudaDeviceGetAttribute(&value, cudaDevAttrTensorMemoryAccessSupported, device) ==
      cudaSuccess) {
    return value != 0;
  }
#endif
  // Fallback: if the attribute is unavailable, assume not supported.
  return false;
}

void launch_baseline(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  dim3 block(BASELINE_BLOCK, BASELINE_BLOCK);
  dim3 grid((b.size(1) + block.x - 1) / block.x,
            (a.size(0) + block.y - 1) / block.y);
  auto stream = at::cuda::getCurrentCUDAStream();
  baseline_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      reinterpret_cast<half*>(c.data_ptr<at::Half>()), a.size(0),
      b.size(1), a.size(1));
  AT_CUDA_CHECK(cudaGetLastError());
}

void launch_pipeline(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  dim3 block(PIPE_THREADS);
  dim3 grid((b.size(1) + PIPE_TILE_N - 1) / PIPE_TILE_N,
            (a.size(0) + PIPE_TILE_M - 1) / PIPE_TILE_M);
  const size_t shared_bytes =
      (PIPE_TILE_M * PIPE_TILE_K + PIPE_TILE_K * PIPE_TILE_N) * sizeof(half) +
      PIPE_TILE_M * PIPE_TILE_N * sizeof(float);
  cudaFuncSetAttribute(
      pipeline_prefetch_kernel<PIPE_TILE_M, PIPE_TILE_N, PIPE_TILE_K>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  auto stream = at::cuda::getCurrentCUDAStream();
  pipeline_prefetch_kernel<PIPE_TILE_M, PIPE_TILE_N, PIPE_TILE_K>
      <<<grid, block, shared_bytes, stream>>>(
          reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<half*>(c.data_ptr<at::Half>()), a.size(0),
          b.size(1), a.size(1));
  AT_CUDA_CHECK(cudaGetLastError());
}

void launch_cluster(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  TORCH_CHECK(
      cluster_launch_supported_impl(),
      "Grace-Blackwell cluster launch requires cudaDevAttrClusterLaunch=1.");

  const int device = at::cuda::current_device();
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  const int cluster_dim = prop.major >= 10 ? 8 : 4;

  const int tiles_x = (b.size(1) + PIPE_TILE_N - 1) / PIPE_TILE_N;
  const int tiles_y = (a.size(0) + PIPE_TILE_M - 1) / PIPE_TILE_M;

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(tiles_x * cluster_dim, tiles_y, 1);
  cfg.blockDim = dim3(CLUSTER_THREADS, 1, 1);
  cfg.dynamicSmemBytes =
      (PIPE_TILE_M * PIPE_TILE_K + PIPE_TILE_K * PIPE_TILE_N) * sizeof(half) +
      PIPE_TILE_M * PIPE_TILE_N * sizeof(float);

  cudaLaunchAttribute cluster_attr{};
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = cluster_dim;
  cluster_attr.val.clusterDim.y = 1;
  cluster_attr.val.clusterDim.z = 1;
  cfg.attrs = &cluster_attr;
  cfg.numAttrs = 1;

  cudaFuncSetAttribute(cluster_kernel<PIPE_TILE_M, PIPE_TILE_N, PIPE_TILE_K>,
                       cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaFuncSetAttribute(cluster_kernel<PIPE_TILE_M, PIPE_TILE_N, PIPE_TILE_K>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       cfg.dynamicSmemBytes);

  auto stream = at::cuda::getCurrentCUDAStream();
  cfg.stream = stream;

  const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
  const half* B_ptr = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
  half* C_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());
  int M = a.size(0);
  int N = b.size(1);
  int K = a.size(1);

  AT_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, cluster_kernel<PIPE_TILE_M, PIPE_TILE_N, PIPE_TILE_K>, A_ptr, B_ptr,
      C_ptr, M, N, K));
  AT_CUDA_CHECK(cudaGetLastError());
}

void launch_tma(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 100
  if (!tma_supported_impl()) {
    TORCH_CHECK(false,
                "Blackwell TMA unavailable on this device; "
                "use optimized_blackwell_matmul_pseudo instead.");
  }
  dim3 block(TMA_THREADS);
  dim3 grid((b.size(1) + PIPE_TILE_N - 1) / PIPE_TILE_N,
            (a.size(0) + PIPE_TILE_M - 1) / PIPE_TILE_M);
  const size_t shared_bytes =
      (PIPE_TILE_M * PIPE_TILE_K + PIPE_TILE_K * PIPE_TILE_N) * sizeof(half) +
      PIPE_TILE_M * PIPE_TILE_N * sizeof(float);
  cudaFuncSetAttribute(tma_prefetch_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shared_bytes);
  auto stream = at::cuda::getCurrentCUDAStream();
  tma_prefetch_kernel<<<grid, block, shared_bytes, stream>>>(
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      reinterpret_cast<half*>(c.data_ptr<at::Half>()), a.size(0), b.size(1),
      a.size(1));
  AT_CUDA_CHECK(cudaGetLastError());
#else
  TORCH_CHECK(false,
              "Blackwell TMA unavailable in this build; recompile with SM100 support.");
#endif
}

torch::Tensor run_kernel(torch::Tensor a,
                         torch::Tensor b,
                         void (*launcher)(torch::Tensor, torch::Tensor, torch::Tensor)) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "expected 2D tensors");
  TORCH_CHECK(a.size(1) == b.size(0), "incompatible shapes");
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "tensors must live on CUDA");
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16,
              "use float16 tensors");
  auto c = torch::empty({a.size(0), b.size(1)}, a.options());
  launcher(a.contiguous(), b.contiguous(), c);
  return c;
}

}  // namespace

TORCH_LIBRARY(grace_blackwell_capstone, m) {
  m.def("baseline_blackwell_matmul(Tensor a, Tensor b) -> Tensor");
  m.def("optimized_blackwell_matmul_tma(Tensor a, Tensor b) -> Tensor");
  m.def("optimized_blackwell_matmul_pipeline(Tensor a, Tensor b) -> Tensor");
  m.def("optimized_blackwell_matmul_cluster(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(grace_blackwell_capstone, CUDA, m) {
  m.impl("baseline_blackwell_matmul",
         [](torch::Tensor a, torch::Tensor b) {
           return run_kernel(a, b, launch_baseline);
         });
  m.impl("optimized_blackwell_matmul_tma",
         [](torch::Tensor a, torch::Tensor b) {
           return run_kernel(a, b, launch_pipeline);
         });
  m.impl("optimized_blackwell_matmul_pipeline",
         [](torch::Tensor a, torch::Tensor b) {
           return run_kernel(a, b, launch_pipeline);
         });
  m.impl("optimized_blackwell_matmul_cluster",
         [](torch::Tensor a, torch::Tensor b) {
           return run_kernel(a, b, launch_cluster);
         });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("baseline_blackwell_matmul", [](torch::Tensor a, torch::Tensor b) {
    return run_kernel(a, b, launch_baseline);
  });
  m.def("optimized_blackwell_matmul_pseudo", [](torch::Tensor a, torch::Tensor b) {
    return run_kernel(a, b, launch_pipeline);
  });
  m.def("optimized_blackwell_matmul_tma",
        [](torch::Tensor a, torch::Tensor b) {
          TORCH_CHECK(
              tma_supported_impl(),
              "Blackwell TMA unavailable on this device; use optimized_blackwell_matmul_pseudo instead.");
          return run_kernel(a, b, launch_pipeline);
        });
  m.def("optimized_blackwell_matmul_cluster",
        [](torch::Tensor a, torch::Tensor b) {
          return run_kernel(a, b, launch_cluster);
        });
  m.def("cluster_launch_supported", []() { return cluster_launch_supported_impl(); });
  m.def("tma_supported", []() { return tma_supported_impl(); });
}
