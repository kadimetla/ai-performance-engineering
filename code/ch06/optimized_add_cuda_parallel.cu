// add_parallel.cu
// Efficient CUDA vector addition (Chapter 6 best practice).

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

constexpr int N = 1'000'000;

__global__ void addParallel(const float* A, const float* B, float* C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    C[idx] = A[idx] + B[idx];
  }
}

int main() {
    NVTX_RANGE("main");
  float *h_A, *h_B, *h_C;
  cudaMallocHost(&h_A, N * sizeof(float));
  cudaMallocHost(&h_B, N * sizeof(float));
  cudaMallocHost(&h_C, N * sizeof(float));

  for (int i = 0; i < N; ++i) {
      NVTX_RANGE("setup");
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(2 * i);
  }

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMallocAsync(&d_A, N * sizeof(float), stream);
  cudaMallocAsync(&d_B, N * sizeof(float), stream);
  cudaMallocAsync(&d_C, N * sizeof(float), stream);

  cudaMemcpyAsync(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice, stream);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  
  cudaEventRecord(start, stream);
  addParallel<<<blocks, threads, 0, stream>>>(d_A, d_B, d_C, N);
  cudaEventRecord(stop, stream);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeAsync(d_A, stream);
    cudaFreeAsync(d_B, stream);
    cudaFreeAsync(d_C, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    return 1;
  }

  cudaMemcpyAsync(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  
  // Calculate elapsed time
  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  
  printf("C[0]=%.1f, C[N-1]=%.1f\n", h_C[0], h_C[N - 1]);
  printf("TIME_MS: %.4f\n", elapsed_ms);

#ifdef VERIFY
  double checksum = 0.0;
  for (int i = 0; i < N; ++i) {
    checksum += std::abs(h_C[i]);
  }
  VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFreeAsync(d_A, stream);
  cudaFreeAsync(d_B, stream);
  cudaFreeAsync(d_C, stream);
  cudaStreamDestroy(stream);
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);
  return 0;
}
