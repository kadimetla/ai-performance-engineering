// optimized_kernel_fusion.cu - Fused kernel with CUDA graphs (optimized)
// Demonstrates kernel fusion using CUDA graphs to reduce memory traffic

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include "../core/common/headers/profiling_helpers.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(status));                              \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Fused kernel: Single kernel combining all operations
__global__ void kernel_fused(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        val = val + 1.0f;
        val = val * 2.0f;
        val = sqrtf(val);
        data[idx] = val;
    }
}

int main() {
    constexpr int N = 10'000'000;
    constexpr int ITERATIONS = 100;
    
    std::vector<float> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }
    
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // Capture graph with fused kernel
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    kernel_fused<<<grid, block, 0, stream>>>(d_data, N);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Measure
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start, stream));
    {
        PROFILE_KERNEL_LAUNCH("optimized_kernel_fusion");
        for (int i = 0; i < ITERATIONS; ++i) {
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    printf("Optimized (fused + graph): %.3f ms\n", ms);
    
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    
    return 0;
}

