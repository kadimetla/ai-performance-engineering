// Optimized HBM binary with vectorized accesses.

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "hbm_common.cuh"
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

using namespace ch08;

int main() {
    NVTX_RANGE("main");
    const int rows = 4096;
    const int cols = 2048;
    const size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);

    std::vector<float> host_matrix(rows * cols);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& value : host_matrix) {
        NVTX_RANGE("setup");
        value = dist(gen);
    }

    float* d_row = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_row, bytes);
    cudaMalloc(&d_output, rows * sizeof(float));
    cudaMemcpy(d_row, host_matrix.data(), bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 5; ++i) {
        NVTX_RANGE("iteration");
        launch_hbm_vectorized(d_row, d_output, rows, cols, 0);
    }
    cudaDeviceSynchronize();

    const int iterations = 30;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("iteration");
        launch_hbm_vectorized(d_row, d_output, rows, cols, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    const float avg_ms = total_ms / iterations;
    std::cout << "Optimized HBM: " << avg_ms << " ms\n";
    std::printf("TIME_MS: %.6f\n", avg_ms);

#ifdef VERIFY
    std::vector<float> host_output(rows, 0.0f);
    cudaMemcpy(host_output.data(), d_output, rows * sizeof(float), cudaMemcpyDeviceToHost);
    double checksum = 0.0;
    for (float v : host_output) {
        checksum += std::abs(v);
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_row);
    cudaFree(d_output);
    return 0;
}
