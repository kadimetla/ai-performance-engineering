/*
 * Optimized FP4 Hardware Kernel for Blackwell using CUDA 13 fp4 intrinsics.
 *
 * Uses cuda_fp4.h conversions instead of manual quantization to exercise
 * native FP4 support (E2M1). Baseline manual variant lives in
 * baseline_fp4_hardware_kernel.cu.
 */

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// FP4 GEMM using cuda_fp4.h conversions (E2M1)
template<int M, int N, int K>
__global__ void fp4_intrinsics_gemm_kernel(
    const float* A_float,
    const float* B_float,
    float* C,
    int M_dim, int N_dim, int K_dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M_dim || col >= N_dim) return;

    float sum = 0.0f;
    for (int k = 0; k < K_dim; ++k) {
        __nv_fp4_e2m1 a_fp4 = __nv_fp4_e2m1(A_float[row * K_dim + k]);
        __nv_fp4_e2m1 b_fp4 = __nv_fp4_e2m1(B_float[k * N_dim + col]);
        float a = static_cast<float>(a_fp4);
        float b = static_cast<float>(b_fp4);
        sum += a * b;
    }
    C[row * N_dim + col] = sum;
}

int main() {
    std::cout << "=== Optimized FP4 Intrinsics Kernel Benchmark ===" << std::endl;
    const int M = 1024, N = 1024, K = 1024;
    const size_t size_A = M * K;
    const size_t size_B = K * N;
    const size_t size_C = M * N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    for (size_t i = 0; i < size_A; ++i) h_A[i] = (rand() % 200 - 100) / 100.0f;
    for (size_t i = 0; i < size_B; ++i) h_B[i] = (rand() % 200 - 100) / 100.0f;

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    auto start = std::chrono::high_resolution_clock::now();
    fp4_intrinsics_gemm_kernel<M, N, K><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Kernel time: " << ms << " ms" << std::endl;

    // Validate a single element vs CPU float GEMM for sanity
    float h_ref = 0.0f;
    for (int k = 0; k < K; ++k) {
        h_ref += h_A[0 * K + k] * h_B[k * N + 0];
    }
    float h_out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_out, d_C, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Reference C[0,0] (float GEMM): " << h_ref << std::endl;
    std::cout << "FP4 intrinsics C[0,0]:       " << h_out << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
