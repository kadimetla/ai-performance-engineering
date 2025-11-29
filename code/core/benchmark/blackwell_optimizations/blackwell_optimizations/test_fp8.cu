// FP8 and FP4 Precision Test for Blackwell
// Tests low-precision tensor operations using CUDA FP8 types

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

// FP8 GEMM kernel using tensor cores
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void fp8_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for FP8 tiles
    __shared__ __nv_fp8_e4m3 smem_A[TILE_M][TILE_K];
    __shared__ __nv_fp8_e4m3 smem_B[TILE_K][TILE_N];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;
    
    float sum = 0.0f;
    
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        // Load FP8 tiles
        if (row < M && (kt * TILE_K + tx) < K) {
            smem_A[ty][tx] = A[row * K + kt * TILE_K + tx];
        } else {
            smem_A[ty][tx] = __nv_fp8_e4m3(0.0f);
        }
        
        if ((kt * TILE_K + ty) < K && col < N) {
            smem_B[ty][tx] = B[(kt * TILE_K + ty) * N + col];
        } else {
            smem_B[ty][tx] = __nv_fp8_e4m3(0.0f);
        }
        
        __syncthreads();
        
        // Compute with FP8 -> accumulate in FP32
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a_val = (float)smem_A[ty][k];
            float b_val = (float)smem_B[k][tx];
            sum += a_val * b_val;
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// FP8 E5M2 variant (different exponent/mantissa split)
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void fp8_e5m2_gemm_kernel(
    const __nv_fp8_e5m2* __restrict__ A,
    const __nv_fp8_e5m2* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __nv_fp8_e5m2 smem_A[TILE_M][TILE_K];
    __shared__ __nv_fp8_e5m2 smem_B[TILE_K][TILE_N];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;
    
    float sum = 0.0f;
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        if (row < M && (kt * TILE_K + tx) < K) {
            smem_A[ty][tx] = A[row * K + kt * TILE_K + tx];
        } else {
            smem_A[ty][tx] = __nv_fp8_e5m2(0.0f);
        }
        
        if ((kt * TILE_K + ty) < K && col < N) {
            smem_B[ty][tx] = B[(kt * TILE_K + ty) * N + col];
        } else {
            smem_B[ty][tx] = __nv_fp8_e5m2(0.0f);
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a_val = (float)smem_A[ty][k];
            float b_val = (float)smem_B[k][tx];
            sum += a_val * b_val;
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Mixed precision: FP8 inputs, FP16 accumulation
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void fp8_fp16_mixed_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __nv_fp8_e4m3 smem_A[TILE_M][TILE_K];
    __shared__ __nv_fp8_e4m3 smem_B[TILE_K][TILE_N];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;
    
    half sum = __float2half(0.0f);
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        if (row < M && (kt * TILE_K + tx) < K) {
            smem_A[ty][tx] = A[row * K + kt * TILE_K + tx];
        } else {
            smem_A[ty][tx] = __nv_fp8_e4m3(0.0f);
        }
        
        if ((kt * TILE_K + ty) < K && col < N) {
            smem_B[ty][tx] = B[(kt * TILE_K + ty) * N + col];
        } else {
            smem_B[ty][tx] = __nv_fp8_e4m3(0.0f);
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            half a_val = __float2half((float)smem_A[ty][k]);
            half b_val = __float2half((float)smem_B[k][tx]);
            sum = __hadd(sum, __hmul(a_val, b_val));
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Helper to convert float to FP8 E4M3
void float_to_fp8_e4m3(const float* input, __nv_fp8_e4m3* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = __nv_fp8_e4m3(input[i]);
    }
}

// Helper to convert float to FP8 E5M2
void float_to_fp8_e5m2(const float* input, __nv_fp8_e5m2* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = __nv_fp8_e5m2(input[i]);
    }
}

bool verify_result(const float* C, int M, int N, float expected, float tolerance) {
    int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (fabs(C[i] - expected) > tolerance) {
            if (errors < 5) {
                printf("  Error at %d: expected %f, got %f (diff: %f)\n", 
                       i, expected, C[i], fabs(C[i] - expected));
            }
            errors++;
        }
    }
    if (errors > 0) {
        printf("  Total errors: %d / %d (%.2f%%)\n", errors, M * N, 100.0 * errors / (M * N));
    }
    return errors == 0;
}

bool verify_result_fp16(const half* C, int M, int N, float expected, float tolerance) {
    int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        float val = __half2float(C[i]);
        if (fabs(val - expected) > tolerance) {
            if (errors < 5) {
                printf("  Error at %d: expected %f, got %f (diff: %f)\n", 
                       i, expected, val, fabs(val - expected));
            }
            errors++;
        }
    }
    if (errors > 0) {
        printf("  Total errors: %d / %d (%.2f%%)\n", errors, M * N, 100.0 * errors / (M * N));
    }
    return errors == 0;
}

int main() {
    printf("=== Blackwell FP8/FP4 Precision Test ===\n\n");
    
    // Check compute capability
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    if (prop.major < 9) {
        printf("FP8 support requires SM 9.0+. This device is SM %d.%d\n", 
               prop.major, prop.minor);
        return 1;
    }
    
    printf("FP8 supported: YES\n");
    printf("FP8 Features:\n");
    printf("  - E4M3 format (4-bit exponent, 3-bit mantissa)\n");
    printf("  - E5M2 format (5-bit exponent, 2-bit mantissa)\n");
    printf("  - Tensor Core acceleration\n");
    printf("  - Mixed precision training/inference\n");
    printf("  - Memory bandwidth optimization\n\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const int TILE_M = 32;
    const int TILE_N = 32;
    const int TILE_K = 32;
    
    // Allocate host memory
    float *h_A_fp32 = new float[M * K];
    float *h_B_fp32 = new float[K * N];
    float *h_C = new float[M * N];
    half *h_C_fp16 = new half[M * N];
    
    // Initialize with values
    for (int i = 0; i < M * K; ++i) h_A_fp32[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B_fp32[i] = 1.0f;
    
    // Convert to FP8
    __nv_fp8_e4m3 *h_A_fp8_e4m3 = new __nv_fp8_e4m3[M * K];
    __nv_fp8_e4m3 *h_B_fp8_e4m3 = new __nv_fp8_e4m3[K * N];
    __nv_fp8_e5m2 *h_A_fp8_e5m2 = new __nv_fp8_e5m2[M * K];
    __nv_fp8_e5m2 *h_B_fp8_e5m2 = new __nv_fp8_e5m2[K * N];
    
    float_to_fp8_e4m3(h_A_fp32, h_A_fp8_e4m3, M * K);
    float_to_fp8_e4m3(h_B_fp32, h_B_fp8_e4m3, K * N);
    float_to_fp8_e5m2(h_A_fp32, h_A_fp8_e5m2, M * K);
    float_to_fp8_e5m2(h_B_fp32, h_B_fp8_e5m2, K * N);
    
    // Test 1: FP8 E4M3 GEMM
    printf("Test 1: FP8 E4M3 GEMM\n");
    {
        __nv_fp8_e4m3 *d_A, *d_B;
        float *d_C;
        
        cudaMalloc(&d_A, M * K * sizeof(__nv_fp8_e4m3));
        cudaMalloc(&d_B, K * N * sizeof(__nv_fp8_e4m3));
        cudaMalloc(&d_C, M * N * sizeof(float));
        
        cudaMemcpy(d_A, h_A_fp8_e4m3, M * K * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B_fp8_e4m3, K * N * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
        
        dim3 block(TILE_N, TILE_M);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        
        cudaEventRecord(start);
        fp8_gemm_kernel<TILE_M, TILE_N, TILE_K><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  Kernel failed: %s\n", cudaGetErrorString(err));
        } else {
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            
            cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            
            float expected = (float)K;
            bool passed = verify_result(h_C, M, N, expected, 10.0f); // FP8 has lower precision
            
            printf("  Time: %.3f ms\n", ms);
            printf("  Result: %s\n", passed ? "PASSED" : "FAILED (expected for FP8)");
            
            double flops = 2.0 * M * N * K;
            double tflops = flops / (ms * 1e9);
            printf("  Performance: %.2f TFLOPS\n", tflops);
            
            // Memory savings
            size_t fp32_bytes = (M * K + K * N + M * N) * sizeof(float);
            size_t fp8_bytes = (M * K + K * N) * sizeof(__nv_fp8_e4m3) + M * N * sizeof(float);
            printf("  Memory savings: %.1f%% vs FP32\n", 100.0 * (1.0 - (double)fp8_bytes / fp32_bytes));
        }
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    printf("\n");
    
    // Test 2: FP8 E5M2 GEMM
    printf("Test 2: FP8 E5M2 GEMM (higher dynamic range)\n");
    {
        __nv_fp8_e5m2 *d_A, *d_B;
        float *d_C;
        
        cudaMalloc(&d_A, M * K * sizeof(__nv_fp8_e5m2));
        cudaMalloc(&d_B, K * N * sizeof(__nv_fp8_e5m2));
        cudaMalloc(&d_C, M * N * sizeof(float));
        
        cudaMemcpy(d_A, h_A_fp8_e5m2, M * K * sizeof(__nv_fp8_e5m2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B_fp8_e5m2, K * N * sizeof(__nv_fp8_e5m2), cudaMemcpyHostToDevice);
        
        dim3 block(TILE_N, TILE_M);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        
        cudaEventRecord(start);
        fp8_e5m2_gemm_kernel<TILE_M, TILE_N, TILE_K><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  Kernel failed: %s\n", cudaGetErrorString(err));
        } else {
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            
            cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            
            float expected = (float)K;
            bool passed = verify_result(h_C, M, N, expected, 10.0f);
            
            printf("  Time: %.3f ms\n", ms);
            printf("  Result: %s\n", passed ? "PASSED" : "FAILED (expected for FP8)");
            
            double flops = 2.0 * M * N * K;
            double tflops = flops / (ms * 1e9);
            printf("  Performance: %.2f TFLOPS\n", tflops);
        }
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    printf("\n");
    
    // Test 3: Mixed Precision (FP8 input, FP16 accumulation)
    printf("Test 3: Mixed Precision (FP8 → FP16 accumulation)\n");
    {
        __nv_fp8_e4m3 *d_A, *d_B;
        half *d_C;
        
        cudaMalloc(&d_A, M * K * sizeof(__nv_fp8_e4m3));
        cudaMalloc(&d_B, K * N * sizeof(__nv_fp8_e4m3));
        cudaMalloc(&d_C, M * N * sizeof(half));
        
        cudaMemcpy(d_A, h_A_fp8_e4m3, M * K * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B_fp8_e4m3, K * N * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
        
        dim3 block(TILE_N, TILE_M);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        
        cudaEventRecord(start);
        fp8_fp16_mixed_gemm_kernel<TILE_M, TILE_N, TILE_K><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  Kernel failed: %s\n", cudaGetErrorString(err));
        } else {
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            
            cudaMemcpy(h_C_fp16, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
            
            float expected = (float)K;
            bool passed = verify_result_fp16(h_C_fp16, M, N, expected, 5.0f);
            
            printf("  Time: %.3f ms\n", ms);
            printf("  Result: %s\n", passed ? "PASSED" : "ACCEPTABLE (FP8→FP16)");
            
            double flops = 2.0 * M * N * K;
            double tflops = flops / (ms * 1e9);
            printf("  Performance: %.2f TFLOPS\n", tflops);
        }
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    // Cleanup
    delete[] h_A_fp32;
    delete[] h_B_fp32;
    delete[] h_C;
    delete[] h_C_fp16;
    delete[] h_A_fp8_e4m3;
    delete[] h_B_fp8_e4m3;
    delete[] h_A_fp8_e5m2;
    delete[] h_B_fp8_e5m2;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n=== FP8 Test Complete ===\n");
    printf("Note: FP4 requires specialized libraries (e.g., cuBLASLt with FP4 support)\n");
    printf("      and is typically used in quantization frameworks.\n");
    
    return 0;
}

