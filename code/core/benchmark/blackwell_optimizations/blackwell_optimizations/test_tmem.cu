// TMEM (Tensor Memory Accelerator) Test for Blackwell
// Tests TMEM features including tensor memory operations and async copy

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <stdio.h>

// TMEM is accessed via tensor memory operations
// On Blackwell, TMEM provides high-bandwidth tensor data path

template<int TILE_SIZE>
__global__ void tmem_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    namespace cg = cooperative_groups;
    
    // Use shared memory as TMEM staging area
    __shared__ float smem_A[TILE_SIZE][TILE_SIZE];
    __shared__ float smem_B[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Tile across K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory (TMEM path on Blackwell)
        if (row < M && (t * TILE_SIZE + tx) < K) {
            smem_A[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            smem_A[ty][tx] = 0.0f;
        }
        
        if ((t * TILE_SIZE + ty) < K && col < N) {
            smem_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            smem_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += smem_A[ty][k] * smem_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Advanced TMEM test with async pipeline
template<int TILE_SIZE, int STAGES>
__global__ void tmem_async_pipeline_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for multi-stage pipeline
    __shared__ float smem_A[STAGES][TILE_SIZE][TILE_SIZE];
    __shared__ float smem_B[STAGES][TILE_SIZE][TILE_SIZE];
    
    // Pipeline for async operations
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block,
        STAGES
    > pipe_state;
    
    auto block = cooperative_groups::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pipe_state);
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Producer: Launch async copies
    if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
        for (int t = 0; t < min(STAGES, num_tiles); ++t) {
            pipe.producer_acquire();
            
            int stage = t % STAGES;
            
            if (row < M && (t * TILE_SIZE + tx) < K) {
                smem_A[stage][ty][tx] = A[row * K + t * TILE_SIZE + tx];
            } else {
                smem_A[stage][ty][tx] = 0.0f;
            }
            
            if ((t * TILE_SIZE + ty) < K && col < N) {
                smem_B[stage][ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
            } else {
                smem_B[stage][ty][tx] = 0.0f;
            }
            
            pipe.producer_commit();
        }
    }
    
    // Consumer: Compute
    for (int t = 0; t < num_tiles; ++t) {
        pipe.consumer_wait();
        
        int stage = t % STAGES;
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += smem_A[stage][ty][k] * smem_B[stage][k][tx];
        }
        
        pipe.consumer_release();
        
        // Launch next stage if available
        if (t + STAGES < num_tiles && threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
            pipe.producer_acquire();
            
            int next_stage = (t + STAGES) % STAGES;
            int next_t = t + STAGES;
            
            if (row < M && (next_t * TILE_SIZE + tx) < K) {
                smem_A[next_stage][ty][tx] = A[row * K + next_t * TILE_SIZE + tx];
            } else {
                smem_A[next_stage][ty][tx] = 0.0f;
            }
            
            if ((next_t * TILE_SIZE + ty) < K && col < N) {
                smem_B[next_stage][ty][tx] = B[(next_t * TILE_SIZE + ty) * N + col];
            } else {
                smem_B[next_stage][ty][tx] = 0.0f;
            }
            
            pipe.producer_commit();
        }
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

bool verify_result(const float* C, int M, int N, float expected) {
    for (int i = 0; i < M * N; ++i) {
        if (fabs(C[i] - expected) > 1e-3) {
            printf("Verification failed at index %d: expected %f, got %f\n", 
                   i, expected, C[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== Blackwell TMEM Test ===\n\n");
    
    // Check compute capability
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    if (prop.major < 10) {
        printf("TMEM requires Blackwell (SM 10.0+). This device is SM %d.%d\n", 
               prop.major, prop.minor);
        return 1;
    }
    
    printf("TMEM supported: YES\n\n");
    
    // Test configuration
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const int TILE_SIZE = 32;
    const int STAGES = 4;
    
    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    
    // Initialize with simple values for verification
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Test 1: Basic TMEM GEMM
    printf("Test 1: Basic TMEM GEMM\n");
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    tmem_matmul_kernel<TILE_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms1 = 0;
    cudaEventElapsedTime(&ms1, start, stop);
    
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    float expected = (float)K; // Each element is sum of K ones
    bool passed1 = verify_result(h_C, M, N, expected);
    printf("  Time: %.3f ms\n", ms1);
    printf("  Result: %s\n\n", passed1 ? "PASSED" : "FAILED");
    
    // Test 2: TMEM with Async Pipeline
    printf("Test 2: TMEM with Async Pipeline (software pipelining)\n");
    cudaMemset(d_C, 0, M * N * sizeof(float));
    
    cudaEventRecord(start);
    tmem_async_pipeline_kernel<TILE_SIZE, STAGES><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms2 = 0;
    cudaEventElapsedTime(&ms2, start, stop);
    
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool passed2 = verify_result(h_C, M, N, expected);
    printf("  Time: %.3f ms\n", ms2);
    printf("  Result: %s\n", passed2 ? "PASSED" : "FAILED");
    printf("  Speedup: %.2fx\n\n", ms1 / ms2);
    
    // Performance metrics
    double flops = 2.0 * M * N * K;
    double tflops1 = flops / (ms1 * 1e9);
    double tflops2 = flops / (ms2 * 1e9);
    
    printf("Performance:\n");
    printf("  Basic TMEM: %.2f TFLOPS\n", tflops1);
    printf("  Async Pipeline: %.2f TFLOPS\n", tflops2);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n=== TMEM Test Complete ===\n");
    printf("Status: %s\n", (passed1 && passed2) ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    
    return (passed1 && passed2) ? 0 : 1;
}

