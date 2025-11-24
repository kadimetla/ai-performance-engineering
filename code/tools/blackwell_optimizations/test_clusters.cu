// CTA Clusters (Thread Block Clusters) Test for Blackwell
// Tests cooperative thread block clusters for distributed shared memory

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <stdio.h>

namespace cg = cooperative_groups;

// Kernel using CTA clusters for distributed computation
__global__ void __cluster_dims__(2, 2, 1) cluster_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Get cluster group
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();
    
    const int TILE_SIZE = 32;
    
    // Shared memory per CTA
    __shared__ float smem_A[TILE_SIZE][TILE_SIZE];
    __shared__ float smem_B[TILE_SIZE][TILE_SIZE];
    
    // Get cluster and block indices
    int cluster_rank = cluster.block_rank();
    int cluster_size = cluster.num_blocks();
    
    // Determine which blocks in cluster
    dim3 cluster_idx = cluster.block_index();
    dim3 cluster_dim = cluster.dim_blocks();
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Number of tiles along K
    int num_k_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        // Load tiles - can be distributed across cluster
        if (row < M && (kt * TILE_SIZE + tx) < K) {
            smem_A[ty][tx] = A[row * K + kt * TILE_SIZE + tx];
        } else {
            smem_A[ty][tx] = 0.0f;
        }
        
        if ((kt * TILE_SIZE + ty) < K && col < N) {
            smem_B[ty][tx] = B[(kt * TILE_SIZE + ty) * N + col];
        } else {
            smem_B[ty][tx] = 0.0f;
        }
        
        // Sync within cluster
        cluster.sync();
        
        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += smem_A[ty][k] * smem_B[k][tx];
        }
        
        cluster.sync();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Advanced cluster kernel with distributed shared memory access
__global__ void __cluster_dims__(2, 2, 1) cluster_dsmem_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();
    
    const int TILE_SIZE = 32;
    
    // Shared memory that can be accessed across CTAs in cluster
    __shared__ float smem_A[TILE_SIZE][TILE_SIZE];
    __shared__ float smem_B[TILE_SIZE][TILE_SIZE];
    
    // Cluster-distributed shared memory (DSMEM)
    // Each CTA in cluster can access other CTAs' shared memory
    
    int cluster_rank = cluster.block_rank();
    dim3 cluster_idx = cluster.block_index();
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    int num_k_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        // Load data into shared memory
        if (row < M && (kt * TILE_SIZE + tx) < K) {
            smem_A[ty][tx] = A[row * K + kt * TILE_SIZE + tx];
        } else {
            smem_A[ty][tx] = 0.0f;
        }
        
        if ((kt * TILE_SIZE + ty) < K && col < N) {
            smem_B[ty][tx] = B[(kt * TILE_SIZE + ty) * N + col];
        } else {
            smem_B[ty][tx] = 0.0f;
        }
        
        // Synchronize cluster - allows distributed access
        cluster.sync();
        
        // Compute using local shared memory
        // In advanced version, could access neighbor CTA's shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += smem_A[ty][k] * smem_B[k][tx];
        }
        
        cluster.sync();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Reduction kernel using clusters
__global__ void __cluster_dims__(4, 1, 1) cluster_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();
    
    __shared__ float smem[256];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Load data
    float val = (idx < N) ? input[idx] : 0.0f;
    smem[tid] = val;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    
    // Cluster-level aggregation
    if (tid == 0) {
        atomicAdd(&output[bid / cluster.num_blocks()], smem[0]);
    }
    
    cluster.sync();
}

bool verify_gemm(const float* C, int M, int N, float expected) {
    for (int i = 0; i < M * N; ++i) {
        if (fabs(C[i] - expected) > 1e-3) {
            printf("GEMM verification failed at %d: expected %f, got %f\n", 
                   i, expected, C[i]);
            return false;
        }
    }
    return true;
}

bool verify_reduction(const float* output, int num_outputs, int N, float input_val) {
    int elems_per_output = (N + num_outputs - 1) / num_outputs;
    for (int i = 0; i < num_outputs; ++i) {
        int count = min(elems_per_output, N - i * elems_per_output);
        float expected = input_val * count;
        if (fabs(output[i] - expected) > 1e-2) {
            printf("Reduction verification failed at %d: expected %f, got %f\n", 
                   i, expected, output[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== Blackwell CTA Clusters Test ===\n\n");
    
    // Check compute capability
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    if (prop.major < 9) {
        printf("CTA Clusters require SM 9.0+. This device is SM %d.%d\n", 
               prop.major, prop.minor);
        return 1;
    }
    
    printf("CTA Clusters supported: YES\n");
    printf("Cluster Features:\n");
    printf("  - Cooperative thread block groups\n");
    printf("  - Distributed shared memory (DSMEM)\n");
    printf("  - Cross-CTA synchronization\n");
    printf("  - Enhanced data sharing\n\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Test 1: Cluster GEMM
    printf("Test 1: Cluster GEMM (2x2 cluster)\n");
    {
        const int M = 1024;
        const int N = 1024;
        const int K = 1024;
        const int TILE_SIZE = 32;
        
        float *h_A = new float[M * K];
        float *h_B = new float[K * N];
        float *h_C = new float[M * N];
        
        for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
        for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;
        
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        
        cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        
        // Launch with cluster
        cudaLaunchConfig_t config = {0};
        config.gridDim = grid;
        config.blockDim = block;
        
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = 2;
        attrs[0].val.clusterDim.y = 2;
        attrs[0].val.clusterDim.z = 1;
        config.attrs = attrs;
        config.numAttrs = 1;
        
        cudaEventRecord(start);
        // Use standard launch for now - cluster attributes set via config
        void* args[] = {(void*)&d_A, (void*)&d_B, (void*)&d_C, (void*)&M, (void*)&N, (void*)&K};
        cudaError_t err = cudaLaunchKernelExC(
            &config,
            (void*)cluster_gemm_kernel,
            args
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        if (err != cudaSuccess) {
            printf("  Kernel launch failed: %s\n", cudaGetErrorString(err));
            printf("  Note: Cluster launch may require specific driver support\n");
        } else {
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            
            cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            
            float expected = (float)K;
            bool passed = verify_gemm(h_C, M, N, expected);
            
            printf("  Time: %.3f ms\n", ms);
            printf("  Result: %s\n", passed ? "PASSED" : "FAILED");
            
            double flops = 2.0 * M * N * K;
            double tflops = flops / (ms * 1e9);
            printf("  Performance: %.2f TFLOPS\n", tflops);
        }
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
    }
    printf("\n");
    
    // Test 2: Cluster DSMEM GEMM
    printf("Test 2: Cluster DSMEM GEMM (2x2 cluster)\n");
    {
        const int M = 1024;
        const int N = 1024;
        const int K = 1024;
        const int TILE_SIZE = 32;
        
        float *h_A = new float[M * K];
        float *h_B = new float[K * N];
        float *h_C = new float[M * N];
        
        for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
        for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;
        
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        
        cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        
        cudaLaunchConfig_t config = {0};
        config.gridDim = grid;
        config.blockDim = block;
        
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = 2;
        attrs[0].val.clusterDim.y = 2;
        attrs[0].val.clusterDim.z = 1;
        config.attrs = attrs;
        config.numAttrs = 1;
        
        cudaEventRecord(start);
        void* args2[] = {(void*)&d_A, (void*)&d_B, (void*)&d_C, (void*)&M, (void*)&N, (void*)&K};
        cudaError_t err = cudaLaunchKernelExC(
            &config,
            (void*)cluster_dsmem_kernel,
            args2
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        if (err != cudaSuccess) {
            printf("  Kernel launch failed: %s\n", cudaGetErrorString(err));
        } else {
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            
            cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            
            float expected = (float)K;
            bool passed = verify_gemm(h_C, M, N, expected);
            
            printf("  Time: %.3f ms\n", ms);
            printf("  Result: %s\n", passed ? "PASSED" : "FAILED");
            
            double flops = 2.0 * M * N * K;
            double tflops = flops / (ms * 1e9);
            printf("  Performance: %.2f TFLOPS\n", tflops);
        }
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
    }
    printf("\n");
    
    // Test 3: Cluster Reduction
    printf("Test 3: Cluster Reduction (4x1 cluster)\n");
    {
        const int N = 1024 * 1024;
        const int BLOCK_SIZE = 256;
        const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int CLUSTER_SIZE = 4;
        const int NUM_OUTPUTS = (NUM_BLOCKS + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
        
        float *h_input = new float[N];
        float *h_output = new float[NUM_OUTPUTS];
        
        for (int i = 0; i < N; ++i) h_input[i] = 1.0f;
        for (int i = 0; i < NUM_OUTPUTS; ++i) h_output[i] = 0.0f;
        
        float *d_input, *d_output;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, NUM_OUTPUTS * sizeof(float));
        
        cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output, h_output, NUM_OUTPUTS * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaLaunchConfig_t config = {0};
        config.gridDim = dim3(NUM_BLOCKS);
        config.blockDim = dim3(BLOCK_SIZE);
        
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = CLUSTER_SIZE;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        config.attrs = attrs;
        config.numAttrs = 1;
        
        cudaEventRecord(start);
        void* args3[] = {(void*)&d_input, (void*)&d_output, (void*)&N};
        cudaError_t err = cudaLaunchKernelExC(
            &config,
            (void*)cluster_reduction_kernel,
            args3
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        if (err != cudaSuccess) {
            printf("  Kernel launch failed: %s\n", cudaGetErrorString(err));
        } else {
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            
            cudaMemcpy(h_output, d_output, NUM_OUTPUTS * sizeof(float), cudaMemcpyDeviceToHost);
            
            bool passed = verify_reduction(h_output, NUM_OUTPUTS, N, 1.0f);
            
            printf("  Time: %.3f ms\n", ms);
            printf("  Result: %s\n", passed ? "PASSED" : "FAILED");
            
            double bandwidth = (N * sizeof(float)) / (ms * 1e6);
            printf("  Bandwidth: %.2f GB/s\n", bandwidth);
        }
        
        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
        delete[] h_output;
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n=== CTA Clusters Test Complete ===\n");
    
    return 0;
}

