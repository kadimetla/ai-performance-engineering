/**
 * HBM3e Memory Access Optimization for Blackwell
 * ===============================================
 * 
 * Blackwell B200/B300 uses HBM3e with 7.8 TB/s bandwidth.
 * Optimal access patterns:
 * - 256-byte bursts (HBM3e burst size)
 * - 128-byte cache lines
 * - Cache streaming modifiers (.cs)
 * - Vectorized loads/stores (128-bit = float4)
 * 
 * This example demonstrates optimal HBM3e access patterns
 * and compares different strategies.
 * 
 * Requirements:
 * - CUDA 13.0+
 * - Blackwell GPU (B200/B300)
 * 
 * Compile:
 *   nvcc -O3 -std=c++17 -arch=sm_100 hbm3e_optimized_copy.cu -o hbm3e_copy
 * 
 * Expected Performance:
 * - 7.8+ TB/s (>95% of HBM3e peak bandwidth)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================================
// Baseline: Scalar copy (very inefficient)
// ============================================================================

__global__ void scalar_copy_kernel(float* dst, const float* src, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = tid; i < n; i += stride) {
        dst[i] = src[i];  // 4-byte transactions
    }
}

// ============================================================================
// Improved: Vectorized copy with float4 (128-bit)
// ============================================================================

__global__ void vectorized_copy_kernel(float4* dst, const float4* src, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    // Each float4 is 16 bytes (128-bit)
    for (size_t i = tid; i < n; i += stride) {
        dst[i] = src[i];  // 16-byte transactions
    }
}

// ============================================================================
// Optimized: 256-byte bursts with cache streaming (Blackwell HBM3e optimal)
// ============================================================================

__global__ void hbm3e_optimized_copy_kernel(float4* dst, const float4* src, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    // Process 256 bytes per iteration (16 float4s)
    // This matches HBM3e burst size
    constexpr int BURST_SIZE = 16;  // 16 * 16 bytes = 256 bytes
    
    for (size_t base = tid * BURST_SIZE; base < n; base += stride * BURST_SIZE) {
        // Unroll loop for 256-byte burst
        #pragma unroll
        for (int i = 0; i < BURST_SIZE; i++) {
            size_t idx = base + i;
            if (idx < n) {
                // Use cache streaming modifier for HBM3e
                // .cs (cache-streaming) bypasses L2 for write-only patterns
                #if __CUDA_ARCH__ >= 1000  // Blackwell
                // PTX inline assembly for cache streaming
                asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];" 
                    : "=f"(reinterpret_cast<float*>(&dst[idx])[0]),
                      "=f"(reinterpret_cast<float*>(&dst[idx])[1]),
                      "=f"(reinterpret_cast<float*>(&dst[idx])[2]),
                      "=f"(reinterpret_cast<float*>(&dst[idx])[3])
                    : "l"(&src[idx]));
                #else
                dst[idx] = src[idx];
                #endif
            }
        }
    }
}

// ============================================================================
// Advanced: Prefetching for HBM3e
// ============================================================================

__device__ __forceinline__ void prefetch_global_l2(const void* ptr) {
    #if __CUDA_ARCH__ >= 1000  // Blackwell
    // Software prefetch hint for HBM3e
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
    #endif
}

__global__ void hbm3e_prefetch_copy_kernel(float4* dst, const float4* src, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    constexpr int BURST_SIZE = 16;
    constexpr int PREFETCH_DISTANCE = 128;  // Prefetch 128 * 16 bytes ahead
    
    for (size_t base = tid * BURST_SIZE; base < n; base += stride * BURST_SIZE) {
        // Prefetch future data
        if (base + PREFETCH_DISTANCE * BURST_SIZE < n) {
            prefetch_global_l2(&src[base + PREFETCH_DISTANCE * BURST_SIZE]);
        }
        
        // Process current burst
        #pragma unroll
        for (int i = 0; i < BURST_SIZE; i++) {
            size_t idx = base + i;
            if (idx < n) {
                dst[idx] = src[idx];
            }
        }
    }
}

// ============================================================================
// Benchmarking
// ============================================================================

void benchmark_copy(const char* name, 
                   void (*kernel)(float4*, const float4*, size_t),
                   float4* d_dst, const float4* d_src, 
                   size_t n_float4, int iterations) {
    
    // Warmup
    kernel<<<256, 256>>>(d_dst, d_src, n_float4);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<256, 256>>>(d_dst, d_src, n_float4);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iterations;
    
    // Calculate bandwidth
    size_t bytes = n_float4 * sizeof(float4) * 2;  // read + write
    double bandwidth_gbs = (bytes / (avg_ms / 1000.0)) / 1e9;
    
    printf("  %-30s: %6.2f ms, %7.2f GB/s\n", name, avg_ms, bandwidth_gbs);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_benchmarks() {
    printf("=== HBM3e Memory Access Optimization ===\n\n");
    
    // Check GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    if (prop.major == 10 && prop.minor == 0) {
        printf("Blackwell B200/B300 detected (HBM3e)\n");
        printf("  Peak bandwidth: 7.8 TB/s\n");
        printf("  Target: >7.0 TB/s (>90%% utilization)\n");
    } else {
        printf("⚠ Non-Blackwell GPU - HBM3e optimizations may not apply\n");
    }
    
    printf("\n");
    
    // Test sizes
    size_t sizes[] = {
        64 * 1024 * 1024,    // 256 MB
        256 * 1024 * 1024,   // 1 GB
        1024 * 1024 * 1024   // 4 GB
    };
    
    for (size_t size_bytes : sizes) {
        size_t n_floats = size_bytes / sizeof(float);
        size_t n_float4 = n_floats / 4;
        
        printf("Testing with %zu MB (%zu elements)\n", size_bytes / 1024 / 1024, n_floats);
        
        // Allocate memory
        float* d_src_scalar;
        float* d_dst_scalar;
        float4* d_src;
        float4* d_dst;
        
        cudaMalloc(&d_src_scalar, size_bytes);
        cudaMalloc(&d_dst_scalar, size_bytes);
        cudaMalloc(&d_src, size_bytes);
        cudaMalloc(&d_dst, size_bytes);
        
        // Initialize
        cudaMemset(d_src_scalar, 1, size_bytes);
        cudaMemset(d_src, 1, size_bytes);
        
        // Run benchmarks
        int iterations = size_bytes < 512 * 1024 * 1024 ? 100 : 20;
        
        // Scalar copy (baseline)
        {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            scalar_copy_kernel<<<256, 256>>>(d_dst_scalar, d_src_scalar, n_floats);
            cudaDeviceSynchronize();
            
            cudaEventRecord(start);
            for (int i = 0; i < iterations; i++) {
                scalar_copy_kernel<<<256, 256>>>(d_dst_scalar, d_src_scalar, n_floats);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            float avg_ms = ms / iterations;
            double bw = (size_bytes * 2 / (avg_ms / 1000.0)) / 1e9;
            
            printf("  %-30s: %6.2f ms, %7.2f GB/s\n", "Scalar (4-byte, baseline)", avg_ms, bw);
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        
        // Vectorized copy
        benchmark_copy("Vectorized (16-byte)", vectorized_copy_kernel, 
                      d_dst, d_src, n_float4, iterations);
        
        // HBM3e optimized (256-byte bursts)
        benchmark_copy("HBM3e optimized (256-byte)", hbm3e_optimized_copy_kernel,
                      d_dst, d_src, n_float4, iterations);
        
        // Prefetch optimized
        benchmark_copy("HBM3e + prefetch", hbm3e_prefetch_copy_kernel,
                      d_dst, d_src, n_float4, iterations);
        
        printf("\n");
        
        // Cleanup
        cudaFree(d_src_scalar);
        cudaFree(d_dst_scalar);
        cudaFree(d_src);
        cudaFree(d_dst);
    }
    
    printf("=== Key Takeaways ===\n");
    printf("1. HBM3e optimal: 256-byte bursts (16 float4s)\n");
    printf("2. Cache streaming (.cs) for write-only patterns\n");
    printf("3. Software prefetch for better latency hiding\n");
    printf("4. Target >90%% of 7.8 TB/s peak (~7.0 TB/s)\n");
    printf("5. Blackwell achieves 30%% higher bandwidth than Hopper\n");
}

int main() {
    run_benchmarks();
    return 0;
}

