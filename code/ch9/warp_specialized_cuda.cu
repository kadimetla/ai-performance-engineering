// Warp-specialized kernel for Chapter 9: Kernel Efficiency / Arithmetic Intensity
// Based on Chapter 10's warp_specialized_pipeline_enhanced.cu
// Demonstrates producer/consumer warp specialization pattern
// CUDA 13 Update: Float8 available for Blackwell 256-bit loads

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
            exit(1); \
        } \
    } while(0)

// CUDA 13 + Blackwell: 32-byte aligned type for 256-bit loads
struct alignas(32) Float8 {
    float elems[8];
};
static_assert(sizeof(Float8) == 32, "Float8 must be 32 bytes");
static_assert(alignof(Float8) == 32, "Float8 must be 32-byte aligned");

constexpr int WARP_SIZE = 32;
constexpr int TILE_SIZE = 256;
constexpr int PIPELINE_DEPTH = 2;

// Warp specialization: Producer/Consumer pattern
// Warp 0: Producer (loads data)
// Warps 1-6: Compute (process data)
// Warp 7: Consumer (stores results)
__global__ void warp_specialized_ch9_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_elements
) {
    cg::thread_block block = cg::this_thread_block();
    
    // Shared memory for double buffering
    extern __shared__ float smem[];
    float* stage_buffer = smem;
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    // Warp roles (8 warps per block: 1 producer, 6 compute, 1 consumer)
    constexpr int PRODUCER_WARP = 0;
    constexpr int CONSUMER_WARP = 7;
    constexpr int COMPUTE_WARPS_START = 1;
    constexpr int COMPUTE_WARPS_END = 6;
    
    bool is_producer = (warp_id == PRODUCER_WARP);
    bool is_consumer = (warp_id == CONSUMER_WARP);
    bool is_compute = (warp_id >= COMPUTE_WARPS_START && warp_id <= COMPUTE_WARPS_END);
    
    int num_tiles = (n_elements + TILE_SIZE - 1) / TILE_SIZE;
    
    // Initial sync: all warps start together
    block.sync();
    
    // Unified loop so all warps hit the same synchronization points.
    // This prevents deadlocks from warp-specific __syncthreads barriers.
    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        int buf_idx = tile % PIPELINE_DEPTH;
        int tile_start = tile * TILE_SIZE;
        
        // Producer stage: load tile into shared memory
        // On Blackwell, could use Float8 for 256-bit loads, but float4 is sufficient
        // for this pipeline pattern (shared memory staging)
        if (is_producer) {
            for (int idx = lane; idx < TILE_SIZE / 4; idx += WARP_SIZE) {
                int global_idx = tile_start + idx * 4;
                if (global_idx + 3 < n_elements) {
                    float4 val4 = *reinterpret_cast<const float4*>(&input[global_idx]);
                    *reinterpret_cast<float4*>(&stage_buffer[buf_idx * TILE_SIZE + idx * 4]) = val4;
                } else {
                    for (int i = 0; i < 4 && global_idx + i < n_elements; ++i) {
                        stage_buffer[buf_idx * TILE_SIZE + idx * 4 + i] = input[global_idx + i];
                    }
                }
            }
        }
        
        block.sync();
        
        // Compute stage: wait for data, process slice
        if (is_compute) {
            int compute_warp_id = warp_id - COMPUTE_WARPS_START;
            int num_compute_warps = COMPUTE_WARPS_END - COMPUTE_WARPS_START + 1;
            
            int elems_per_warp = TILE_SIZE / num_compute_warps;
            int start_idx = compute_warp_id * elems_per_warp;
            
            for (int idx = start_idx + lane; idx < start_idx + elems_per_warp; idx += WARP_SIZE) {
                if (idx < TILE_SIZE) {
                    float val = stage_buffer[buf_idx * TILE_SIZE + idx];
                    val = fmaxf(val, 0.0f);  // ReLU
                    stage_buffer[buf_idx * TILE_SIZE + idx] = val;
                }
            }
        }
        
        block.sync();
        
        // Consumer stage: write tile back to global memory
        if (is_consumer) {
            for (int idx = lane; idx < TILE_SIZE / 4; idx += WARP_SIZE) {
                int global_idx = tile_start + idx * 4;
                if (global_idx + 3 < n_elements) {
                    float val0 = stage_buffer[buf_idx * TILE_SIZE + idx * 4];
                    float val1 = stage_buffer[buf_idx * TILE_SIZE + idx * 4 + 1];
                    float val2 = stage_buffer[buf_idx * TILE_SIZE + idx * 4 + 2];
                    float val3 = stage_buffer[buf_idx * TILE_SIZE + idx * 4 + 3];
                    
                    float4 result4 = make_float4(val0 * 0.5f, val1 * 0.5f, val2 * 0.5f, val3 * 0.5f);
                    *reinterpret_cast<float4*>(&output[global_idx]) = result4;
                } else {
                    for (int i = 0; i < 4 && global_idx + i < n_elements; ++i) {
                        float val = stage_buffer[buf_idx * TILE_SIZE + idx * 4 + i];
                        output[global_idx + i] = val * 0.5f;
                    }
                }
            }
        }
        
        block.sync();
    }
}

torch::Tensor warp_specialized_ch9_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    int n_elements = input.numel();
    auto output = torch::empty_like(input);
    
    // Launch configuration: 8 warps per block (1 producer + 6 compute + 1 consumer)
    dim3 block(8 * WARP_SIZE);  // 256 threads = 8 warps
    dim3 grid((n_elements + TILE_SIZE - 1) / TILE_SIZE);
    int shared_mem_size = PIPELINE_DEPTH * TILE_SIZE * sizeof(float);
    
    warp_specialized_ch9_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warp_specialized_ch9_forward", &warp_specialized_ch9_forward, "Warp specialized forward pass for ch9");
}
