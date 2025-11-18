#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {

constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int CHUNK_M = 32;

__device__ void decode_tile_math(float* tile, int pitch, int rows, int cols) {
    for (int r = threadIdx.y; r < rows; r += blockDim.y) {
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            float v = tile[r * pitch + c];
            tile[r * pitch + c] = v * 1.0002f + 0.0001f;
        }
    }
}

template <int TILE_N_VALUE, int CHUNK_M_VALUE>
__global__ void baseline_decode_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols,
    int ld_input,
    int ld_output) {
    __shared__ alignas(16) float stage_buffer[CHUNK_M_VALUE][TILE_N_VALUE];

    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int row0 = tile_m * TILE_M;
    const int col0 = tile_n * TILE_N_VALUE;

    if (row0 >= rows || col0 >= cols) {
        return;
    }

    const int tile_rows = min(TILE_M, rows - row0);
    const int tile_cols = min(TILE_N_VALUE, cols - col0);
    const int num_chunks = (tile_rows + CHUNK_M_VALUE - 1) / CHUNK_M_VALUE;

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int rows_this_chunk = min(CHUNK_M_VALUE, tile_rows - chunk * CHUNK_M_VALUE);
        const int row_base = row0 + chunk * CHUNK_M_VALUE;
        float* tile_ptr = &stage_buffer[0][0];

        for (int r = threadIdx.y; r < rows_this_chunk; r += blockDim.y) {
            const int gr = row_base + r;
            for (int c = threadIdx.x; c < tile_cols; c += blockDim.x) {
                const int gc = col0 + c;
                tile_ptr[r * TILE_N_VALUE + c] = input[gr * ld_input + gc];
            }
        }
        __syncthreads();

        decode_tile_math(tile_ptr, TILE_N_VALUE, rows_this_chunk, tile_cols);
        __syncthreads();

        for (int r = threadIdx.y; r < rows_this_chunk; r += blockDim.y) {
            const int gr = row_base + r;
            for (int c = threadIdx.x; c < tile_cols; c += blockDim.x) {
                const int gc = col0 + c;
                output[gr * ld_output + gc] = tile_ptr[r * TILE_N_VALUE + c];
            }
        }
        __syncthreads();
    }
}

void launch_baseline_kernel(
    torch::Tensor input,
    torch::Tensor output) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "Output must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "Output must be float32");
    TORCH_CHECK(input.sizes() == output.sizes(), "Input/output shapes must match");
    TORCH_CHECK(input.dim() == 2, "Decode kernel expects 2D tensors");

    auto in_contig = input.contiguous();
    auto out_contig = output.contiguous();

    const int rows = static_cast<int>(in_contig.size(0));
    const int cols = static_cast<int>(in_contig.size(1));
    if (rows == 0 || cols == 0) {
        return;
    }

    const int ld_input = static_cast<int>(in_contig.stride(0));
    const int ld_output = static_cast<int>(out_contig.stride(0));
    const float* input_ptr = in_contig.data_ptr<float>();
    float* output_ptr = out_contig.data_ptr<float>();

    dim3 block(32, 4, 1);
    dim3 grid(
        (cols + TILE_N - 1) / TILE_N,
        (rows + TILE_M - 1) / TILE_M,
        1);

    auto stream = at::cuda::getCurrentCUDAStream();
    baseline_decode_kernel<TILE_N, CHUNK_M><<<grid, block, 0, stream>>>(
        input_ptr,
        output_ptr,
        rows,
        cols,
        ld_input,
        ld_output);
    AT_CUDA_CHECK(cudaGetLastError());
    if (!output.is_contiguous()) {
        output.copy_(out_contig);
    }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_baseline", &launch_baseline_kernel, "Baseline decode kernel (naive global loads)");
}
