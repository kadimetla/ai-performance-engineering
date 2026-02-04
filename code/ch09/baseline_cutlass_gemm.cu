// baseline_cutlass_gemm.cu -- CUTLASS SIMT GEMM baseline (no tensor cores).
// Inputs are preloaded on device; H2D transfers are excluded from timing.

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUTLASS_CHECK(status)                                                    \
  do {                                                                           \
    cutlass::Status error = (status);                                            \
    if (error != cutlass::Status::kSuccess) {                                    \
      std::cerr << "CUTLASS error " << __FILE__ << ":" << __LINE__ << " "         \
                << cutlassGetStatusString(error) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

int main() {
    NVTX_RANGE("main");
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr int kIterations = 5;
    constexpr int kRepeats = 32;

    using Element = float;
    using Layout = cutlass::layout::RowMajor;
    using ElementAccumulator = float;

    using Gemm = cutlass::gemm::device::Gemm<
        Element, Layout,
        Element, Layout,
        Element, Layout,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm100
    >;

    const size_t elements_A = static_cast<size_t>(M) * K;
    const size_t elements_B = static_cast<size_t>(K) * N;
    const size_t elements_C = static_cast<size_t>(M) * N;
    const size_t size_A = elements_A * sizeof(Element);
    const size_t size_B = elements_B * sizeof(Element);
    const size_t size_C = elements_C * sizeof(Element);

    Element* h_A = nullptr;
    Element* h_B = nullptr;
    Element* h_C = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_A, size_A));
    CUDA_CHECK(cudaMallocHost(&h_B, size_B));
    CUDA_CHECK(cudaMallocHost(&h_C, size_C));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < elements_A; ++i) {
        NVTX_RANGE("setup");
        h_A[i] = static_cast<Element>(dis(gen));
    }
    for (size_t i = 0; i < elements_B; ++i) {
        NVTX_RANGE("setup");
        h_B[i] = static_cast<Element>(dis(gen));
    }
    std::fill(h_C, h_C + elements_C, Element(0));

    Element *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    cutlass::gemm::GemmCoord problem_size(M, N, K);
    cutlass::TensorRef<Element const, Layout> ref_A(d_A, Layout(lda));
    cutlass::TensorRef<Element const, Layout> ref_B(d_B, Layout(ldb));
    cutlass::TensorRef<Element const, Layout> ref_C(d_C, Layout(ldc));
    cutlass::TensorRef<Element, Layout> ref_D(d_C, Layout(ldc));

    typename Gemm::Arguments args(
        problem_size,
        ref_A,
        ref_B,
        ref_C,
        ref_D,
        {ElementAccumulator(1.0f), ElementAccumulator(0.0f)}
    );

    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }
    CUTLASS_CHECK(gemm_op.initialize(args, workspace, stream));

    // Warmup
    CUTLASS_CHECK(gemm_op.run(stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iter = 0; iter < kIterations; ++iter) {
        NVTX_RANGE("compute_math:cutlass");
        for (int rep = 0; rep < kRepeats; ++rep) {
            CUTLASS_CHECK(gemm_op.run(stream));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(kIterations * kRepeats);
    std::cout << "CUTLASS SIMT GEMM (baseline): " << avg_ms << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    std::cout << "Checksum sample: " << h_C[0] << std::endl;

#ifdef VERIFY
    double checksum = 0.0;
    for (size_t i = 0; i < elements_C; ++i) {
        NVTX_RANGE("verify");
        checksum += std::abs(static_cast<double>(h_C[i]));
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (workspace) {
        CUDA_CHECK(cudaFree(workspace));
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return 0;
}
