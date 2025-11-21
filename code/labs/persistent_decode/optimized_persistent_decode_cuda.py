"""Persistent decode in CUDA via a tiny inline extension."""

from __future__ import annotations

import functools
from typing import Optional

import torch
from torch.utils.cpp_extension import load_inline

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


@functools.lru_cache(None)
def _load_extension() -> object:
    """Compile and return the CUDA extension once per process."""
    cpp_src = r"""
#include <torch/extension.h>
void persistent_decode_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out, int blocks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("persistent_decode", &persistent_decode_cuda, "Persistent decode (CUDA)");
}
"""

    cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

namespace {

__device__ int g_work_item_idx = 0;

// Helper for cp.async tile copies. TILE_BYTES is a compile-time cap; we clamp
// runtime valid_bytes to avoid over-fetching for smaller head_dims.
template<int TILE_BYTES>
__device__ inline void cp_async_tile(void* smem_dst, const void* gmem_src, int valid_bytes) {
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    unsigned gmem_addr = static_cast<unsigned>(__cvta_generic_to_global(gmem_src));

    int nbytes = valid_bytes > 0 ? valid_bytes : TILE_BYTES;
    for (int offset = threadIdx.x * 16; offset < nbytes; offset += blockDim.x * 16) {
        __pipeline_memcpy_async(
            reinterpret_cast<void*>(smem_addr + offset),
            reinterpret_cast<const void*>(gmem_addr + offset),
            16);
    }
}

// Simple tiled dot using shared memory for reduction (used for fallback).
__device__ float dot_tile_fallback(const float* q, const float* k, int head_dim) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        acc += q[d] * k[d];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

__global__ void persistent_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int batch,
    int seq_len,
    int head_dim
) {
    // GPT-OSS-20B uses 128-dim heads; keep buffers sized accordingly.
    constexpr int MAX_HEAD_DIM = 128;
    constexpr int TILE_BYTES = MAX_HEAD_DIM * sizeof(float);

    extern __shared__ float smem_f[];
    float* smem_k0 = smem_f;
    float* smem_k1 = smem_k0 + MAX_HEAD_DIM;
    float* smem_v0 = smem_k1 + MAX_HEAD_DIM;
    float* smem_v1 = smem_v0 + MAX_HEAD_DIM;
    float* red = smem_v1 + MAX_HEAD_DIM;  // reduction buffer (threads slots)

    const bool use_cp_async = head_dim <= MAX_HEAD_DIM;

    while (true) {
        int idx;
        if (threadIdx.x == 0) {
            idx = atomicAdd(&g_work_item_idx, 1);
        }
        idx = __shfl_sync(0xffffffff, idx, 0);
        if (idx >= batch) {
            return;
        }

        int seq_id = idx;
        for (int t = 0; t < seq_len; ++t) {
            const float* q_ptr = q + (seq_id * seq_len + t) * head_dim;
            const float* k_ptr = k + (seq_id * seq_len + t) * head_dim;
            const float* v_ptr = v + (seq_id * seq_len + t) * head_dim;
            float* out_ptr = out + (seq_id * seq_len + t) * head_dim;

            if (!use_cp_async) {
                float dot = dot_tile_fallback(q_ptr, k_ptr, head_dim);
                for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                    out_ptr[d] = v_ptr[d] * dot;
                }
                __syncthreads();
                continue;
            }

            // Double-buffered cp.async pipeline for K/V.
            int buf = t & 1;
            int nextbuf = buf ^ 1;
            float* k_smem_curr = buf == 0 ? smem_k0 : smem_k1;
            float* v_smem_curr = buf == 0 ? smem_v0 : smem_v1;

            // Warmup: load first tile before compute.
            if (t == 0) {
                cp_async_tile<TILE_BYTES>(k_smem_curr, k_ptr, head_dim * static_cast<int>(sizeof(float)));
                cp_async_tile<TILE_BYTES>(v_smem_curr, v_ptr, head_dim * static_cast<int>(sizeof(float)));
                __pipeline_commit();
            }

            // Wait for the current tile to land before use.
            __pipeline_wait_prior(0);
            __syncthreads();

            // Prefetch next tile while we work on this one.
            int next_t = t + 1;
            if (next_t < seq_len) {
                const float* k_ptr_next = k + (seq_id * seq_len + next_t) * head_dim;
                const float* v_ptr_next = v + (seq_id * seq_len + next_t) * head_dim;
                float* k_smem_next = nextbuf == 0 ? smem_k0 : smem_k1;
                float* v_smem_next = nextbuf == 0 ? smem_v0 : smem_v1;
                cp_async_tile<TILE_BYTES>(k_smem_next, k_ptr_next, head_dim * static_cast<int>(sizeof(float)));
                cp_async_tile<TILE_BYTES>(v_smem_next, v_ptr_next, head_dim * static_cast<int>(sizeof(float)));
                __pipeline_commit();
            }

            // Compute dot(q, k_tile) using shared buffers.
            float acc = 0.0f;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                acc += q_ptr[d] * k_smem_curr[d];
            }
            red[threadIdx.x] = acc;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    red[threadIdx.x] += red[threadIdx.x + stride];
                }
                __syncthreads();
            }

            float dot = red[0];
            __syncthreads();
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                out_ptr[d] = v_smem_curr[d] * dot;
            }
            __syncthreads();
        }
    }
}

} // namespace

void persistent_decode_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out, int blocks) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat, "q must be float32");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shapes must match");
    TORCH_CHECK(out.sizes() == q.sizes(), "out shape mismatch");
    TORCH_CHECK(q.size(2) <= 128, "head_dim exceeds MAX_HEAD_DIM=128 for cp.async path");

    const int batch = static_cast<int>(q.size(0));
    const int seq_len = static_cast<int>(q.size(1));
    const int head_dim = static_cast<int>(q.size(2));
    const int threads = 64;

    constexpr int MAX_HEAD_DIM = 128;
    const size_t smem_bytes = (
        // K0/K1/V0/V1
        4 * MAX_HEAD_DIM
        // Reduction buffer
        + threads
    ) * sizeof(float);

    c10::cuda::CUDAGuard device_guard(q.get_device());
    cudaDeviceProp prop{};
    AT_CUDA_CHECK(cudaGetDeviceProperties(&prop, q.get_device()));
    TORCH_CHECK(prop.major >= 8, "cp.async pipeline requires SM80+ (Ampere or newer)");
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    int* counter_ptr = nullptr;
    AT_CUDA_CHECK(cudaGetSymbolAddress((void**)&counter_ptr, g_work_item_idx));
    AT_CUDA_CHECK(cudaMemsetAsync(counter_ptr, 0, sizeof(int), stream));

    persistent_decode_kernel<<<blocks, threads, smem_bytes, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        seq_len,
        head_dim);
    AT_CUDA_CHECK(cudaGetLastError());
}
"""

    return load_inline(
        name="persistent_decode_ext",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=None,
        extra_cuda_cflags=["--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


class OptimizedPersistentDecodeCUDABenchmark(BaseBenchmark):
    """Persistent decode using a cooperative CUDA kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.blocks = 8
        self._ext: Optional[object] = None
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        self.inputs = build_inputs(self.device)
        self._ext = _load_extension()
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self._ext is None:
            raise RuntimeError("Extension or inputs not initialized")

        with self._nvtx_range("persistent_decode_cuda"):
            self._ext.persistent_decode(
                self.inputs.q,
                self.inputs.k,
                self.inputs.v,
                self.inputs.out,
                self.blocks,
            )
            self._synchronize()

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=12, warmup=4)

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedPersistentDecodeCUDABenchmark()
