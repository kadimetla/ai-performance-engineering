#!/usr/bin/env python3
"""
Accurate Blackwell CUDA feature verification.

This script covers the hardware features that were previously only exercised by
legacy C++ samples under tools/verification_blackwell:
  1) Distributed shared memory across thread-block clusters
  2) Warp-level async copies (cp.async via cooperative_groups::memcpy_async)
  3) Warp-specialized producer/consumer pipelines (cuda::pipeline)

It builds a tiny CUDA extension on the fly to exercise the real primitives and
fails fast if any check misbehaves. The script is intended for Blackwell-class
GPUs (SM >= 100) but will skip gracefully on older architectures.
"""

from __future__ import annotations

import os
import pathlib
import sys

import torch
from torch.utils.cpp_extension import load_inline


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD_ROOT = PROJECT_ROOT / "artifacts" / "torch_extensions" / "blackwell_cuda_checks"


def _format_status(name: str, ok: bool, detail: str | None = None) -> None:
    prefix = "✓" if ok else "✗"
    line = f"{prefix} {name}"
    if detail:
        line = f"{line}: {detail}"
    print(line)


def _load_extension() -> object:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    major, minor = torch.cuda.get_device_capability()
    arch_tag = f"{major}{minor}"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    name = f"blackwell_cuda_checks_sm{arch_tag}"

    cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <algorithm>

namespace cg = cooperative_groups;

__global__ __cluster_dims__(2, 1, 1) void dsmem_probe_kernel(int* out) {
    __shared__ int smem[32];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank());
    const int lane = static_cast<int>(threadIdx.x);
    if (lane < 32) {
        smem[lane] = rank;
    }
    cluster.sync();
    const int peer = (rank + 1) % cluster.dim_blocks().x;
    int* peer_smem = cluster.map_shared_rank(smem, peer);
    const int peer_val = peer_smem[0];
    if (lane == 0) {
        out[rank] = peer_val;
    }
}

__global__ void warp_async_copy_kernel(const float* src, float* dst, int n) {
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ float smem[];
    const int base = static_cast<int>(blockIdx.x * blockDim.x);
    const int count = min(static_cast<int>(blockDim.x), n - base);
    if (count <= 0) return;

    cg::memcpy_async(block, smem, src + base, count * sizeof(float));
    cg::wait(block);
    block.sync();

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        dst[base + i] = smem[i] + 1.0f;
    }
}

__global__ void warp_specialization_pipeline_kernel(const float* src, float* dst, int n) {
    cg::thread_block block = cg::this_thread_block();
    constexpr int kWarpSize = 32;
    constexpr int kStages = 2;
    constexpr int kTileElems = 256;

    using pipeline_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, kStages>;
    __shared__ alignas(pipeline_state_t) unsigned char pipeline_storage[sizeof(pipeline_state_t)];
    __shared__ float stages[kStages][kTileElems];

    const int warp_id = threadIdx.x / kWarpSize;
    const bool is_producer = warp_id == 0;
    if (threadIdx.x == 0) {
        new (reinterpret_cast<pipeline_state_t*>(pipeline_storage)) pipeline_state_t();
    }
    block.sync();
    auto pipe = cuda::make_pipeline(block, reinterpret_cast<pipeline_state_t*>(pipeline_storage));

    const int tiles = (n + kTileElems - 1) / kTileElems;
    for (int tile = 0; tile < tiles; ++tile) {
        const int stage_idx = tile % kStages;
        const int base = tile * kTileElems;
        const int limit = min(kTileElems, n - base);

        pipe.producer_acquire();
        if (is_producer) {
            for (int i = threadIdx.x % kWarpSize; i < limit; i += kWarpSize) {
                stages[stage_idx][i] = src[base + i];
            }
        }
        pipe.producer_commit();

        pipe.consumer_wait();
        block.sync();

        if (!is_producer) {
            const int consumer_idx = threadIdx.x - kWarpSize;
            const int consumer_stride = blockDim.x - kWarpSize;
            for (int i = consumer_idx; i < limit; i += consumer_stride) {
                float v = stages[stage_idx][i];
                dst[base + i] = v * 2.0f + 1.0f;
            }
        }

        block.sync();
        pipe.consumer_release();
        block.sync();
    }
}

void run_dsmem_probe(torch::Tensor out) {
    TORCH_CHECK(out.is_cuda(), "out tensor must be CUDA");
    TORCH_CHECK(out.dtype() == torch::kInt32, "out tensor must be int32");
    TORCH_CHECK(out.is_contiguous(), "out tensor must be contiguous");
    TORCH_CHECK(out.numel() >= 2, "out tensor must hold at least two entries");

    c10::cuda::CUDAGuard guard(out.get_device());
    auto stream = at::cuda::getDefaultCUDAStream();

    const dim3 grid(2, 1, 1);
    const dim3 block(32, 1, 1);
    dsmem_probe_kernel<<<grid, block, 0, stream>>>(out.data_ptr<int>());
    C10_CUDA_CHECK(cudaGetLastError());
}

void run_warp_async_copy(torch::Tensor src, torch::Tensor dst) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA tensors");
    TORCH_CHECK(src.dtype() == torch::kFloat32 && dst.dtype() == torch::kFloat32,
                "src/dst must be float32");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src/dst must be contiguous");
    TORCH_CHECK(src.numel() == dst.numel(), "src/dst must have the same number of elements");

    c10::cuda::CUDAGuard guard(src.get_device());
    auto stream = at::cuda::getDefaultCUDAStream();

    const int n = static_cast<int>(src.numel());
    const dim3 block(256, 1, 1);
    const dim3 grid((n + block.x - 1) / block.x, 1, 1);
    const std::size_t shmem_bytes = static_cast<std::size_t>(block.x) * sizeof(float);
    warp_async_copy_kernel<<<grid, block, shmem_bytes, stream>>>(
        src.data_ptr<float>(), dst.data_ptr<float>(), n);
    C10_CUDA_CHECK(cudaGetLastError());
}

void run_warp_specialization(torch::Tensor src, torch::Tensor dst) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA tensors");
    TORCH_CHECK(src.dtype() == torch::kFloat32 && dst.dtype() == torch::kFloat32,
                "src/dst must be float32");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src/dst must be contiguous");
    TORCH_CHECK(src.numel() == dst.numel(), "src/dst size mismatch");
    const int n = static_cast<int>(src.numel());
    if (n == 0) {
        return;
    }

    c10::cuda::CUDAGuard guard(src.get_device());
    auto stream = at::cuda::getDefaultCUDAStream();
    constexpr int kBlockThreads = 96;  // 3 warps: 1 producer + 2 consumers
    const dim3 block(kBlockThreads, 1, 1);
    const dim3 grid(1, 1, 1);
    warp_specialization_pipeline_kernel<<<grid, block, 0, stream>>>(
        src.data_ptr<float>(), dst.data_ptr<float>(), n);
    C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_dsmem_probe", &run_dsmem_probe, "Distributed shared memory probe");
    m.def("run_warp_async_copy", &run_warp_async_copy, "Warp-level async copy check");
    m.def("run_warp_specialization", &run_warp_specialization, "Warp-specialized pipeline check");
}
"""

    return load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=None,
        extra_cuda_cflags=["-lineinfo", "--use_fast_math", "-std=c++20"],
        extra_cflags=["-O3", "-std=c++20"],
        build_directory=str(BUILD_ROOT),
        verbose=False,
    )


def _dsmem_test(ext: object) -> bool:
    out = torch.full((2,), -1, device="cuda", dtype=torch.int32)
    ext.run_dsmem_probe(out)
    vals = out.cpu().tolist()
    expected = [1, 0]
    ok = vals == expected
    detail = f"observed {vals}, expected {expected}"
    _format_status("Distributed shared memory (cluster map_shared_rank)", ok, detail if not ok else None)
    return ok


def _warp_async_copy_test(ext: object) -> bool:
    n = 256
    src = torch.arange(n, device="cuda", dtype=torch.float32)
    dst = torch.zeros_like(src)
    ext.run_warp_async_copy(src, dst)
    ok = torch.allclose(dst, src + 1.0, rtol=0, atol=0)
    if not ok:
        diff = (dst - (src + 1.0)).abs().max().item()
        _format_status("Warp-level async copy (cp.async)", False, f"max abs diff {diff}")
    else:
        _format_status("Warp-level async copy (cp.async)", True)
    return ok


def _warp_specialization_test(ext: object) -> bool:
    n = 512
    src = torch.arange(n, device="cuda", dtype=torch.float32)
    dst = torch.full_like(src, -123.0)
    ext.run_warp_specialization(src, dst)
    expected = src * 2.0 + 1.0
    ok = torch.allclose(dst, expected, rtol=0, atol=0)
    detail = None
    if not ok:
        max_diff = (dst - expected).abs().max().item()
        detail = f"max abs diff {max_diff}"
    _format_status("Warp specialization pipeline", ok, detail)
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping Blackwell CUDA verification.")
        return 0

    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        print(f"Compute capability {major}.{minor} lacks cluster/TMA features; skipping.")
        return 0

    print("Blackwell CUDA Feature Verification")
    print("===================================")
    print(f"Detected GPU compute capability: {major}.{minor}\n")

    try:
        ext = _load_extension()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to build CUDA verification extension: {exc}")
        return 1

    results = [
        _dsmem_test(ext),
        _warp_async_copy_test(ext),
        _warp_specialization_test(ext),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
