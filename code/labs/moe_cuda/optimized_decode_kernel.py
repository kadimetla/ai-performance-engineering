"""Benchmark wrapper for the optimized CUDA decode kernel."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

from labs.moe_cuda.decode_kernels import (
    optimized_kernel_supported,
    run_optimized_kernel,
)


class OptimizedDecodeKernelBenchmark(BaseBenchmark):
    """Runs the TMA double-buffered CUDA decode kernel."""

    def __init__(self) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("labs.moe_cuda decode kernels require CUDA")
        self.rows = 4096
        self.cols = 1024
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        tokens = self.rows * self.cols
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(7)
        self.input = torch.randn(
            self.rows,
            self.cols,
            device=self.device,
            dtype=torch.float32,
        )
        self.output = torch.zeros_like(self.input)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.input is None or self.output is None:
            raise RuntimeError("Decode tensors not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_cuda_decode_kernel_optimized", enable=enable_nvtx):
            run_optimized_kernel(self.input, self.output)
        torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.input = None
        self.output = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=3)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.input is None or self.output is None:
            return "Decode tensors missing"
        return None


def get_benchmark() -> BaseBenchmark:
    candidate = OptimizedDecodeKernelBenchmark()
    if not optimized_kernel_supported(candidate.rows, candidate.cols):
        class SkipBenchmark(BaseBenchmark):
            def setup(self) -> None:  # pragma: no cover
                arch = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
                raise RuntimeError(
                    f"SKIPPED: labs.moe_cuda optimized decode kernel requires Hopper/Blackwell "
                    f"with CUDA 13+ TMA; current GPU sm_{arch[0]}{arch[1]} lacks support."
                )

            def benchmark_fn(self) -> None:  # pragma: no cover
                return

            def get_config(self) -> BenchmarkConfig:
                return BenchmarkConfig(iterations=1, warmup=1)

        return SkipBenchmark()
    return candidate
