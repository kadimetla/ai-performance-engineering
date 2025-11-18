"""optimized_cublas.py - Pure cuBLAS matmul with TF32 tensor-core acceleration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.compile_utils import enable_tf32, configure_tf32, restore_tf32
from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class OptimizedCublasBenchmark(BaseBenchmark):
    """
    Optimized: pure cuBLAS GEMM with TF32 and warmed-up heuristics.

    This keeps the math in FP32 but lets cuBLAS route workloads through tensor cores
    (TF32) while running a few warmup matmuls so Lt heuristics cache the best kernel.
    """

    def __init__(self):
        super().__init__()
        self.m = 2048
        self.n = 2048
        self.k = 2048
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self._tf32_state: Optional[Tuple[Optional[str], Optional[str]]] = None
        self._prev_precision: Optional[str] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )

    def setup(self) -> None:
        """Enable TF32, allocate FP32 matrices, and warm up cuBLAS."""
        self._prev_precision = torch.get_float32_matmul_precision()

        enable_tf32()
        self._tf32_state = configure_tf32(enable_matmul=True, enable_cudnn=True)
        torch.set_float32_matmul_precision("high")

        torch.manual_seed(42)
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)

        # Warmup a handful of GEMMs so cuBLAS Lt heuristics settle before measurement.
        for _ in range(10):
            _ = torch.matmul(self.A, self.B)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """cuBLAS TF32 GEMM."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("cublas", enable=enable_nvtx):
            _ = torch.matmul(self.A, self.B)
        self._synchronize()

    def teardown(self) -> None:
        """Restore TF32 knobs and free tensors."""
        self.A = None
        self.B = None
        if self._tf32_state is not None:
            restore_tf32(self._tf32_state)
            self._tf32_state = None
        if self._prev_precision is not None:
            torch.set_float32_matmul_precision(self._prev_precision)  # type: ignore[arg-type]
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedCublasBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5),
    )
    result = harness.benchmark(benchmark)
    timing = result.timing.mean_ms if result.timing else 0.0
    print(f"\nOptimized cuBLAS (TF32): {timing:.3f} ms")
