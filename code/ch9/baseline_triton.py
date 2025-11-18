"""baseline_triton.py - Baseline Triton matmul wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402
try:
    import triton  # noqa: F401
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


def triton_matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, **_: int) -> None:
    """
    Fallback baseline matmul. If Triton is unavailable, use torch.matmul to keep
    the harness green; optimized path lives in the matching optimized_* file.
    """
    torch.matmul(a, b, out=out)


class BaselineTritonBenchmark(BaseBenchmark):
    """Baseline Triton matmul benchmark."""

    def __init__(self):
        super().__init__()
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.m = 1024
        self.n = 1024
        self.k = 1024
        tokens = self.m * self.n * self.k
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        self.output = torch.empty(self.m, self.n, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("baseline_triton", enable=enable_nvtx):
            triton_matmul(self.A, self.B, self.output, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)
            torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.A = None
        self.B = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=2)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineTritonBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BaselineTritonBenchmark().get_config(),
    )
    result = harness.benchmark(get_benchmark())
    print(f"Baseline Triton matmul: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
