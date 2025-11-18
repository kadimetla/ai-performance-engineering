"""baseline_compute_bound.py - Compute-bound kernel baseline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

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


class BaselineComputeBoundBenchmark(BaseBenchmark):
    """Compute-heavy kernel to illustrate high arithmetic intensity."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.repeats = 16
        self.N = 4096
        tokens = self.N * self.repeats
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(self.N, self.N * 2),
            nn.ReLU(),
            nn.Linear(self.N * 2, self.N),
        ).to(self.device, dtype=torch.float16).eval()
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("baseline_compute_bound", enable=enable_nvtx):
            out = self.input
            for _ in range(self.repeats):
                out = self.model(out)
            torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.model = None
        self.input = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=12, warmup=3)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.input is None or self.model is None:
            return "Model/input not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineComputeBoundBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BaselineComputeBoundBenchmark().get_config(),
    )
    result = harness.benchmark(get_benchmark())
    print(f"Baseline compute-bound: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
