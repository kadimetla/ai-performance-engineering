"""baseline_warp_specialization_producer_consumer.py - Producer/consumer without specialization."""

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


class BaselineWarpSpecializationProducerConsumerBenchmark(BaseBenchmark):
    """Baseline: single buffer, producer/consumer in one warp group."""

    def __init__(self):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.repeats = 16
        self.N = 1_048_576
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * self.repeats),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        self.data = torch.randn(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("baseline_warp_specialization", enable=enable_nvtx):
            for _ in range(self.repeats):
                # Simulate producer/consumer with a single warp set
                transformed = self.data * 1.0003 + 0.0007
                self.data = transformed
            torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.data = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=12, warmup=3)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.data is None:
            return "Data tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineWarpSpecializationProducerConsumerBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BaselineWarpSpecializationProducerConsumerBenchmark().get_config(),
    )
    result = harness.benchmark(get_benchmark())
    print(f"Baseline warp specialization (producer/consumer): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
