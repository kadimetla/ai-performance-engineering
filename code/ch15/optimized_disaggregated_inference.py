"""Optimized disaggregated inference benchmark with speculative decoding."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch15.baseline_disaggregated_inference import (  # noqa: E402
    _DisaggregatedInferenceBenchmark,
)


class OptimizedDisaggregatedInferenceBenchmark(_DisaggregatedInferenceBenchmark):
    """Overlap decode work by batching speculative windows."""

    def __init__(self) -> None:
        super().__init__(speculative_window=4, decode_parallelism=2)


def get_benchmark():
    return OptimizedDisaggregatedInferenceBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(result)
