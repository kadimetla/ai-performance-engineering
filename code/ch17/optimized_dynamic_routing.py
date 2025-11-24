"""Vectorized dynamic routing benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch17.baseline_dynamic_routing import (  # noqa: E402
    _DynamicRoutingBenchmark,
)


class OptimizedDynamicRoutingBenchmark(_DynamicRoutingBenchmark):
    def __init__(self) -> None:
        super().__init__(batch_size=64, vectorized=True)


def get_benchmark():
    return OptimizedDynamicRoutingBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(result)
