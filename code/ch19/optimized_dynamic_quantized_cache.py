"""Optimized dynamic quantized cache benchmark with adaptive bit-width."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch19.baseline_dynamic_quantized_cache import (  # noqa: E402
    _DynamicQuantizedCacheBenchmark,
)


class OptimizedDynamicQuantizedCacheBenchmark(_DynamicQuantizedCacheBenchmark):
    """Switch to 4-bit cache segments for late decode plus simulated offload."""

    def __init__(self) -> None:
        schedule = [8] * 12 + [6] * 8 + [4] * 12
        super().__init__(schedule_bits=schedule)


def get_benchmark():
    return OptimizedDynamicQuantizedCacheBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(result)
