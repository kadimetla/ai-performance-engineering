"""Benchmark wrapper for symmetric memory example; skips on <2 GPUs."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch4.symmetric_memory_example import main as symmetric_memory_main
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class SymmetricMemoryMultiGPU(BaseBenchmark):
    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: symmetric_memory requires >=2 GPUs")

    def benchmark_fn(self) -> None:
        symmetric_memory_main()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=1, measurement_timeout_seconds=300)


def get_benchmark() -> BaseBenchmark:
    return SymmetricMemoryMultiGPU()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(f"Symmetric memory (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
