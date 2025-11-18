"""Multi-GPU wrapper for optimized continuous batching; skips when GPUs < 2."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch4.optimized_continuous_batching import OptimizedContinuousBatchingBenchmark


def get_benchmark() -> OptimizedContinuousBatchingBenchmark:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("SKIPPED: optimized_continuous_batching_multigpu requires >=2 GPUs")
    return OptimizedContinuousBatchingBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=3, warmup=1))
    result = harness.benchmark(bench)
    print(f"Continuous batching optimized (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
