"""Multi-GPU wrapper for the optimized NVLink benchmark that skips on single-GPU hosts."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch4.optimized_nvlink import OptimizedNVLinkBenchmark


def get_benchmark() -> OptimizedNVLinkBenchmark:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("SKIPPED: optimized_nvlink_multigpu requires >=2 GPUs")
    return OptimizedNVLinkBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=3, warmup=1))
    result = harness.benchmark(bench)
    print(f"NVLink optimized (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
