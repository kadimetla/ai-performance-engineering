"""Multi-GPU wrapper for the NVLink benchmark that skips on single-GPU hosts."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch4.baseline_nvlink import BaselineNVLinkBenchmark


def get_benchmark() -> BaselineNVLinkBenchmark:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("SKIPPED: baseline_nvlink_multigpu requires >=2 GPUs")
    return BaselineNVLinkBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=3, warmup=1))
    result = harness.benchmark(bench)
    print(f"NVLink baseline (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
