"""Baseline prefill/decode wrapper that skips when <2 GPUs are available."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch17.baseline_prefill_decode_disagg import BaselineInferenceMonolithicBenchmark


def get_benchmark() -> BaselineInferenceMonolithicBenchmark:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("SKIPPED: prefill/decode multigpu requires >=2 GPUs")
    return BaselineInferenceMonolithicBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=3, warmup=1))
    result = harness.benchmark(benchmark)
    print(f"Prefill/decode baseline (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
