"""Alias to reuse the optimized MoE inference benchmark implemented in ch18."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch18.run_vllm_decoder import VLLMMoEInferenceBenchmark  # noqa: E402


def get_benchmark():
    return VLLMMoEInferenceBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"Optimized MoE inference mean latency: {mean_ms:.3f} ms")
