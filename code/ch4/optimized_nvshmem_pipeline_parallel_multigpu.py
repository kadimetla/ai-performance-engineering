"""Optimized (placeholder) NVSHMEM pipeline parallel wrapper; skips on <2 GPUs.

This keeps the harness happy with a baseline/optimized pair. When you add a
true optimization (e.g., tuned block sizes or IBGDA), swap the call inside
benchmark_fn accordingly.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from ch4.nvshmem_pipeline_parallel import main as nvshmem_main
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class OptimizedNVSHMEMPipelineParallelMultiGPU(BaseBenchmark):
    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: nvshmem_pipeline_parallel requires >=2 GPUs")

    def benchmark_fn(self) -> None:
        # TODO: swap in tuned parameters when available.
        nvshmem_main()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=1, measurement_timeout_seconds=300)


def get_benchmark() -> BaseBenchmark:
    return OptimizedNVSHMEMPipelineParallelMultiGPU()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(f"NVSHMEM pipeline parallel optimized (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
