"""Python harness wrapper for optimized_hbm3e_copy.cu."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedHBM3ECopyBenchmark(CudaBinaryBenchmark):
    """Wraps the vectorized HBM3e copy kernel as a benchmark."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_hbm3e_copy",
            friendly_name="Optimized HBM3e Copy",
            iterations=3,
            warmup=1,
            timeout_seconds=90,
        )


def get_benchmark() -> OptimizedHBM3ECopyBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedHBM3ECopyBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized HBM3e Copy: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

