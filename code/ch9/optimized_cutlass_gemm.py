"""Python harness wrapper for optimized_cutlass_gemm.cu."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedCutlassGemmBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUTLASS GEMM driver."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_cutlass_gemm",
            friendly_name="Optimized CUTLASS GEMM",
            iterations=3,
            warmup=1,
            timeout_seconds=120,
        )


def get_benchmark() -> OptimizedCutlassGemmBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedCutlassGemmBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized CUTLASS GEMM: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

