"""Python harness wrapper for baseline CUTLASS GEMM binary."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineCutlassGemmBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUTLASS GEMM kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cutlass_gemm",
            friendly_name="Baseline CUTLASS GEMM",
            iterations=5,
            warmup=1,
            timeout_seconds=120,
        )


def get_benchmark() -> BaselineCutlassGemmBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineCutlassGemmBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline CUTLASS GEMM: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
