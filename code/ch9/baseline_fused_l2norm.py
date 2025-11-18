"""Python harness wrapper for baseline_fused_l2norm.cu."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineFusedL2NormBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline fused L2 norm kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_fused_l2norm",
            friendly_name="Baseline Fused L2 Norm",
            iterations=5,
            warmup=1,
            timeout_seconds=90,
        )


def get_benchmark() -> BaselineFusedL2NormBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineFusedL2NormBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Fused L2 Norm: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

