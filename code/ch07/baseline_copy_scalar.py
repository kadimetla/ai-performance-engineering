"""Python harness wrapper for ch07's baseline_copy_scalar.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineCopyScalarBenchmark(CudaBinaryBenchmark):
    """Wraps the scalar copy kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        num_floats = 64 * 1024 * 1024
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_copy_scalar",
            friendly_name="Baseline Copy Scalar",
            iterations=3,
            warmup=5,
            timeout_seconds=90,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
            workload_params={
                "N": num_floats,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(num_floats * 8))

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics."""
        return None

def get_benchmark() -> BaselineCopyScalarBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineCopyScalarBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
