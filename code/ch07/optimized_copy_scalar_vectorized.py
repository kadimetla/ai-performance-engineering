"""Python harness wrapper for ch07's optimized_copy_scalar_vectorized.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedCopyVectorizedBenchmark(CudaBinaryBenchmark):
    """Wraps the vectorized copy kernel that prints CUDA 13 timing."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        num_floats = 64 * 1024 * 1024
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_copy_scalar_vectorized",
            friendly_name="Optimized Copy Scalar Vectorized",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "N": num_floats,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(num_floats * 8))

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics."""
        return None

def get_benchmark() -> OptimizedCopyVectorizedBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedCopyVectorizedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
