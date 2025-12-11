"""Python harness wrapper for baseline_hbm_copy.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineHBMCopyBenchmark(CudaBinaryBenchmark):
    """Wraps the scalar HBM copy kernel as a benchmark."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        size_bytes = 256 * 1024 * 1024
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_hbm_copy",
            friendly_name="Baseline HBM Copy",
            iterations=3,
            warmup=5,
            timeout_seconds=90,
            workload_params={
                "bytes": size_bytes,
                "dtype": "float32",
            },
        )
        # Register workload metadata for compliance
        self.register_workload_metadata(bytes_per_iteration=float(size_bytes * 2))

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics."""
        return None  # Metrics computed by CUDA binary

def get_benchmark() -> BaselineHBMCopyBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineHBMCopyBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
