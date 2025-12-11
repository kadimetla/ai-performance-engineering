"""Python harness wrapper for optimized_tma_bulk_tensor_2d.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedTMABulkTensor2D(CudaBinaryBenchmark):
    """Wraps the cp.async.bulk.tensor 2D path (TMA-backed)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        width = 2048
        height = 2048
        tile_m = 64
        tile_n = 32
        bytes_per_matrix = width * height * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_tma_bulk_tensor_2d",
            friendly_name="Optimized 2D tensor copy (TMA bulk)",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            require_tma_instructions=True,
            workload_params={
                "width": width,
                "height": height,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(bytes_per_matrix * 2))

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics."""
        return None

def get_benchmark() -> OptimizedTMABulkTensor2D:
    """Factory for discover_benchmarks()."""
    return OptimizedTMABulkTensor2D()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
