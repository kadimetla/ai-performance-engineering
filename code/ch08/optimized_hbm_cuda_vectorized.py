"""Python harness wrapper for optimized_hbm_cuda_vectorized.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedHBMCudaVectorizedBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized HBM CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        rows = 4096
        cols = 2048
        bytes_per_iter = rows * cols * 4 + rows * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_hbm_cuda_vectorized",
            friendly_name="Optimized HBM CUDA (Vectorized)",
            iterations=5,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "rows": rows,
                "cols": cols,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(bytes_per_iter))

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedHBMCudaVectorizedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
