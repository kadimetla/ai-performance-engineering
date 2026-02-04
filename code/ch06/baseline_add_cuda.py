"""Python harness wrapper for baseline_add_cuda.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineAddCudaBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUDA add binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        n = 1_000_000
        bytes_per_iter = n * 3 * 4  # A, B, C (float32)
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_add_cuda",
            friendly_name="Baseline Add CUDA",
            iterations=10,
            warmup=5,
            timeout_seconds=60,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
            workload_params={
                "N": n,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(bytes_per_iter))

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineAddCudaBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
