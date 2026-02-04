"""Python harness wrapper for baseline_hbm_copy.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineHbmCopyBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        size_bytes = 256 * 1024 * 1024
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_hbm_copy",
            friendly_name="Baseline Hbm Copy",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "bytes": size_bytes,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(size_bytes * 2))

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineHbmCopyBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
