"""Python harness wrapper for baseline_fused_l2norm.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineFusedL2NormBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline fused L2 norm kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        n_elems = 1 << 20
        iterations = 100
        bytes_per_elem = 12  # two reads + one write (float32)
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_fused_l2norm",
            friendly_name="Baseline Fused L2 Norm",
            iterations=5,
            warmup=5,
            timeout_seconds=90,
            workload_params={
                "N": n_elems,
                "iterations": iterations,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(
            bytes_per_iteration=float(n_elems * bytes_per_elem),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline metrics."""
        return None



def get_benchmark() -> BaselineFusedL2NormBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineFusedL2NormBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
