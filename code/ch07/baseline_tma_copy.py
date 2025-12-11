"""Python harness wrapper for baseline_tma_copy.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineTMACopyBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline TMA descriptor copy kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        n_elems = 1 << 25
        lookahead = 64
        redundant_reads = 8
        bytes_per_element = (redundant_reads * 3 + 1) * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_tma_copy",
            friendly_name="Baseline TMA Copy",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "N": n_elems,
                "lookahead": lookahead,
                "redundant_reads": redundant_reads,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(
            bytes_per_iteration=float(n_elems * bytes_per_element),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics."""
        return None

def get_benchmark() -> BaselineTMACopyBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineTMACopyBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
