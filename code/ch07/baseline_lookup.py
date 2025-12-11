"""Python harness wrapper for ch07's baseline_lookup.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineLookupBenchmark(CudaBinaryBenchmark):
    """Wraps the scattered memory lookup baseline."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        n_elems = 1 << 20
        random_steps = 64
        iterations = 200
        bytes_per_element = (random_steps + 1) * 4  # random reads + write
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_lookup",
            friendly_name="Ch7 Baseline Lookup",
            iterations=3,
            warmup=5,
            timeout_seconds=90,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
            workload_params={
                "N": n_elems,
                "random_steps": random_steps,
                "iterations": iterations,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(
            bytes_per_iteration=float(n_elems * bytes_per_element),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics."""
        return None

def get_benchmark() -> BaselineLookupBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineLookupBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
