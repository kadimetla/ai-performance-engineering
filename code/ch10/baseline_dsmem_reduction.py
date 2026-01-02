"""Python harness wrapper for baseline_dsmem_reduction.cu - Two-Pass Reduction Baseline."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature
class BaselineDSMEMReductionBenchmark(CudaBinaryBenchmark):
    """Wraps the two-pass reduction kernel (baseline for DSMEM comparison)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        workload_n = 16 * 1024 * 1024
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_dsmem_reduction",
            friendly_name="Two-Pass Reduction (Baseline)",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "batch_size": 1024,
                "dtype": "float32",
                "N": workload_n,
                "cluster_size": 4,
                "block_elems": 4096,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(workload_n * 4))

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def get_input_signature(self) -> dict:
        """Signature for two-pass reduction baseline."""
        return simple_signature(
            batch_size=1,
            dtype="float32",
            N=16 * 1024 * 1024,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)
def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineDSMEMReductionBenchmark()
if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
