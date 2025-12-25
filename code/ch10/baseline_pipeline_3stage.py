"""Python harness wrapper for baseline_pipeline_3stage.cu - 2-Stage Pipeline Baseline."""

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
class BaselinePipeline3StageBenchmark(CudaBinaryBenchmark):
    """Wraps the 2-stage pipeline GEMV kernel (baseline for 3-stage comparison)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_pipeline_3stage",
            friendly_name="2-Stage Pipeline GEMV (Baseline)",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "batch_size": 8,
                "dtype": "float32",
                "elements": 8 * 1024 * 1024,
                "segments": 128,
                "segment_size": 65536,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=65536)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def get_input_signature(self) -> dict:
        """Signature for sequential 3-stage pipeline baseline."""
        return simple_signature(
            batch_size=8,
            dtype="float32",
            elements=8 * 1024 * 1024,
            segments=128,
            segment_size=65536,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)
def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselinePipeline3StageBenchmark()
if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
