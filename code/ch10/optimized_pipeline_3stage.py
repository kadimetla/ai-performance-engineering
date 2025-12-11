"""Python harness wrapper for optimized_pipeline_3stage.cu - 3-Stage Software Pipeline."""

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
class OptimizedPipeline3StageBenchmark(CudaBinaryBenchmark):
    """Wraps the 3-stage pipeline GEMV kernel for deeper latency hiding."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_pipeline_3stage",
            friendly_name="3-Stage Pipeline GEMV",
            iterations=10,
            warmup=5,  # Minimum warmup for CUDA binary
            timeout_seconds=120,
            workload_params={
                "batch_size": 8,
                "dtype": "float32",
                "elements": 16 * 1024 * 1024,
                "segments": 8,
                "segment_size": 2 * 1024 * 1024,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=1024 * 1024)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def get_input_signature(self) -> dict:
        """Signature for optimized 3-stage pipeline."""
        return simple_signature(
            batch_size=8,
            dtype="float32",
            elements=16 * 1024 * 1024,
            segments=8,
            segment_size=2 * 1024 * 1024,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)
def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedPipeline3StageBenchmark()
if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
