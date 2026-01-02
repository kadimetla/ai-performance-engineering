"""Python harness wrapper for optimized_warp_spec_pingpong.cu - Ping-Pong Warp Specialization."""

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
class OptimizedWarpSpecPingPongBenchmark(CudaBinaryBenchmark):
    """Wraps the ping-pong warp specialization GEMM kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_warp_spec_pingpong",
            friendly_name="Ping-Pong Warp Specialization",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "batch_size": 4096,
                "dtype": "float32",
                "M": 4096,
                "N": 4096,
                "K": 512,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(4096 * 4096 * 4))

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def get_input_signature(self) -> dict:
        """Signature for ping-pong warp specialization kernel."""
        return simple_signature(
            batch_size=4096,
            dtype="float32",
            M=4096,
            N=4096,
            K=512,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)
def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedWarpSpecPingPongBenchmark()
if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
