"""Python harness wrapper for baseline_tma_multicast.cu - Standard GEMM without Multicast."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class BaselineTMAMulticastBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline tiled GEMM kernel without TMA multicast."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_tma_multicast",
            friendly_name="Baseline GEMM (No Multicast)",
            iterations=3,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "batch_size": 2048,
                "dtype": "float32",
                "M": 2048,
                "N": 2048,
                "K": 2048,
                "tile_m": 64,
                "tile_n": 64,
                "tile_k": 32,
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
        """Signature for baseline TMA multicast GEMM."""
        return simple_signature(
            batch_size=2048,
            dtype="float32",
            M=2048,
            N=2048,
            K=2048,
            tile_m=64,
            tile_n=64,
            tile_k=32,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineTMAMulticastBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
