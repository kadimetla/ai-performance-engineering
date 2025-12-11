"""Benchmark wrapper for the Blackwell TMA 2D pipeline sample.

This exposes the Tensor Memory Accelerator triple-overlap kernel as a harness
target named `ch10:tmem_triple_overlap`, matching the documentation.
"""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class TMEMTripleOverlapBenchmark(CudaBinaryBenchmark):
    """Wraps the CUDA 13 TMA 2D pipeline binary and parses its runtime."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="tma_2d_pipeline_blackwell",
            friendly_name="Blackwell TMA 2D Pipeline",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            time_regex=r"(?:TMA|Baseline)\s+runtime:\s*([0-9.]+)\s*ms",
            requires_pipeline_api=True,
            workload_params={
                "batch_size": 4096,
                "dtype": "float32",
                "M": 4096,
                "N": 4096,
                "tile_n": 128,
                "chunk_m": 64,
                "stages": 1,
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
        """Signature for the TMA triple-overlap sample."""
        return simple_signature(
            batch_size=4096,
            dtype="float32",
            M=4096,
            N=4096,
            tile_n=128,
            chunk_m=64,
            stages=1,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

def get_benchmark() -> TMEMTripleOverlapBenchmark:
    """Factory for benchmark discovery."""
    return TMEMTripleOverlapBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
