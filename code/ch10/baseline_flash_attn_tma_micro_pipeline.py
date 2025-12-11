"""Baseline FlashAttention-style micro-pipeline with blocking copies."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class BaselineFlashAttnTmaMicroPipelineBenchmark(CudaBinaryBenchmark):
    """Runs the baseline flash-attn micro-pipeline (no async copies/TMA)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_flash_attn_tma_micro_pipeline",
            friendly_name="FlashAttn Micro-Pipeline Baseline (blocking copies)",
            iterations=1,
            warmup=5,
            timeout_seconds=120,
            run_args=(),
            # Require pipeline-capable/TMA GPUs to keep A/B runs aligned; SKIP otherwise.
            requires_pipeline_api=True,
            workload_params={
                "batch_size": 2048,
                "dtype": "float32",
                "seq_len": 2048,
                "d_head": 64,
                "tile_kv": 40,
                "threads": 128,
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
        """Signature for FlashAttn micro-pipeline baseline."""
        return simple_signature(
            batch_size=2048,
            dtype="float32",
            seq_len=2048,
            d_head=64,
            tile_kv=40,
            threads=128,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

def get_benchmark() -> BaseBenchmark:
    return BaselineFlashAttnTmaMicroPipelineBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
