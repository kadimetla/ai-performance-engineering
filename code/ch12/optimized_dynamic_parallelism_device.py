"""Optimized dynamic-parallelism device launch benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedDynamicParallelismDeviceBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized device-side launcher binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_dynamic_parallelism_device",
            friendly_name="Dynamic Parallelism (device launches, optimized)",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            time_regex=r"Elapsed_ms:\s*([0-9.]+)",
        )

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_pipeline_metrics

        return compute_pipeline_metrics(
            num_stages=getattr(self, "num_stages", 1),
            stage_times_ms=[getattr(self, "_last_result", None).time_ms if getattr(self, "_last_result", None) else 0.0],
        )


def get_benchmark() -> CudaBinaryBenchmark:
    return OptimizedDynamicParallelismDeviceBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
