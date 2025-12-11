"""Optimized dynamic-parallelism device launch benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


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
            workload_params={
                "batch_size": 262144,
                "dtype": "float32",
                "elements": 262144,
                "segment_size": 256,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=1024 * 1024)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics."""
        return None

    def get_input_signature(self) -> dict:
        return simple_signature(
            batch_size=262144,
            dtype="float32",
            elements=262144,
            segment_size=256,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)



def get_benchmark() -> CudaBinaryBenchmark:
    return OptimizedDynamicParallelismDeviceBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
