"""Baseline FP32 gradient all-reduce (communication-only INT8 compression)."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch04.gradient_compression_common import (
    GradientCompressionBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    bench = GradientCompressionBenchmark(
        compression="none",
        equivalence_group="ch04_gradient_compression_int8_comm_only",
        output_tolerance=(1e-1, 1e-1),
        tensor_size_mb=1024,
        multi_gpu=False,
        simulate_single_gpu_transfer=True,
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
