"""Baseline symmetric memory analogue (single GPU)."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch04.single_gpu_transfer_common import SingleGPUTransferBenchmark, attach_benchmark_metadata


def get_benchmark() -> BaseBenchmark:
    bench = SingleGPUTransferBenchmark(
        size_mb=256,
        inner_iterations=40,
        num_chunks=8,
        use_streams=False,
        sync_per_chunk=True,
        collective_type="symmetric_memory",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
