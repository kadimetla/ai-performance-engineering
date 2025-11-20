"""Optimized target: bm=128, bn=256, bk=64 (smem-heavy)."""

from __future__ import annotations

from labs.occupancy_tuning.triton_matmul_schedules import (
    EXTRA_SCHEDULE,
    TritonMatmulProtonBenchmark,
)


class OptimizedProtonMatmulBM128BN256BK64Benchmark(TritonMatmulProtonBenchmark):
    def __init__(self) -> None:
        super().__init__(schedule=EXTRA_SCHEDULE, iterations=2, warmup=1)


def get_benchmark() -> OptimizedProtonMatmulBM128BN256BK64Benchmark:
    return OptimizedProtonMatmulBM128BN256BK64Benchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"\nProton matmul (bm128_bn256_bk64): {mean_ms:.3f} ms")
