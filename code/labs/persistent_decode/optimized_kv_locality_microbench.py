"""Optimized (alias) wrapper for KV locality H2D microbench."""

from __future__ import annotations

from typing import Dict, Optional

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.kv_locality_microbench import KvLocalityMicrobench


class OptimizedKvLocalityMicrobench(BaseBenchmark):
    """Runs the KV locality microbench under benchmark_cli (same as baseline)."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}

    def setup(self) -> None:
        return

    def benchmark_fn(self) -> None:
        bench = KvLocalityMicrobench()
        bench.setup()
        bench.benchmark_fn()
        self._summary = bench.get_custom_metrics() or {}

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=0)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None


def get_benchmark() -> BaseBenchmark:
    return OptimizedKvLocalityMicrobench()


if __name__ == "__main__":
    b = get_benchmark()
    b.setup()
    b.benchmark_fn()
    print(b.get_custom_metrics())
