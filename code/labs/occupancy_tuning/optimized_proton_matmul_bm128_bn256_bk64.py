#!/usr/bin/env python3
"""Optimized: Triton matmul with wide-N tile for bandwidth-bound cases.

Uses 128x256x64 blocks with more warps for high throughput.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from labs.occupancy_tuning.triton_matmul_schedules import (
    MatmulSchedule,
    TritonMatmulProtonBenchmark,
)

SCHEDULE = MatmulSchedule(
    name="bm128_bn256_bk64",
    block_m=128,
    block_n=256,
    block_k=64,
    num_warps=8,
    notes="Wide-N tile with more warps for higher throughput on large matrices.",
)


class OptimizedProtonMatmulWideN(TritonMatmulProtonBenchmark):
    """Optimized Triton matmul with wide-N tile configuration.
    
    Block config: 128x256x64, 8 warps
    Use case: Higher throughput via larger output tile and more warps.
    """

    def __init__(self, size: int = 4096):
        super().__init__(
            schedule=SCHEDULE,
            size=size,
            iterations=10,
            warmup=5,
        )

def get_benchmark() -> BaseBenchmark:
    return OptimizedProtonMatmulWideN()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
