#!/usr/bin/env python3
"""Optimized: Triton matmul with larger compute-dense tile.

Uses 128x128x64 blocks with 4 warps - better compute throughput
than the baseline small-tile config.
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
    name="bm128_bn128_bk64",
    block_m=128,
    block_n=128,
    block_k=64,
    num_warps=4,
    notes="Larger tile for better compute density - more work per block.",
)


class OptimizedProtonMatmulLargeTile(TritonMatmulProtonBenchmark):
    """Optimized Triton matmul with compute-dense large tile.
    
    Block config: 128x128x64, 4 warps
    Benefit: Larger tiles do more compute per block, improving throughput.
    """

    def __init__(self, size: int = 4096):
        super().__init__(
            schedule=SCHEDULE,
            size=size,
            iterations=10,
            warmup=5,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedProtonMatmulLargeTile()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
