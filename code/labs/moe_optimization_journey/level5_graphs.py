#!/usr/bin/env python3
"""Level 5: CUDA Graphs.

ADDS: Graph capture via reduce-overhead mode.
- Eliminates kernel launch overhead
- Best for repetitive inference workloads
- Uses torch.compile(mode='reduce-overhead')

Cumulative: batched + torch.compile + FP8 + sorting + CUDA graphs
This is the FULLY OPTIMIZED version.
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level5Graphs(MoEJourneyBenchmark):
    """Level 5: + CUDA graphs."""
    LEVEL = 5


def get_benchmark() -> Level5Graphs:
    return Level5Graphs()


if __name__ == "__main__":
    run_level(5)

