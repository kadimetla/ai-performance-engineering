#!/usr/bin/env python3
"""Level 3: Token Sorting for Memory Coalescing.

ADDS: Sort tokens by expert before computation.
- Memory coalescing: consecutive tokens go to same expert
- Better cache utilization
- ~1.2-1.4x speedup over Level 2

Cumulative: Level 0 + batched + torch.compile + token sorting
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
import torch

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level3Sorted(MoEJourneyBenchmark):
    """Level 3: + Token sorting."""
    LEVEL = 3

def get_benchmark() -> Level3Sorted:
    return Level3Sorted()


if __name__ == "__main__":
    run_level(3)
