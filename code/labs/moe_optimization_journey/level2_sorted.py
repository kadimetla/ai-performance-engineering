#!/usr/bin/env python3
"""Level 2: Token Sorting for Memory Coalescing.

ADDS: Sort tokens by expert before computation.
- Groups tokens going to same expert together
- Better memory access patterns (coalescing)
- Reduces random memory access overhead

Cumulative: batched + token sorting
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
import torch

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level2Sorted(MoEJourneyBenchmark):
    """Level 2: + Token sorting."""
    LEVEL = 2

def get_benchmark() -> Level2Sorted:
    return Level2Sorted()


if __name__ == "__main__":
    run_level(2)
