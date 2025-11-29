#!/usr/bin/env python3
"""Level 4: Token Sorting.

ADDS: Sort tokens by expert for memory coalescing.
- Consecutive tokens processed by same expert
- Better cache utilization
- May provide additional speedup on top of previous levels

Cumulative: batched + torch.compile + FP8 + sorting
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level4Sorted(MoEJourneyBenchmark):
    """Level 4: + Token sorting."""
    LEVEL = 4


def get_benchmark() -> Level4Sorted:
    return Level4Sorted()


if __name__ == "__main__":
    run_level(4)

