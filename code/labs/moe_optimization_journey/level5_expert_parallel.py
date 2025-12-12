#!/usr/bin/env python3
"""Level 5: Expert Parallel variant aligned with the baseline dimensions.

This keeps the same batch/sequence sizes as the baseline while enabling the
Level 5 optimizations (grouped routing + fused BMM path). It exists to give
the harness a concrete module for optimized_moe_expert_parallel.py to import.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
import torch

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level5ExpertParallel(MoEJourneyBenchmark):
    """Expert-parallelized MoE with Level 5 optimizations."""

    LEVEL = 5  # Reuse Level 5 stack (BMM fusion) but keep baseline shapes.

def get_benchmark() -> Level5ExpertParallel:
    return Level5ExpertParallel()


if __name__ == "__main__":
    run_level(5)
