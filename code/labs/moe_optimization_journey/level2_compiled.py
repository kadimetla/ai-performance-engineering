#!/usr/bin/env python3
"""Level 2: torch.compile - The Grand Finale!

ADDS: TorchInductor kernel fusion.
- Automatically fuses operations across einsum/attention
- Generates optimized CUDA kernels
- Compounds with batched execution for ~28x total speedup!

Cumulative: batched + torch.compile
This is the FULLY OPTIMIZED version.
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
import torch

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level2Compiled(MoEJourneyBenchmark):
    """Level 2: + torch.compile (the finale!)."""
    LEVEL = 2

def get_benchmark() -> Level2Compiled:
    return Level2Compiled()


if __name__ == "__main__":
    run_level(2)
