#!/usr/bin/env python3
"""Level 3: torch.compile - The Grand Finale!

ADDS: TorchInductor kernel fusion.
- Automatically fuses operations
- Generates optimized CUDA kernels
- The best compound optimization!

Cumulative: ALL previous optimizations + torch.compile
This is the FULLY OPTIMIZED version achieving ~24x speedup!
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level3Compiled(MoEJourneyBenchmark):
    """Level 3: + torch.compile (the finale!)."""
    LEVEL = 3


def get_benchmark() -> Level3Compiled:
    return Level3Compiled()


if __name__ == "__main__":
    run_level(3)

