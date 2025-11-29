#!/usr/bin/env python3
"""Level 5: torch.compile - The Grand Finale!

ADDS: TorchInductor kernel fusion on top of all previous optimizations.
- Fuses operations across the entire forward pass
- Generates optimized CUDA/Triton kernels
- The compound effect of all techniques!

Cumulative: ALL previous optimizations + torch.compile
This is the FULLY OPTIMIZED version.
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level5Compiled(MoEJourneyBenchmark):
    """Level 5: + torch.compile (the finale!)."""
    LEVEL = 5


def get_benchmark() -> Level5Compiled:
    return Level5Compiled()


if __name__ == "__main__":
    run_level(5)
