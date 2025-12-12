#!/usr/bin/env python3
"""Level 3: Grouped GEMM.

ADDS: Per-expert GEMM on sorted tokens.
This is how CUTLASS grouped GEMM works in production MoE.

- Each expert gets contiguous token batch
- Better GPU utilization per expert
- Enables expert-level parallelism

Cumulative: batched + permuted + grouped
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
import torch

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level3Grouped(MoEJourneyBenchmark):
    """Level 3: + Grouped GEMM per expert."""
    LEVEL = 3

def get_benchmark() -> Level3Grouped:
    return Level3Grouped()


if __name__ == "__main__":
    run_level(3)



