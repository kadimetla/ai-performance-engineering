#!/usr/bin/env python3
"""Optimized MoE: Level 2 (FP8)."""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.level2_fp8 import Level2FP8, get_benchmark
__all__ = ["Level2FP8", "get_benchmark"]
