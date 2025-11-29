#!/usr/bin/env python3
"""Optimized MoE Level 5: Expert Parallelism.

Optimization: Multi-stream expert execution for overlapped compute.
Expected speedup: ~1.2x over Level 4
"""
from labs.moe_optimization_journey.level5_expert_parallel import Level5ExpertParallel, get_benchmark

__all__ = ["Level5ExpertParallel", "get_benchmark"]

