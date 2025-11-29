#!/usr/bin/env python3
"""Optimized MoE Level 6: CUDA Graphs.

Optimization: Capture forward pass in CUDA graph for zero launch overhead.
Expected speedup: ~1.3-2x over Level 4
"""
from labs.moe_optimization_journey.level6_full_stack import Level6FullStack, get_benchmark

__all__ = ["Level6FullStack", "get_benchmark"]

