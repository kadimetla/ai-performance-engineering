#!/usr/bin/env python3
"""Optimized MoE Level 4: Triton Fused Kernels.

Optimization: Custom Triton kernels for fused expert computation.
Expected speedup: ~1.3x over Level 2
"""
from labs.moe_optimization_journey.level4_triton import Level4Triton, get_benchmark

__all__ = ["Level4Triton", "get_benchmark"]

