"""Blackwell matmul helper functions exposed to Python."""

from __future__ import annotations

from .grace_blackwell_extension import load_grace_blackwell_module


def _module():
    return load_grace_blackwell_module()


def baseline_blackwell_matmul(a, b):
    """Naive FP16 matmul that mirrors the Part 1 roofline baseline."""
    return _module().baseline_blackwell_matmul(a, b)


def optimized_blackwell_matmul_pseudo(a, b):
    """Part 2 port: warp-specialized loads + shared-memory staging (emulated TMA)."""
    return _module().optimized_blackwell_matmul_pseudo(a, b)


def optimized_blackwell_matmul_tma(a, b):
    """Part 2 port: real TMA path; fails fast if TMA unsupported."""
    return _module().optimized_blackwell_matmul_tma(a, b)


def optimized_blackwell_matmul_pipeline(a, b):
    """Part 3 port: multi-tile accumulation with asynchronous pipelines."""
    return _module().optimized_blackwell_matmul_pipeline(a, b)


def optimized_blackwell_matmul_cluster(a, b):
    """Part 4 port: cluster DSMEM broadcast + cooperative store."""
    return _module().optimized_blackwell_matmul_cluster(a, b)


def is_cluster_launch_supported() -> bool:
    """Expose runtime capability detection so notebooks can branch."""
    return bool(_module().cluster_launch_supported())


def is_tma_supported() -> bool:
    """Blackwell TMA support (SM100+)."""
    return bool(_module().tma_supported())


__all__ = [
    "baseline_blackwell_matmul",
    "optimized_blackwell_matmul_pseudo",
    "optimized_blackwell_matmul_tma",
    "optimized_blackwell_matmul_pipeline",
    "optimized_blackwell_matmul_cluster",
    "is_cluster_launch_supported",
    "is_tma_supported",
]
