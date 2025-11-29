"""Shared GPU requirement helpers for benchmarks and examples.

Use these helpers to enforce multi-GPU/NVLink prerequisites consistently.
"""

from __future__ import annotations

import sys

import torch


def skip_if_insufficient_gpus(min_gpus: int = 2) -> None:
    """Raise a standardized SKIPPED RuntimeError when not enough GPUs exist."""
    available = torch.cuda.device_count()
    if available < min_gpus:
        raise RuntimeError(
            f"SKIPPED: Distributed benchmark requires multiple GPUs (found {available} GPU)"
        )


def require_min_gpus(min_gpus: int, script_name: str | None = None) -> None:
    """
    Raise a standardized SKIPPED RuntimeError when GPU count is insufficient.

    Prefer this helper for standalone scripts that previously exited; the harness
    will record the SKIPPED status instead of failing the run.
    """
    available = torch.cuda.device_count()
    if available < min_gpus:
        raise RuntimeError(
            f"SKIPPED: Requires >= {min_gpus} GPUs (found {available} GPU)"
        )


def warn_optimal_gpu_count(optimal_gpus: int, script_name: str | None = None) -> None:
    """Emit a warning if the system has fewer GPUs than the optimal configuration."""
    available = torch.cuda.device_count()
    if available < optimal_gpus:
        script = script_name or sys.argv[0]
        print(f"\nWARNING: {script}", file=sys.stderr)
        print(
            f"    Optimized for {optimal_gpus} GPUs but only {available} available; performance may be suboptimal.\n",
            file=sys.stderr,
        )


def require_peer_access(src: int, dst: int, script_name: str | None = None) -> None:
    """Fail if peer access between two devices is unavailable."""
    if torch.cuda.device_count() <= max(src, dst):
        require_min_gpus(max(src, dst) + 1, script_name=script_name)
    if not torch.cuda.can_device_access_peer(src, dst):
        raise RuntimeError(f"SKIPPED: Peer access unavailable between GPU {src} and GPU {dst}")
