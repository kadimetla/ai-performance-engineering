"""Helpers for temporarily disabling Inductor's CUDA graph features.

Several benchmarks launch multiple compiled kernels across independent CUDA
streams.  Inductor's cudagraph helpers are not stream-safe on Grace-Blackwell
yet, so we need a reusable guard that toggles the global settings while a
benchmark is active.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

# Ensure compile_utils side effects (e.g., TLS patches) have been applied before
# we start toggling Inductor configuration.
from core.utils import compile_utils as _compile_utils_patch  # noqa: F401

import torch

InductorCudagraphState = Optional[Tuple[Any, Dict[str, Any]]]


def disable_inductor_cudagraph_features() -> InductorCudagraphState:
    """Force-disable Inductor's cudagraph helpers and return previous state."""
    try:
        import torch._inductor.config as inductor_config  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        return None

    triton_cfg = getattr(inductor_config, "triton", None)
    if triton_cfg is None:
        return None

    previous: Dict[str, Any] = {}
    changed = False
    for attr in ("cudagraphs", "cudagraph_trees"):
        if hasattr(triton_cfg, attr):
            previous[attr] = getattr(triton_cfg, attr)
            setattr(triton_cfg, attr, False)
            changed = True

    if not changed:
        return None

    return (triton_cfg, previous)


def restore_inductor_cudagraph_features(state: InductorCudagraphState) -> None:
    """Restore cudagraph settings if they were overridden."""
    if not state:
        return

    triton_cfg, previous = state
    for attr, value in previous.items():
        try:
            setattr(triton_cfg, attr, value)
        except Exception:
            continue


@contextmanager
def inductor_cudagraph_guard():
    """Context manager that temporarily disables Inductor cudagraph helpers."""
    state = disable_inductor_cudagraph_features()
    try:
        yield state
    finally:
        restore_inductor_cudagraph_features(state)
