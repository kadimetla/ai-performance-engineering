"""Backward-compatible shim for the shared Inductor cudagraph helpers."""

from __future__ import annotations

from core.optimization.inductor_guard import (  # noqa: F401,F403
    InductorCudagraphState,
    disable_inductor_cudagraph_features,
    inductor_cudagraph_guard,
    restore_inductor_cudagraph_features,
)

__all__ = [
    "InductorCudagraphState",
    "disable_inductor_cudagraph_features",
    "restore_inductor_cudagraph_features",
    "inductor_cudagraph_guard",
]
