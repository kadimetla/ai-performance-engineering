"""Lightweight numba stub used when the real numba wheel is unavailable/incompatible.

vLLM imports numba in some code paths (e.g., ngram proposer), but our environment
ships NumPy 2.3 while current numba wheels require <=2.2. To avoid an ImportError
we provide the tiny surface area vLLM touches.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "get_num_threads",
    "jit",
    "njit",
    "prange",
    "set_num_threads",
]


def get_num_threads() -> int:
    return 1


def set_num_threads(_n: int) -> None:
    return None


def _decorator(fn: Callable | None = None, **_kwargs: Any):
    if fn is None:
        def wrapper(inner: Callable) -> Callable:
            return inner

        return wrapper
    return fn


jit = _decorator
njit = _decorator
prange = range
