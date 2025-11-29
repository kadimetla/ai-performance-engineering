"""Centralized smoke-test toggle used by benchmarks and harness."""

from __future__ import annotations

_SMOKE_MODE = False


def set_smoke_mode(enabled: bool) -> None:
    """Globally enable or disable smoke-test mode."""
    global _SMOKE_MODE
    _SMOKE_MODE = bool(enabled)


def is_smoke_mode() -> bool:
    """Return True when smoke-test mode is active."""
    return _SMOKE_MODE
