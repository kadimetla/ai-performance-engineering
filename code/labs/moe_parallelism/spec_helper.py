"""CLI preset selector for the MoE parallelism lab."""

from __future__ import annotations

import sys
from typing import List

from .plan import AVAILABLE_SPEC_PRESETS, DEFAULT_SPEC_PRESET, set_active_spec_preset


def _pop_arg(argv: List[str], idx: int) -> str:
    if idx + 1 >= len(argv):
        raise ValueError(f"Missing value for {argv[idx]}")
    value = argv[idx + 1]
    del argv[idx : idx + 2]
    return value


def _pop_inline_arg(argv: List[str], idx: int) -> str:
    token = argv[idx]
    if "=" not in token:
        raise ValueError(f"Missing '=' in {token}")
    key, _, value = token.partition("=")
    if not value:
        raise ValueError(f"Missing value for {key}")
    del argv[idx]
    return value


def configure_spec_from_cli() -> None:
    """Parse --spec flags once and configure the active preset."""

    argv = sys.argv
    preset = None
    idx = 1
    while idx < len(argv):
        token = argv[idx]
        if token == "--spec":
            preset = _pop_arg(argv, idx)
            continue
        if token.startswith("--spec="):
            preset = _pop_inline_arg(argv, idx)
            continue
        idx += 1

    if preset is None:
        preset = DEFAULT_SPEC_PRESET

    if preset not in AVAILABLE_SPEC_PRESETS:
        options = ", ".join(AVAILABLE_SPEC_PRESETS)
        raise ValueError(f"Unknown --spec preset '{preset}'. Choices: {options}")

    set_active_spec_preset(preset)
