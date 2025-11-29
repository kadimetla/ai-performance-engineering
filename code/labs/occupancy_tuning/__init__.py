"""Public helpers for the Triton Proton occupancy lab."""

from __future__ import annotations

import importlib

triton_matmul = importlib.import_module("core.profiling.occupancy_tuning.triton_matmul")
from core.profiling.occupancy_tuning.triton_matmul import (  # noqa: E402
    matmul_kernel,
    run_one,
    describe_schedule,
)

__all__ = [
    "matmul_kernel",
    "run_one",
    "describe_schedule",
    "triton_matmul",
]
