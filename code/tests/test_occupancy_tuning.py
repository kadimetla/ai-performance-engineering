"""Unit tests for labs/occupancy_tuning helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.occupancy_tuning import sweep_schedules
from labs.occupancy_tuning.triton_matmul_schedules import SCHEDULES


def test_resolve_schedules_handles_unknown_names() -> None:
    known = SCHEDULES[0].name
    resolved = sweep_schedules.resolve_schedules([known])
    assert len(resolved) == 1 and resolved[0].name == known

    with pytest.raises(ValueError):
        sweep_schedules.resolve_schedules(["does_not_exist"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_run_sweep_executes_single_schedule() -> None:
    schedule = SCHEDULES[0]
    results = sweep_schedules.run_sweep(
        [schedule],
        size=64,
        iterations=1,
        warmup=0,
        dtype=torch.float16,
        use_compile=False,
    )
    assert len(results) == 1
    result = results[0]
    assert result.name == schedule.name
    assert result.mean_ms >= 0.0
    assert result.tflops >= 0.0
