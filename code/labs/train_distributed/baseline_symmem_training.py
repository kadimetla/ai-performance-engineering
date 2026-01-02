"""Baseline symmetric memory training demo (fallback path, symmetric disabled)."""

from __future__ import annotations

from pathlib import Path

from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parents[2] / "ch04" / "symmetric_memory_training_advanced.py",
        base_args=[
            "--demo",
            "optimizer",
            "--disable-symmetric",
            "--allow-single-gpu",
            "--steps",
            "100",
            "--batch-size",
            "1",
            "--hidden-dim",
            "768",
            "--output-dim",
            "384",
            "--optimizer-layers",
            "64",
            "--sync-interval",
            "1",
        ],
        target_label="labs/train_distributed:symmem_training",
        multi_gpu_required=False,
        default_nproc_per_node=1,
        name="baseline_symmem_training",
    )
