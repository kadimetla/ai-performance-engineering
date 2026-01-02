"""Baseline DDP training with PowerSGD gradient compression (multi-GPU)."""

from __future__ import annotations

from pathlib import Path

from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "ddp_compression.py",
        base_args=[
            "--compression",
            "none",
            "--extra-grad-mb",
            "2048",
            "--batch-size",
            "2",
            "--bucket-cap-mb",
            "256",
            "--naive-allreduce",
            "--naive-allreduce-sync",
        ],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:ddp_compression_multigpu_powersgd",
        multi_gpu_required=True,
        default_iterations=20,
        name="baseline_ddp_compression_multigpu_powersgd",
    )
