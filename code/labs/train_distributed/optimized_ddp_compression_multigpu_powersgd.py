"""Optimized DDP training without compression on fast interconnect (multi-GPU)."""

from __future__ import annotations

from pathlib import Path

from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "ddp_compression.py",
        base_args=[
            "--compression",
            "powersgd",
            "--extra-grad-mb",
            "4096",
            "--batch-size",
            "2",
            "--bucket-cap-mb",
            "256",
            "--powersgd-rank",
            "1",
        ],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:ddp_compression_multigpu_powersgd",
        multi_gpu_required=True,
        default_iterations=20,
        name="optimized_ddp_compression_multigpu_powersgd",
    )
