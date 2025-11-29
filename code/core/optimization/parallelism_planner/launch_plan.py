"""Minimal launch plan generator for dry-run torchrun commands."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class LaunchPlan:
    model_params: int
    nodes: int
    gpus_per_node: int
    tp: int
    pp: int
    dp: int
    batch_size: int
    command: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def generate_launch_plan(
    model_params: int,
    nodes: int,
    gpus_per_node: int,
    tp: int,
    pp: int,
    dp: int,
    batch_size: int = 1,
    script: str = "train.py",
    extra_args: Optional[str] = None,
) -> LaunchPlan:
    world_size = nodes * gpus_per_node
    if tp * pp * dp > world_size:
        raise ValueError(f"TP*PP*DP ({tp*pp*dp}) exceeds total GPUs ({world_size})")
    torchrun_cmd = (
        f"torchrun --nnodes {nodes} --nproc_per_node {gpus_per_node} "
        f"--rdzv_backend c10d --rdzv_endpoint localhost:29400 "
        f"{script} --tp {tp} --pp {pp} --dp {dp} --global-batch {batch_size}"
    )
    if extra_args:
        torchrun_cmd += f" {extra_args}"
    return LaunchPlan(
        model_params=model_params,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        tp=tp,
        pp=pp,
        dp=dp,
        batch_size=batch_size,
        command=torchrun_cmd,
    )
