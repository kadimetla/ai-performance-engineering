"""Baseline plan for DeepSeek-R1-678B on GB200 NVL72 (8 racks, IB)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.common.moe_parallelism_plan import ParallelismPlan, SPEC_PRESETS
from labs.moe_parallelism.benchmarking import PlanBenchmark, run_benchmark


CLUSTER, MODEL = SPEC_PRESETS["deepseek_r1_678b_gb200_ib"]


def build_plan() -> ParallelismPlan:
    return ParallelismPlan(
        name="Baseline DeepSeek-R1-678B (DP9×PP8×TP2×EP4)",
        dp=9,
        pp=8,
        tp=2,
        ep=4,
        microbatch_sequences=32,
        microbatches=24,
        experts_per_gpu=4,
        capacity_factor=1.2,
        dense_checkpoint_fraction=0.5,
        moe_checkpoint_fraction=0.9,
        stage_layers=[16] * 8,
        cross_node_ep=False,
        notes=[
            "Even 8-stage pipeline leaves ~23% bubble and high PP traffic",
            "Uses conservative micro-batch to respect 678B footprint",
        ],
    )


class BaselineDeepseekGb200Benchmark(PlanBenchmark):
    def __init__(self) -> None:
        super().__init__(plan=build_plan(), cluster=CLUSTER, model=MODEL)


def get_benchmark() -> PlanBenchmark:
    return BaselineDeepseekGb200Benchmark()


if __name__ == "__main__":
    run_benchmark(get_benchmark())
