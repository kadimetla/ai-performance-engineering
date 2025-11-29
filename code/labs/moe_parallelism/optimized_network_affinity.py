"""Optimized network affinity: localized EP + hierarchical DP overlap."""

from __future__ import annotations


import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.common.moe_parallelism_plan import ParallelismPlan
from labs.moe_parallelism.benchmarking import PlanBenchmark, run_benchmark


def build_plan() -> ParallelismPlan:
    return ParallelismPlan(
        name="Optimized network affinity (hierarchical collectives)",
        dp=4,
        pp=4,
        tp=2,
        ep=4,
        microbatch_sequences=32,
        microbatches=18,
        experts_per_gpu=4,
        capacity_factor=1.25,
        dense_checkpoint_fraction=0.5,
        moe_checkpoint_fraction=0.95,
        stage_layers=[24, 24, 24, 24],
        cross_node_ep=False,
        notes=[
            "DP=4 leaves each replica on four nodes so NCCL trees stay shallow",
            "Stage neighbors share InfiniBand pairs -> easy NIC pinning for pipeline sends",
            "EP dispatch + TP all-reduce never leave NVSwitch so HDR100 handles only DP/PP traffic",
        ],
    )


class OptimizedNetworkAffinityBenchmark(PlanBenchmark):
    def __init__(self) -> None:
        super().__init__(build_plan())


def get_benchmark() -> PlanBenchmark:
    return OptimizedNetworkAffinityBenchmark()


if __name__ == "__main__":
    run_benchmark(get_benchmark())
