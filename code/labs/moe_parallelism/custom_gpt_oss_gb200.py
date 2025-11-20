"""Custom plan: GPT-OSS-120B MoE on 8 NVL72 GB200 racks (InfiniBand vs Ethernet)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.moe_parallelism.plan import (  # noqa: E402
    ClusterSpec,
    ModelSpec,
    ParallelismPlan,
)
from labs.moe_parallelism.benchmarking import PlanBenchmark, run_benchmark  # noqa: E402

MODEL_GPT_OSS_120B = ModelSpec(
    name="GPT-OSS-120B MoE",
    params_billion=120.0,
    hidden_size=10240,
    sequence_length=4096,
    layers=96,
    moe_layers=32,
    experts_total=256,
    router_topk=2,
    ffn_expansion=4,
    dtype_bytes=2,
)

FABRIC_OPTIONS: Dict[str, ClusterSpec] = {
    "ib": ClusterSpec(
        name="GB200 NVL72 (InfiniBand)",
        nodes=72,  # 8 racks * 9 NVSwitch nodes per rack
        gpus_per_node=8,
        nics_per_node=2,
        nic_bandwidth_gbps=800.0,  # dual 800G CX8s per NVL72 baseboard
        nvlink_bandwidth_tbps=1.8,  # GB200 NVLink aggregate per GPU
        hbm_gb=192.0,
    ),
    "ethernet": ClusterSpec(
        name="GB200 NVL72 (Ethernet)",
        nodes=72,
        gpus_per_node=8,
        nics_per_node=2,
        nic_bandwidth_gbps=400.0,  # dual 400G Ethernet spines per board
        nvlink_bandwidth_tbps=1.8,
        hbm_gb=192.0,
    ),
}

PLAN_GPT_OSS = ParallelismPlan(
    name="GPT-OSS-120B layout (DP9×PP8×TP2×EP4)",
    dp=9,
    pp=8,
    tp=2,
    ep=4,
    microbatch_sequences=48,
    microbatches=24,
    experts_per_gpu=4,
    capacity_factor=1.2,
    dense_checkpoint_fraction=0.6,
    moe_checkpoint_fraction=0.9,
    stage_layers=[12] * 8,
    cross_node_ep=False,
    notes=[
        "Each pipeline stage fits inside one NVSwitch island (8 GPUs) for NVLink-only TP/EP",
        "Nine DP replicas map cleanly onto 72 GB200 GPUs-per-island (8 racks of NVL72)",
        "24 micro-batches keep ≥2×PP chunks in flight so pipeline bubbles stay low",
    ],
)


def run_for_fabrics(fabrics: Iterable[str]) -> None:
    for fabric in fabrics:
        cluster = FABRIC_OPTIONS[fabric]
        print(f"\n=== {cluster.name} ({fabric}) ===")
        benchmark = PlanBenchmark(plan=PLAN_GPT_OSS, cluster=cluster, model=MODEL_GPT_OSS_120B)
        run_benchmark(benchmark)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fabric",
        choices=["ib", "ethernet", "both"],
        default="both",
        help="Which interconnect to analyze (cross-rack fabric)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fabrics = FABRIC_OPTIONS.keys() if args.fabric == "both" else [args.fabric]
    run_for_fabrics(fabrics)


if __name__ == "__main__":
    main()
