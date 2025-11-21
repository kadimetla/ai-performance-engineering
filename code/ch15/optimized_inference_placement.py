"""Optimized inference placement policy honoring NVLink-local TP/EP."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch15.baseline_inference_placement import (  # noqa: E402
    _PlacementBenchmark,
    PlacementConfig,
)


class OptimizedInferencePlacementBenchmark(_PlacementBenchmark):
    """Heuristic-aligned placement: TP intra-node for prefill, TP=1 for decode, sticky sessions."""

    def __init__(self) -> None:
        cfg = PlacementConfig(
            prefill_tp_size=8,
            prefill_span_nodes=False,  # keep TP inside the NVLink island
            decode_tp_size=1,  # collapse TP for decode to kill all-reduce
            decode_span_nodes=False,
            decode_microbatch=4,
            remote_expert_fraction=0.05,  # expert pinning favors local shards
            router_sticky_decode=True,
            kv_transfer_policy="local_only",  # never walk KV across nodes mid-session
            notes="Prefill TP within node, decode TP=1, MoE local-first, KV stickiness.",
        )
        super().__init__(cfg, prefix="placement_optimized")


def get_benchmark():
    return OptimizedInferencePlacementBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    summary = bench.get_custom_metrics() or {}
    print("Optimized placement summary:", summary)
