"""Compare baseline/optimized plan metrics for key pairs."""

from __future__ import annotations

import importlib
from typing import List, Tuple

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.moe_parallelism.plan import PlanEvaluator, format_report

PAIRS: List[Tuple[str, str]] = [
    ("labs.moe_parallelism.baseline_gpt_gb200", "labs.moe_parallelism.optimized_gpt_gb200"),
    ("labs.moe_parallelism.baseline_deepseek_gb200", "labs.moe_parallelism.optimized_deepseek_gb200"),
]

METRIC_FIELDS = [
    ("step_ms", "estimated_step_ms"),
    ("throughput_tokens_s", "throughput_tokens_per_s"),
    ("bubble", "bubble_fraction"),
    ("params_gb", "params_gb"),
    ("optimizer_gb", "optimizer_gb"),
    ("grads_gb", "grad_gb"),
    ("activations_gb", "activation_gb"),
    ("margin_gb", "memory_margin_gb"),
]


def _load(plan_module: str):
    mod = importlib.import_module(plan_module)
    plan = mod.build_plan()  # type: ignore[attr-defined]
    cluster = mod.CLUSTER  # type: ignore[attr-defined]
    model = mod.MODEL  # type: ignore[attr-defined]
    evaluator = PlanEvaluator(cluster, model)
    return evaluator.analyze(plan)


def _fmt(val: float, key: str) -> str:
    if key in {"step_ms", "throughput_tokens_s"}:
        return f"{val:,.0f}"
    if "gb" in key:
        return f"{val:,.1f}"
    if key == "bubble":
        return f"{val*100:.1f}%"
    return f"{val:.3f}"


def compare_pair(baseline: str, optimized: str) -> None:
    base_report = _load(baseline)
    opt_report = _load(optimized)
    name = optimized.split(".")[-1].replace("optimized_", "")
    print(f"\n=== {name} ===")
    print(format_report(base_report))
    print("---")
    print(format_report(opt_report))
    print("\nMetric comparison (baseline -> optimized):")
    rows = []
    for label, attr in METRIC_FIELDS:
        bval = getattr(base_report, attr)
        oval = getattr(opt_report, attr)
        rows.append((label, bval, oval, (oval / bval) if bval else 0.0))
    header = f"{'metric':<18}{'baseline':>15}{'optimized':>15}{'opt/base':>12}"
    print(header)
    print("-" * len(header))
    for label, bval, oval, ratio in rows:
        print(f"{label:<18}{_fmt(bval, label):>15}{_fmt(oval, label):>15}{ratio:>12.2f}")


def main() -> None:
    for baseline, optimized in PAIRS:
        compare_pair(baseline, optimized)


if __name__ == "__main__":
    main()
