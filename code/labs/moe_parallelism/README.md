# Lab - MoE Parallelism Planner

## Summary
Provides scenario planning for mixture-of-experts clusters: memory budgeting, network affinity, parallelism breakdown, and pipeline schedules expressed as baseline/optimized harness targets.

## Learning Goals
- Quantify memory budgets for experts, routers, and KV caches before deploying models.
- Explore different grouping strategies (hashing, topology-aware) and their throughput impact.
- Model network affinity to decide where experts should live in an NVLink/NVSwitch fabric.
- Simulate pipeline schedules to identify bottlenecks before touching production systems.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_memory_budget.py`, `optimized_memory_budget.py` | Memory planners that prove how optimized layouts free additional HBM for experts. |
| `baseline_moe_grouping.py`, `optimized_moe_grouping.py`, `plan.py` | Grouping strategies spanning naive, locality-aware, and latency-balanced heuristics. |
| `baseline_network_affinity.py`, `optimized_network_affinity.py` | Network-affinity calculators comparing NVLink, NVSwitch, and PCIe hops. |
| `baseline_parallelism_breakdown.py`, `optimized_parallelism_breakdown.py`, `baseline_pipeline_schedule.py`, `optimized_pipeline_schedule.py` | Parallelism and scheduling studies for sharding experts across GPUs. |
| `benchmarking.py`, `run_lab.py`, `__init__.py` | Lab driver, Typer CLI, and exports used by the harness. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python tools/cli/benchmark_cli.py list-targets --chapter labs/moe_parallelism
python tools/cli/benchmark_cli.py run --targets labs/moe_parallelism --profile minimal
```
- Targets follow the `labs/moe_parallelism:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/moe_parallelism:<workload>="--flag value"` to sweep schedule knobs.

### Switching model / cluster specs

All baseline/optimized pairs use a shared preset system. Pass `--spec <name>` to any script (or via
`--target-extra-arg labs/moe_parallelism:<target>="--spec dgx_a100_175b"` when using the harness).

```bash
# GPT-OSS-120B on GB200 NVL72 racks w/ dual 800G InfiniBand (default)
python labs/moe_parallelism/run_lab.py              # implicit gpt_oss_120b_gb200_ib

# Same model but dual 400G Ethernet between racks
python labs/moe_parallelism/run_lab.py --spec gpt_oss_120b_gb200_ethernet

# Original interview scenario (175B on 16×8 DGX A100)
python labs/moe_parallelism/run_lab.py --spec dgx_a100_175b
```

`labs/moe_parallelism/spec_helper.py` removes the flag from `sys.argv` and calls into the planner before anything else
initializes, so every scenario automatically shares the same `ClusterSpec`/`ModelSpec`.

## Validation Checklist
- `python tools/cli/benchmark_cli.py run --targets labs/moe_parallelism --profile minimal` runs every planner pair and drops JSON/Markdown summaries.
- `python labs/moe_parallelism/run_lab.py --scenario grouped` prints an actionable plan (experts/GPU, bandwidth needs) for the chosen scenario.
- `python labs/moe_parallelism/optimized_memory_budget.py --validate` ensures optimized allocations meet the same correctness checks as the baseline.

## Notes
- `plan.py` centralizes scenario definitions so you only update one file when adding a new MoE topology.
- `benchmarking.py` can emit Markdown tables for documentation by passing `--format markdown`.
- `custom_gpt_oss_gb200.py` shows how to swap in a GPT‑OSS‑120B model spec and compare an 8×NVL72 GB200 deployment over InfiniBand (`--fabric ib`) versus Ethernet (`--fabric ethernet`). Use it as a template for other clusters.
