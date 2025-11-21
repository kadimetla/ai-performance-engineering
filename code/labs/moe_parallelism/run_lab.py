"""Convenience runner for the MoE parallelism lab."""

from __future__ import annotations


import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Callable, List, Tuple

from labs.moe_parallelism.benchmarking import PlanBenchmark, run_benchmark

from labs.moe_parallelism.baseline_parallelism_breakdown import get_benchmark as baseline_parallelism
from labs.moe_parallelism.optimized_parallelism_breakdown import get_benchmark as optimized_parallelism
from labs.moe_parallelism.baseline_pipeline_schedule import get_benchmark as baseline_pipeline
from labs.moe_parallelism.optimized_pipeline_schedule import get_benchmark as optimized_pipeline
from labs.moe_parallelism.baseline_moe_grouping import get_benchmark as baseline_moe
from labs.moe_parallelism.optimized_moe_grouping import get_benchmark as optimized_moe
from labs.moe_parallelism.baseline_memory_budget import get_benchmark as baseline_memory
from labs.moe_parallelism.optimized_memory_budget import get_benchmark as optimized_memory
from labs.moe_parallelism.baseline_network_affinity import get_benchmark as baseline_network
from labs.moe_parallelism.optimized_network_affinity import get_benchmark as optimized_network
from labs.moe_parallelism.baseline_gpt_gb200 import get_benchmark as baseline_gpt_gb200
from labs.moe_parallelism.optimized_gpt_gb200 import get_benchmark as optimized_gpt_gb200
from labs.moe_parallelism.baseline_deepseek_gb200 import get_benchmark as baseline_deepseek_gb200
from labs.moe_parallelism.optimized_deepseek_gb200 import get_benchmark as optimized_deepseek_gb200
from labs.moe_parallelism.baseline_moe_vllm_env import get_benchmark as baseline_moe_env
from labs.moe_parallelism.optimized_moe_vllm_env import get_benchmark as optimized_moe_env


SCENARIOS: List[Tuple[str, Callable[[], PlanBenchmark]]] = [
    ("baseline_parallelism_breakdown", baseline_parallelism),
    ("optimized_parallelism_breakdown", optimized_parallelism),
    ("baseline_pipeline_schedule", baseline_pipeline),
    ("optimized_pipeline_schedule", optimized_pipeline),
    ("baseline_moe_grouping", baseline_moe),
    ("optimized_moe_grouping", optimized_moe),
    ("baseline_memory_budget", baseline_memory),
    ("optimized_memory_budget", optimized_memory),
    ("baseline_network_affinity", baseline_network),
    ("optimized_network_affinity", optimized_network),
    ("baseline_gpt_gb200", baseline_gpt_gb200),
    ("optimized_gpt_gb200", optimized_gpt_gb200),
    ("baseline_deepseek_gb200", baseline_deepseek_gb200),
    ("optimized_deepseek_gb200", optimized_deepseek_gb200),
    ("baseline_moe_vllm_env", baseline_moe_env),
    ("optimized_moe_vllm_env", optimized_moe_env),
]


def main() -> None:
    for name, factory in SCENARIOS:
        print(f"\n=== Running {name} ===")
        benchmark = factory()
        run_benchmark(benchmark)


if __name__ == "__main__":
    main()
