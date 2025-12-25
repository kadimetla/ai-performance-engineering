"""Optimized disaggregated inference benchmark (multi-GPU torchrun pipeline).

Chapter 15: Disaggregated Inference (Optimized)

Optimizations:
- Overlap prefill and decode by pipelining KV transfers per request.
- Use non-blocking device transfers in the local verification path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ch15.baseline_disaggregated_inference_multigpu import (  # noqa: E402
    DisaggConfig,
    _DisaggregatedInferenceMultiGPUBenchmark,
    _run_torchrun_worker,
)
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, TorchrunLaunchSpec  # noqa: E402


class OptimizedDisaggregatedInferenceMultiGPUBenchmark(_DisaggregatedInferenceMultiGPUBenchmark):
    """Pipelined prefill/decode overlap across multi-GPU ranks."""

    def __init__(self) -> None:
        super().__init__(overlap=True, label="optimized_disaggregated_inference_multigpu")

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=[],
            env={
                "AISP_DISAGG_WORLD_SIZE": str(self.world_size),
                "NCCL_DEBUG": "WARN",
                "NCCL_P2P_LEVEL": "NVL",
                "NCCL_P2P_DISABLE": "0",
            },
            parse_rank0_only=True,
            multi_gpu_required=True,
            name="optimized_disaggregated_inference_multigpu",
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedDisaggregatedInferenceMultiGPUBenchmark()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _run_torchrun_worker(
        DisaggConfig(),
        overlap=True,
        label="optimized_disaggregated_inference_multigpu",
        iters=int(args.iters),
        warmup=int(args.warmup),
    )


if __name__ == "__main__":
    main()
