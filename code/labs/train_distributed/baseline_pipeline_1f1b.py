"""Baseline 1F1B demo with limited overlap to highlight idle bubbles."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from labs.train_distributed.pipeline import (
    PipelineConfig,
    PipelineExperiment,
    PipelineTelemetry,
    add_pipeline_args,
    format_telemetry,
)
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline 1F1B pipeline.")
    add_pipeline_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    stage_count = args.n_stages
    default_micro = args.micro_batch_size
    if default_micro is None:
        default_micro = max(args.batch_size // 2, stage_count)

    config = PipelineConfig(
        schedule="1f1b",
        n_stages=stage_count,
        batch_size=args.batch_size,
        micro_batch_size=default_micro,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        learning_rate=args.learning_rate,
        non_blocking=False,
        seed=args.seed,
    )

    experiment = PipelineExperiment(config)
    cumulative = PipelineTelemetry(config.n_stages, schedule=config.schedule)
    total_loss = 0.0
    start = perf_counter()

    for step in range(args.steps):
        inputs = torch.randn(config.batch_size, config.input_dim)
        targets = torch.randn_like(inputs)
        loss, telemetry = experiment.run_batch(inputs, targets)
        cumulative.merge(telemetry)
        total_loss += loss

        if step % args.log_every == 0:
            print(
                f"[baseline-1f1b] step {step + 1}/{args.steps} loss={loss:.4f} "
                f"micro_batch={config.micro_batch_size}"
            )

    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    elems = args.steps * config.batch_size * config.input_dim
    elems_per_sec = elems / elapsed if elapsed > 0 else 0.0
    avg_loss = total_loss / max(1, args.steps)

    print(
        f"[baseline-1f1b] done in {elapsed:.2f}s | avg_loss={avg_loss:.4f} | "
        f"elements/s={elems_per_sec:,.0f}"
    )
    print(format_telemetry("baseline-1f1b", cumulative))


if __name__ == "__main__":
    main()


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "pipeline_1f1b.py",
        base_args=["--mode", "baseline"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:1f1b_2stages",
        default_nproc_per_node=1,
        multi_gpu_required=True,
        name="baseline_pipeline_1f1b_2stages",
    )
