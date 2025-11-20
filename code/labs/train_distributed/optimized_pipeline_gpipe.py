"""Optimized GPipe pipeline demo with aggressive micro-batching."""

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

GP_STAGES = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized GPipe pipeline.")
    add_pipeline_args(parser)
    parser.add_argument(
        "--micro-batch-overlap",
        type=int,
        default=GP_STAGES,
        help="Default microbatch fan-out used when --micro-batch-size is not provided.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_micro = args.micro_batch_size
    if target_micro is None:
        target_micro = max(4, args.batch_size // max(1, args.micro_batch_overlap))

    config = PipelineConfig(
        schedule="gpipe",
        n_stages=args.n_stages or GP_STAGES,
        batch_size=args.batch_size,
        micro_batch_size=target_micro,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        learning_rate=args.learning_rate * 2,
        non_blocking=True,
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
                f"[optimized-gpipe] step {step + 1}/{args.steps} loss={loss:.4f} "
                f"micro_batch={config.micro_batch_size}"
            )

    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    elems = args.steps * config.batch_size * config.input_dim
    elems_per_sec = elems / elapsed if elapsed > 0 else 0.0
    avg_loss = total_loss / max(1, args.steps)

    print(
        f"[optimized-gpipe] done in {elapsed:.2f}s | avg_loss={avg_loss:.4f} | "
        f"elements/s={elems_per_sec:,.0f}"
    )
    print(format_telemetry("optimized-gpipe", cumulative))


if __name__ == "__main__":
    main()


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "pipeline_gpipe.py",
        base_args=["--mode", "optimized"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:pipeline_gpipe_2stages",
        default_nproc_per_node=1,
        multi_gpu_required=True,
        name="optimized_pipeline_gpipe_2stages",
    )
