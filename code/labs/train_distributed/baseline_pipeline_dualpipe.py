"""Baseline DualPipe demo mirroring DeepSeek's published overlap idea."""

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
    parser = argparse.ArgumentParser(description="Baseline DualPipe schedule.")
    add_pipeline_args(parser)
    parser.add_argument(
        "--dual-window-default",
        type=int,
        default=STAGES,
        help="Fallback window if --dual-window is not supplied.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stage_count = args.n_stages
    dual_window = args.dual_window or args.dual_window_default

    config = PipelineConfig(
        schedule="dualpipe",
        n_stages=stage_count,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size or stage_count,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        learning_rate=args.learning_rate,
        non_blocking=False,
        dual_window=dual_window,
        seed=args.seed,
    )

    experiment = PipelineExperiment(config)
    telemetry_total = PipelineTelemetry(config.n_stages, schedule=config.schedule)
    total_loss = 0.0
    start = perf_counter()

    for step in range(args.steps):
        inputs = torch.randn(config.batch_size, config.input_dim)
        targets = torch.randn_like(inputs)
        loss, telemetry = experiment.run_batch(inputs, targets)
        telemetry_total.merge(telemetry)
        total_loss += loss

        if step % args.log_every == 0:
            print(
                f"[baseline-dualpipe] step {step + 1}/{args.steps} loss={loss:.4f} "
                f"dual_window={config.dual_window} micro_batch={config.micro_batch_size}"
            )

    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    avg_loss = total_loss / max(1, args.steps)
    elems = args.steps * config.batch_size * config.input_dim
    elems_per_sec = elems / elapsed if elapsed > 0 else 0.0

    print(
        f"[baseline-dualpipe] done in {elapsed:.2f}s | avg_loss={avg_loss:.4f} | "
        f"elements/s={elems_per_sec:,.0f}"
    )
    print(format_telemetry("baseline-dualpipe", telemetry_total))


if __name__ == "__main__":
    main()


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "pipeline_dualpipe.py",
        base_args=["--mode", "baseline"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:dualpipe_2stages",
        default_nproc_per_node=1,
        multi_gpu_required=True,
        name="baseline_pipeline_dualpipe_2stages",
    )
