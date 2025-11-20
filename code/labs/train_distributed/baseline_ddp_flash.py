"""Baseline DDP training loop using flash attention (requires compatible kernels)."""

from __future__ import annotations

import argparse
from time import perf_counter

import torch
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import PartialState
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

from labs.train_distributed.training_utils.utils import (
    build_dataloader,
    build_text_model_flash,
    build_tokenizer,
    get_dataset,
    set_seed,
)
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50, help="Number of optimization steps.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-rank batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(1234)

    state = PartialState()
    device = state.device

    tokenizer = build_tokenizer()
    dataset = get_dataset()["train"]
    dataloader = build_dataloader(
        dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        distributed=True,
        num_workers=2,
        prefetch_factor=2,
    )

    model = build_text_model_flash()
    model.to(device)
    model.train()

    ddp_model = DDP(
        model,
        device_ids=[device],
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
    )

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.learning_rate)

    num_steps = min(args.steps, len(dataloader))
    start = perf_counter()
    total_tokens = 0

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        optimizer.zero_grad(set_to_none=True)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        batch["labels"] = batch["input_ids"].clone()
        outputs = ddp_model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_tokens += batch["input_ids"].numel()

        if state.is_main_process and step % 10 == 0:
            print(f"[baseline-ddp-flash] step {step}/{num_steps} | loss={loss.item():.4f}")

    torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    if state.is_main_process:
        toks_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
        print(f"[baseline-ddp-flash] finished {num_steps} steps in {elapsed:.1f}s "
              f"({toks_per_sec:,.0f} toks/s per rank)")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "ddp.py",
        base_args=["--mode", "baseline_flash"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:ddp_flash",
        default_nproc_per_node=None,
        name="baseline_ddp_flash",
    )
