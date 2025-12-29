"""DDP training loop with optional gradient compression hooks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50, help="Number of optimization steps.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-rank batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument(
        "--compression",
        choices=["none", "int8", "powersgd"],
        default="none",
        help="Gradient compression mode.",
    )
    parser.add_argument(
        "--powersgd-rank",
        type=int,
        default=1,
        help="PowerSGD low-rank approximation rank.",
    )
    parser.add_argument(
        "--extra-grad-mb",
        type=int,
        default=64,
        help="Extra gradient payload size (MB) to make communication dominant.",
    )
    return parser.parse_args()


def main():
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.algorithms.ddp_comm_hooks import quantization_hooks, powerSGD

    from labs.train_distributed.training_utils.utils import (
        build_dataloader,
        build_text_model,
        build_tokenizer,
        get_dataset,
    )

    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if not dist.is_initialized() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", device_id=local_rank)

    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise RuntimeError("SKIPPED: requires >=2 GPUs for gradient compression")

    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = rank == 0

    tokenizer = build_tokenizer()
    dataset = get_dataset()["train"]
    dataloader = build_dataloader(
        dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        distributed=dist.is_initialized() and dist.get_world_size() > 1,
        num_workers=2,
        prefetch_factor=2,
    )

    model = build_text_model()
    model.to(device)
    model.train()

    extra_param = None
    if args.extra_grad_mb > 0:
        elem_bytes = torch.tensor([], dtype=torch.bfloat16).element_size()
        numel = (args.extra_grad_mb * 1024 * 1024) // elem_bytes
        extra_param = torch.nn.Parameter(torch.zeros(numel, device=device, dtype=torch.bfloat16))
        model.register_parameter("extra_grad_payload", extra_param)

    ddp_model = DDP(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
    )

    if args.compression == "int8":
        ddp_model.register_comm_hook(None, quantization_hooks.quantization_pertensor_hook)
    elif args.compression == "powersgd":
        state = powerSGD.PowerSGDState(
            process_group=dist.group.WORLD,
            matrix_approximation_rank=args.powersgd_rank,
        )
        ddp_model.register_comm_hook(state, powerSGD.powerSGD_hook)

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
        if extra_param is not None:
            loss = loss + extra_param.sum() * 0.0
        loss.backward()
        optimizer.step()

        total_tokens += batch["input_ids"].numel()

        if is_main and step % 10 == 0:
            print(
                f"[ddp-compression:{args.compression}] step {step}/{num_steps} | loss={loss.item():.4f}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    if is_main:
        toks_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
        print(
            f"[ddp-compression:{args.compression}] finished {num_steps} steps in {elapsed:.1f}s "
            f"({toks_per_sec:,.0f} toks/s per rank)"
        )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
