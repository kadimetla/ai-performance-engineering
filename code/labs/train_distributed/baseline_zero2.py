"""Baseline ZeRO-2: shard optimizer state and gradients."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam, Optimizer

from labs.train_distributed.training_utils.memory import print_memory_stats
from labs.train_distributed.training_utils.utils import get
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


class GradientSharder:
    """ZeRO-2 style gradient+optimizer sharding."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.params = [p for group in optimizer.param_groups for p in group["params"]]
        self._shard_parameters()
        self.communication_time = 0.0
        self.step_time = 0.0

    def _shard_parameters(self):
        world_size = get("ws")
        rank = get("rank")
        shard_size = len(self.params) // world_size
        remainder = len(self.params) % world_size

        start = rank * shard_size + min(rank, remainder)
        end = start + shard_size + (1 if rank < remainder else 0)
        self.local_indices = list(range(start, end))
        self.local_params = {self.params[i] for i in self.local_indices}

        for group in self.optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p in self.local_params]

    def step(self, closure=None):
        step_start = time.perf_counter()
        comm_start = step_start

        for idx, param in enumerate(self.params):
            grad = param.grad
            if grad is None:
                continue

            flattened = grad.data.contiguous().view(-1)
            in_tensor = torch.cat([flattened for _ in range(get("ws"))], dim=0)
            shard_grad = torch.empty_like(flattened)
            dist.reduce_scatter_tensor(shard_grad, in_tensor, op=dist.ReduceOp.SUM)

            if idx in self.local_indices:
                param.grad = (shard_grad / get("ws")).view_as(grad.data)
            else:
                param.grad = None

        torch.cuda.synchronize()
        self.communication_time += time.perf_counter() - comm_start

        self.optimizer.step(closure)

        shard_size = len(self.params) // get("ws")
        remainder = len(self.params) % get("ws")
        for idx, param in enumerate(self.params):
            if idx < (shard_size + 1) * remainder:
                owner = idx // (shard_size + 1)
            else:
                owner = (idx - remainder) // shard_size
            dist.broadcast(param.data, src=owner)

        torch.cuda.synchronize()
        self.step_time += time.perf_counter() - step_start

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)


def _build_model(hidden_size: int, device):
    layers = []
    for _ in range(6):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
    layers.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*layers).to(device)


def train(model, optimizer, batch_size, device, steps, label):
    rank = get("rank")
    input_dim = model[0].in_features

    optimizer.zero_grad()
    warm_x = torch.randn(batch_size, input_dim, device=device)
    warm_y = torch.randn_like(warm_x)
    nn.functional.mse_loss(model(warm_x), warm_y).backward()
    optimizer.step()
    torch.cuda.synchronize()

    if rank == 0:
        print_memory_stats(f"{label} warmup", model, optimizer, rank, device)
    dist.barrier()

    peaks = []
    for step in range(steps):
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad()

        x = torch.randn(batch_size, input_dim, device=device)
        y = torch.randn_like(x)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        peaks.append(peak)
        if rank == 0 and step == 0:
            print(f"[{label}] peak memory first step: {peak:.2f} MB")
        dist.barrier()

    if rank == 0:
        print(f"[{label}] max peak memory over {steps} steps: {max(peaks):.2f} MB")
    return max(peaks)


def main():
    args = parse_args()
    local_rank = get("lrank")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank)
    if get("ws") < 2:
        print("Warning: baseline ZeRO-2 demo is running on a single GPU; sharding benefits require world_size>=2.")
    rank = get("rank")
    device = torch.device(f"cuda:{local_rank}")

    baseline_model = _build_model(args.hidden_size, device)
    baseline_opt = Adam(baseline_model.parameters(), lr=1e-3)
    mem_baseline = train(baseline_model, baseline_opt, args.batch_size, device, args.steps, "baseline-adam")

    zero2_model = _build_model(args.hidden_size, device)
    zero2_opt = GradientSharder(Adam(zero2_model.parameters(), lr=1e-3))
    mem_zero2 = train(zero2_model, zero2_opt, args.batch_size, device, args.steps, "zero2")

    if rank == 0:
        saved = mem_baseline - mem_zero2
        pct = (saved / mem_baseline * 100) if mem_baseline > 0 else 0
        print(f"[summary] baseline peak: {mem_baseline:.2f} MB | zero2 peak: {mem_zero2:.2f} MB "
              f"({saved:.2f} MB saved, {pct:.1f}% reduction)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "zero2.py",
        base_args=["--mode", "baseline"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:zero2",
        default_nproc_per_node=None,
        name="baseline_zero2",
    )
