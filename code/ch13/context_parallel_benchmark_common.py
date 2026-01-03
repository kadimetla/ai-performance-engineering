"""Shared helpers for context-parallel attention benchmarks (multi-GPU)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass(frozen=True)
class ContextParallelConfig:
    batch_size: int = 1
    seq_len: int = 8192  # total sequence length across ranks
    hidden_size: int = 1024
    num_heads: int = 8
    num_layers: int = 2
    dtype: torch.dtype = torch.bfloat16
    causal: bool = True


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[name]


def align_seq_len(seq_len: int, world_size: int) -> int:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if seq_len % world_size == 0:
        return seq_len
    return world_size * ((seq_len + world_size - 1) // world_size)


def init_distributed() -> Tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Context-parallel benchmarks require torchrun (RANK/WORLD_SIZE missing).")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


class ContextParallelLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dtype: torch.dtype) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False, dtype=dtype)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)

    def split_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        return q, k, v

    def merge_heads(self, attn: torch.Tensor) -> torch.Tensor:
        return attn.transpose(1, 2).contiguous().view(attn.shape[0], attn.shape[2], self.hidden_size)


def _apply_causal_mask(
    scores: torch.Tensor,
    *,
    rank: int,
    seq_shard: int,
    world_size: int,
) -> torch.Tensor:
    seq_total = seq_shard * world_size
    global_q = (rank * seq_shard) + torch.arange(seq_shard, device=scores.device)
    global_k = torch.arange(seq_total, device=scores.device)
    return scores.masked_fill(
        global_k.view(1, 1, 1, seq_total) > global_q.view(1, 1, seq_shard, 1),
        float("-inf"),
    )


def all_gather_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    process_group: Optional[dist.ProcessGroup],
    causal: bool,
    seq_shard: int,
    scale: float,
) -> torch.Tensor:
    if world_size > 1 and dist.is_initialized():
        gather_k = [torch.empty_like(k) for _ in range(world_size)]
        gather_v = [torch.empty_like(v) for _ in range(world_size)]
        dist.all_gather(gather_k, k, group=process_group)
        dist.all_gather(gather_v, v, group=process_group)
        k_full = torch.cat(gather_k, dim=2)
        v_full = torch.cat(gather_v, dim=2)
    else:
        k_full = k
        v_full = v

    scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
    if causal:
        scores = _apply_causal_mask(scores, rank=rank, seq_shard=seq_shard, world_size=world_size)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v_full)


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    process_group: Optional[dist.ProcessGroup],
    causal: bool,
    seq_shard: int,
    scale: float,
) -> torch.Tensor:
    if world_size <= 1 or not dist.is_initialized():
        return all_gather_attention(
            q,
            k,
            v,
            rank=rank,
            world_size=1,
            process_group=None,
            causal=causal,
            seq_shard=seq_shard,
            scale=scale,
        )

    k_current = k
    v_current = v
    attn_num: Optional[torch.Tensor] = None
    global_max: Optional[torch.Tensor] = None
    global_sum: Optional[torch.Tensor] = None

    global_q = (rank * seq_shard) + torch.arange(seq_shard, device=q.device)
    global_q = global_q.view(1, 1, seq_shard, 1)
    k_indices = torch.arange(seq_shard, device=q.device).view(1, 1, 1, seq_shard)

    for step in range(world_size):
        target_rank = (rank - step) % world_size
        scores = torch.matmul(q, k_current.transpose(-2, -1)) * scale

        if causal:
            global_k = target_rank * seq_shard + k_indices
            scores = scores.masked_fill(global_k > global_q, float("-inf"))

        local_max = scores.amax(dim=-1, keepdim=True)
        exp_scores = torch.exp(scores - local_max)
        local_sum = exp_scores.sum(dim=-1, keepdim=True)
        local_num = torch.matmul(exp_scores, v_current)

        if global_max is None:
            global_max = local_max
            global_sum = local_sum
            attn_num = local_num
        else:
            new_max = torch.maximum(global_max, local_max)
            scale_prev = torch.exp(global_max - new_max)
            scale_local = torch.exp(local_max - new_max)
            attn_num = attn_num * scale_prev + local_num * scale_local  # type: ignore[assignment]
            global_sum = global_sum * scale_prev + local_sum * scale_local
            global_max = new_max

        if step < world_size - 1:
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1) % world_size

            k_recv = torch.empty_like(k_current)
            v_recv = torch.empty_like(v_current)

            ops = [
                dist.P2POp(dist.isend, k_current, next_rank, group=process_group),
                dist.P2POp(dist.irecv, k_recv, prev_rank, group=process_group),
                dist.P2POp(dist.isend, v_current, next_rank, group=process_group),
                dist.P2POp(dist.irecv, v_recv, prev_rank, group=process_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

            k_current = k_recv
            v_current = v_recv

    if attn_num is None or global_sum is None:
        raise RuntimeError("Ring attention accumulation failed")
    return attn_num / (global_sum + 1e-8)


def build_layers(config: ContextParallelConfig, device: torch.device) -> nn.ModuleList:
    return nn.ModuleList(
        [ContextParallelLayer(config.hidden_size, config.num_heads, config.dtype) for _ in range(config.num_layers)]
    ).to(device=device)


def run_context_parallel(
    *,
    config: ContextParallelConfig,
    iters: int,
    warmup: int,
    attention_fn: Callable[..., torch.Tensor],
) -> None:
    rank, world_size, local_rank = init_distributed()
    seq_len = align_seq_len(config.seq_len, world_size)
    seq_shard = seq_len // world_size

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(f"cuda:{local_rank}")
    layers = build_layers(config, device)
    inputs = torch.randn(
        config.batch_size,
        seq_shard,
        config.hidden_size,
        device=device,
        dtype=config.dtype,
    )

    def _step() -> torch.Tensor:
        x = inputs
        for layer in layers:
            q, k, v = layer.split_qkv(x)
            attn_out = attention_fn(
                q,
                k,
                v,
                rank=rank,
                world_size=world_size,
                process_group=dist.group.WORLD,
                causal=config.causal,
                seq_shard=seq_shard,
                scale=layer.scale,
            )
            x = layer.proj(layer.merge_heads(attn_out))
        return x

    for _ in range(max(warmup, 0)):
        _step()
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(max(iters, 1)):
        _step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    if rank == 0:
        tokens_per_iter = config.batch_size * seq_len
        tokens_per_s = tokens_per_iter * (max(iters, 1) / max(elapsed, 1e-9))
        print(f"rank0 tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 time_per_iter_ms: {(elapsed / max(iters, 1)) * 1000.0:.3f}")

    dist.barrier()
    dist.destroy_process_group()
