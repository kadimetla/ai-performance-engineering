"""Structured sparsity helpers for 2:4 pruning patterns."""

from __future__ import annotations

import torch


def prune_2_4(weight: torch.Tensor) -> torch.Tensor:
    """Apply 2:4 structured sparsity along the last dimension.

    Keeps the two largest-magnitude elements in each contiguous group of 4.
    """
    if weight.dim() != 2:
        raise ValueError("prune_2_4 expects a 2D weight matrix")
    if weight.shape[1] % 4 != 0:
        raise ValueError("Last dimension must be divisible by 4 for 2:4 sparsity")

    grouped = weight.view(weight.shape[0], weight.shape[1] // 4, 4)
    scores = grouped.abs()
    topk = scores.topk(2, dim=-1).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, topk, True)
    pruned = torch.where(mask, grouped, torch.zeros_like(grouped))
    return pruned.view_as(weight)
