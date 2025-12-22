"""Utilities for block-sparse attention masks (FlashInfer + dense baseline)."""

from __future__ import annotations

from typing import Tuple

import torch


def build_block_sparse_pattern(
    *,
    seq_len: int,
    block_size: int,
    window_blocks: int,
) -> torch.Tensor:
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size")
    blocks = seq_len // block_size
    mask = torch.zeros((blocks, blocks), dtype=torch.bool)
    for row in range(blocks):
        start = max(0, row - window_blocks)
        end = min(blocks, row + window_blocks + 1)
        mask[row, start:end] = True
    return mask


def build_dense_attention_mask(
    block_mask: torch.Tensor,
    *,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    values = torch.where(block_mask.to(device), zero, neg_inf)
    return values.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)


def build_bsr_from_block_mask(
    block_mask: torch.Tensor,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    blocks = block_mask.shape[0]
    indices_list = []
    indptr_list = [0]
    for row in range(blocks):
        cols = torch.nonzero(block_mask[row], as_tuple=False).flatten().tolist()
        indices_list.extend(cols)
        indptr_list.append(len(indices_list))
    indptr = torch.tensor(indptr_list, dtype=torch.int32, device=device)
    indices = torch.tensor(indices_list, dtype=torch.int32, device=device)
    total_blocks = float(blocks * blocks)
    allowed_blocks = float(block_mask.sum().item())
    sparsity_ratio = 1.0 - (allowed_blocks / total_blocks)
    return indptr, indices, sparsity_ratio
