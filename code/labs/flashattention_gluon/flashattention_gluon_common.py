"""Shared helpers for FlashAttention Gluon lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class FlashAttentionInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


@dataclass
class FlashAttentionKernel:
    fn: Callable
    provider: str


def build_flashattention_inputs(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> FlashAttentionInputs:
    """Create random Q/K/V batches."""
    shape = (batch, seq_len, heads, head_dim)
    generator = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    k = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    v = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return FlashAttentionInputs(q=q, k=k, v=v)


def resolve_gluon_flash_attention() -> FlashAttentionKernel:
    """
    Resolve a Gluon/Triton warp-specialized FlashAttention kernel.

    This lab requires Gluon; fail fast if not available.
    """
    try:  # pragma: no cover - environment dependent
        from gluon.ops.flash_attention import flash_attention as gluon_flash_attention
    except Exception as exc:
        raise RuntimeError(
            "Gluon flash_attention is required for the optimized lab. "
            "Run setup.sh to install Gluon and retry."
        ) from exc

    def _gluon_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return gluon_flash_attention(q, k, v, causal=False, dropout_p=0.0, softmax_scale=None)

    return FlashAttentionKernel(fn=_gluon_op, provider="gluon")
