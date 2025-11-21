"""Decode kernel builders shared by the bucketed decode demos."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPILE_MODE = "reduce-overhead" if torch.cuda.is_available() else "default"


@dataclass
class DecodeKernel:
    """Light wrapper so callers can introspect the backend type."""

    fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
    backend: str

    def __call__(
        self, tokens: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.fn(tokens, kv, mask)


class VLLMDecodeKernel:
    """
    Small PagedAttention-backed decode step.

    Uses vLLM's fused paged attention kernel when available. Falls back to
    eager instantiation when CUDA or vLLM custom ops are missing.
    """

    def __init__(self, hidden: int, max_batch: int = 32, device: str = DEVICE) -> None:
        from vllm.attention.ops.paged_attn import PagedAttention  # type: ignore

        self.device = device
        self.hidden = hidden
        self.num_heads = 1
        self.head_size = hidden
        self.kv_cache_dtype = "auto"
        self.scale = 1.0 / math.sqrt(float(self.head_size))

        self.max_batch = max_batch
        # Keep one block per sequence; block size matches a single token.
        self.block_size = 1
        self.num_blocks = 1

        # Preallocate a tiny KV cache and the block tables/seq_lens used per call.
        kv_shape = PagedAttention.get_kv_cache_shape(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            num_kv_heads=self.num_heads,
            head_size=self.head_size,
            cache_dtype_str=self.kv_cache_dtype,
        )
        self.kv_cache = torch.randn(kv_shape, device=self.device, dtype=torch.float16)
        self.key_cache, self.value_cache = PagedAttention.split_kv_cache(
            self.kv_cache, num_kv_heads=self.num_heads, head_size=self.head_size
        )

        self.block_tables = torch.zeros(
            (self.max_batch, 1), dtype=torch.int32, device=self.device
        )
        self.seq_lens = torch.ones(
            self.max_batch, dtype=torch.int32, device=self.device
        )

        self._paged_attention = PagedAttention.forward_decode

    def ensure_capacity(self, batch: int) -> None:
        if batch <= self.max_batch:
            return
        # Resize block tables / seq_lens for a larger batch.
        new_bt = torch.zeros(
            (batch, self.block_tables.shape[1]),
            dtype=self.block_tables.dtype,
            device=self.block_tables.device,
        )
        new_bt[: self.block_tables.shape[0]] = self.block_tables
        self.block_tables = new_bt

        new_seq = torch.ones(batch, dtype=self.seq_lens.dtype, device=self.seq_lens.device)
        new_seq[: self.seq_lens.shape[0]] = self.seq_lens
        self.seq_lens = new_seq
        self.max_batch = batch

    def __call__(
        self, tokens: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch = tokens.size(0)
        self.ensure_capacity(batch)

        query = tokens.view(batch, self.num_heads, self.head_size)
        # Reuse preallocated block tables / seq_lens; they are trivial for the toy decode.
        out = self._paged_attention(
            query,
            self.key_cache,
            self.value_cache,
            self.block_tables[:batch],
            self.seq_lens[:batch],
            self.block_size,
            self.kv_cache_dtype,
            self.num_heads,
            self.scale,
            None,  # alibi_slopes
            None,  # k_scale
            None,  # v_scale
        )

        flat = out.view(batch, self.hidden)
        if mask is not None:
            flat = flat.masked_fill(~mask[:, None], float("-inf"))
        return flat

    @property
    def bytes(self) -> int:
        return self.kv_cache.numel() * self.kv_cache.element_size()


def _torch_decode(hidden: int) -> Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    def _decode(tokens: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_scores = torch.tanh(tokens + kv)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask[:, None], float("-inf"))
        return attn_scores

    try:
        return torch.compile(  # type: ignore[arg-type]
            _decode, mode=COMPILE_MODE, fullgraph=False, dynamic=False
        )
    except Exception:
        return _decode


def build_decode_kernel(
    hidden: int,
    *,
    max_batch: int = 32,
    prefer_vllm: bool = True,
    device: str = DEVICE,
) -> DecodeKernel:
    """
    Try to build a vLLM-backed decode kernel; fall back to torch.compile/eager.
    """
    if prefer_vllm:
        try:
            kernel = VLLMDecodeKernel(hidden=hidden, max_batch=max_batch, device=device)
            return DecodeKernel(fn=kernel, backend="vllm")
        except Exception:
            pass

    torch_kernel = _torch_decode(hidden)
    return DecodeKernel(fn=torch_kernel, backend="torch")
