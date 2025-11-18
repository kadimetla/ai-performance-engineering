"""Shared configuration helpers for the paged attention benchmarks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

ENV_PREFIX = "PAGED_ATTN_"


def _env_int(name: str, default: int) -> int:
    """Fetch integer override from environment."""
    raw = os.getenv(f"{ENV_PREFIX}{name.upper()}")
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_bool(name: str, default: bool) -> bool:
    """Fetch boolean override from environment."""
    raw = os.getenv(f"{ENV_PREFIX}{name.upper()}")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_dtype(default: torch.dtype) -> torch.dtype:
    """Read dtype override."""
    raw = os.getenv(f"{ENV_PREFIX}DTYPE")
    if raw is None:
        return default
    normalized = raw.strip().lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(normalized, default)


@dataclass
class PagedAttnConfig:
    """Runtime configuration for paged attention benchmarks."""

    batch_size: int = 4
    context_tokens: int = 4096
    decode_tokens: int = 128
    hidden_dim: int = 2048
    num_heads: int = 16
    block_size: int = 128
    chunk_size: int = 2048
    experts: int = 32
    top_k: int = 2
    moe_hidden_dim: int = 4096
    router_hidden_dim: int = 2048
    capture_prefill: bool = True
    dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        self.head_dim = self.hidden_dim // self.num_heads
        self.context_tokens = max(self.context_tokens, self.block_size)
        self.decode_tokens = max(self.decode_tokens, 1)
        self.block_size = max(self.block_size, 16)
        self.chunk_size = max(self.chunk_size, self.block_size)


def resolve_paged_attn_config() -> PagedAttnConfig:
    """Create config populated from environment overrides."""
    config = PagedAttnConfig(
        batch_size=_env_int("batch", 4),
        context_tokens=_env_int("context", 4096),
        decode_tokens=_env_int("decode", 128),
        hidden_dim=_env_int("hidden", 2048),
        num_heads=_env_int("heads", 16),
        block_size=_env_int("block_size", 128),
        chunk_size=_env_int("chunk", 2048),
        experts=_env_int("experts", 32),
        top_k=_env_int("top_k", 2),
        moe_hidden_dim=_env_int("moe_hidden", 4096),
        router_hidden_dim=_env_int("router_hidden", 2048),
        capture_prefill=_env_bool("capture_prefill", True),
        dtype=_env_dtype(torch.float16),
    )
    return config


def resolve_device() -> torch.device:
    """Return CUDA device, raising if unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError("These paged attention demos require CUDA-capable hardware.")
    return torch.device("cuda")


def seed_everything(seed: int = 42) -> None:
    """Deterministic seeding for repeatable benchmarks."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
