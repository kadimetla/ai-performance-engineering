"""Lightweight device helpers shared across chapters."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def get_preferred_device() -> Tuple[torch.device, Optional[str]]:
    """Return the best available device and an error message if CUDA is absent."""
    if torch.cuda.is_available():
        return torch.device("cuda"), None
    return torch.device("cpu"), "CUDA not available"


def cuda_supported() -> bool:
    """Convenience helper to check CUDA availability."""
    return torch.cuda.is_available()


__all__ = ["get_preferred_device", "cuda_supported"]
