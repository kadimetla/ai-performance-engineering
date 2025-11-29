#!/usr/bin/env python3

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path

"""Synthetic DeepSeek workload referenced in Chapter 13 profiling walkthrough."""

import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from core.common.device_utils import get_preferred_device

try:
    _compiler_api = torch.compiler  # type: ignore[attr-defined]
except AttributeError:
    _compiler_api = None

if _compiler_api is not None and hasattr(_compiler_api, "nested_compile_region"):
    @_compiler_api.nested_compile_region  # type: ignore[misc]
    def _run_expert(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return block(x)
else:
    def _run_expert(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return block(x)


class ExpertMLP(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DeepSeekToy(nn.Module):
    def __init__(self, hidden: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(ExpertMLP(hidden) for _ in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + _run_expert(layer, x)
        return x


def main() -> None:
    device, cuda_err = get_preferred_device()
    if cuda_err:
        print(f"WARNING: CUDA unavailable ({cuda_err}); using CPU.")
    torch.manual_seed(0)

    hidden = 3072
    layers = 12
    model = DeepSeekToy(hidden, layers).to(device)
    if device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")
    else:
        print("Running in eager mode because CUDA is unavailable.")

    batch = 8
    seq = 1024
    data = torch.randn(batch, seq, hidden, device=device)

    with torch.inference_mode():
        out = model(data)
        checksum = out.square().mean().sqrt()

    tokens = batch * seq
    print(f"DeepSeek v3 synthetic forward pass complete on {tokens} tokens.")
    print(f"Output RMS: {checksum.item():.4f}")


if __name__ == "__main__":
    main()
