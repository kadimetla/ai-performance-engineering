"""optimized_moe_shared_expert_overlap.py - Stream-overlapped MoE dispatch.

Simulates Megatron-Core-style `--moe-shared-expert-overlap` by splitting router
and expert compute across CUDA streams so communication/computation can overlap.
The computation is intentionally lightweight so it runs anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class OverlappedMoE(nn.Module):
    def __init__(self, hidden_dim: int = 1024, num_experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()) for _ in range(num_experts)]
        )
        self.combine = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.gate(tokens)
        top2_w, top2_idx = torch.topk(F.softmax(logits, dim=-1), k=2, dim=-1)

        batch, seq, hidden = tokens.shape
        flat_tokens = tokens.view(batch * seq, hidden)
        flat_idx = top2_idx.view(batch * seq, 2)
        flat_w = top2_w.view(batch * seq, 2)

        out = torch.zeros_like(flat_tokens)
        streams = [torch.cuda.Stream(device=tokens.device) for _ in range(2)]

        # Two-way overlap: process the top-2 experts on separate streams.
        for s_idx, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                expert_id = int(flat_idx[0, s_idx].item()) % len(self.experts)
                expert = self.experts[expert_id]
                contrib = expert(flat_tokens) * flat_w[:, s_idx:s_idx + 1]
                out += contrib

        torch.cuda.synchronize(tokens.device)
        return self.combine(out.view(batch, seq, hidden))


class OptimizedMoeOverlapBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[OverlappedMoE] = None
        self.inputs: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(tokens_per_iteration=1024.0)

    def setup(self) -> None:
        torch.manual_seed(4)
        hidden = 1024
        batch = 64
        seq = 16
        self.model = OverlappedMoE(hidden_dim=hidden, num_experts=4).to(self.device).to(torch.bfloat16)
        self.inputs = torch.randn(batch, seq, hidden, device=self.device, dtype=torch.bfloat16)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if self.model is None or self.inputs is None:
            raise RuntimeError("SKIPPED: overlapped MoE not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_overlap_optimized", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.inputs)
        torch.cuda.synchronize(self.device)
        return {}

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    return OptimizedMoeOverlapBenchmark()
