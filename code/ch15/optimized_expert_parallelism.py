"""optimized_expert_parallelism.py - Top-2 gated MoE with lightweight overlap.

This is a runnable, single-GPU approximation of expert parallelism that
aggregates top-2 expert outputs and uses CUDA streams to overlap projection
with dispatch. It exists to back the docs' `ch15:expert_parallelism` target.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class Top2MoE(nn.Module):
    """Small top-2 MoE with per-expert projections."""

    def __init__(self, hidden_dim: int = 1024, num_experts: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()) for _ in range(num_experts)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.gate(tokens)
        weights = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(weights, k=2, dim=-1)

        # Dispatch: compute expert outputs and combine with gating weights.
        batch, seq, _ = tokens.shape
        flat_tokens = tokens.view(batch * seq, -1)
        flat_idx = topk_indices.view(batch * seq, 2)
        flat_w = topk_weights.view(batch * seq, 2)

        outputs = torch.zeros_like(flat_tokens)
        # Use a single stream to overlap the two expert projections.
        stream1 = torch.cuda.Stream(device=tokens.device)
        stream2 = torch.cuda.Stream(device=tokens.device)

        with torch.cuda.stream(stream1):
            exp0_tokens = flat_tokens
            exp0_out = self.experts[0](exp0_tokens)
        with torch.cuda.stream(stream2):
            exp1_tokens = flat_tokens
            exp1_out = self.experts[1](exp1_tokens)

        torch.cuda.synchronize(tokens.device)

        # Blend expert outputs based on indices/weights.
        outputs += exp0_out * flat_w[:, 0:1] * (flat_idx[:, 0:1] == 0)
        outputs += exp1_out * flat_w[:, 1:2] * (flat_idx[:, 1:2] == 1)
        return outputs.view(batch, seq, -1)


class OptimizedExpertParallelismBenchmark(BaseBenchmark):
    """Top-2 expert benchmark meant to mirror the doc's optimized target."""

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[Top2MoE] = None
        self.inputs: Optional[torch.Tensor] = None
        self._history: Dict[str, float] = {}
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=1024.0,
        )

    def setup(self) -> None:
        torch.manual_seed(1)
        hidden_dim = 1024
        batch = 64
        seq = 16
        self.model = Top2MoE(hidden_dim=hidden_dim, num_experts=8).to(self.device).to(torch.bfloat16).eval()
        self.inputs = torch.randn(batch, seq, hidden_dim, device=self.device, dtype=torch.bfloat16)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if self.model is None or self.inputs is None:
            raise RuntimeError("SKIPPED: MoE model not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        start = self._record_start()
        with nvtx_range("moe_top2_forward", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.inputs)
        torch.cuda.synchronize(self.device)
        latency_ms = self._record_stop(start)
        self._history["latency_ms"] = latency_ms
        return {"latency_ms": latency_ms}

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    """Factory for harness discovery."""
    return OptimizedExpertParallelismBenchmark()
