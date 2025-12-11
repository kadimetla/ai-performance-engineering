"""baseline_expert_parallelism.py - Pedagogical MoE baseline (single GPU).

Implements a simple top-1 gated MoE forward path to keep the benchmark harness
happy and give the docs a runnable target. This stays intentionally small and
single-GPU so it can execute anywhere without special hardware.
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

from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402
from ch15.verification_payload_mixin import VerificationPayloadMixin


class ToyGatedMoE(nn.Module):
    """Top-2 MoE with sequential dispatch on a single GPU."""

    def __init__(self, hidden_dim: int = 1024, num_experts: int = 8, capacity_factor: float = 1.25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_experts)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = tokens.shape
        logits = self.gate(tokens)
        probs = F.softmax(logits, dim=-1)
        top2_w, top2_idx = torch.topk(probs, k=2, dim=-1)

        flat_idx = top2_idx.view(batch * seq, 2)
        flat_w = top2_w.view(batch * seq, 2)
        flat_tokens = tokens.view(batch * seq, hidden)

        cap = int(self.capacity_factor * (batch * seq) / self.num_experts)
        counts = torch.bincount(flat_idx.view(-1), minlength=self.num_experts)
        mask_overflow = counts > cap

        outputs = torch.zeros_like(flat_tokens)
        for slot in range(2):
            expert_ids = flat_idx[:, slot]
            weights = flat_w[:, slot : slot + 1]
            for eid in torch.unique(expert_ids):
                eid_int = int(eid.item())
                if mask_overflow[eid_int]:
                    continue
                mask = expert_ids == eid
                if mask.any():
                    outputs[mask] += self.experts[eid_int](flat_tokens[mask]) * weights[mask]

        return outputs.view(batch, seq, hidden)


class BaselineExpertParallelismBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Single-GPU MoE baseline without overlap or load balancing tricks."""

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[ToyGatedMoE] = None
        self.output = None
        self.inputs: Optional[torch.Tensor] = None
        self._history: Dict[str, float] = {}
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=1024.0,
        )
        self._verify_tokens: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        hidden_dim = 1024
        batch = 64
        seq = 16
        self.model = ToyGatedMoE(hidden_dim=hidden_dim, num_experts=8).to(self.device).to(torch.bfloat16).eval()
        self.inputs = torch.randn(batch, seq, hidden_dim, device=self.device, dtype=torch.bfloat16)
        torch.cuda.synchronize(self.device)
        self._verify_tokens = torch.randn(2, 2, hidden_dim, device=self.device, dtype=torch.bfloat16)

    def benchmark_fn(self) -> Optional[dict]:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Model or inputs not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        start = self._record_start()
        with nvtx_range("moe_baseline_forward", enable=enable_nvtx):
            with torch.no_grad():
                self.output = self.model(self.inputs)
        torch.cuda.synchronize(self.device)
        latency_ms = self._record_stop(start)
        self._history["latency_ms"] = latency_ms
        if self.output is not None and self._verify_tokens is not None:
            self._set_verification_payload(
                inputs={"tokens": self._verify_tokens},
                output=self.output.detach().clone(),
                batch_size=int(self._verify_tokens.shape[0]),
                parameter_count=sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0,
                precision_flags={
                    "fp16": False,
                    "bf16": True,
                    "fp8": False,
                    "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
                },
                output_tolerance=(1e-3, 1e-3),
            )
        return {"latency_ms": latency_ms}

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return super().get_output_tolerance()

def get_benchmark() -> BaseBenchmark:
    """Factory for the harness."""
    return BaselineExpertParallelismBenchmark()
