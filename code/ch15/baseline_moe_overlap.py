"""baseline_moe_overlap.py - Non-overlapped MoE dispatch baseline.

Runs a simple sequence of (router -> expert -> combine) on a single CUDA stream
to provide a baseline for overlap comparisons.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402
from ch15.verification_payload_mixin import VerificationPayloadMixin  # noqa: E402


class BaselineOverlapMoE(nn.Module):
    def __init__(self, hidden_dim: int = 1024, num_experts: int = 4):
        super().__init__()
        self.output = None
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU()) for _ in range(num_experts)]
        )
        self.combine = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.gate(tokens)
        top1 = torch.argmax(logits, dim=-1)
        outputs = torch.zeros_like(tokens)
        for idx, expert in enumerate(self.experts):
            mask = top1 == idx
            if mask.any():
                outputs[mask] = expert(tokens[mask])
        return self.combine(outputs)


class BaselineMoeOverlapBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[BaselineOverlapMoE] = None
        self.inputs: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(tokens_per_iteration=1024.0)
        self._verify_tokens: Optional[torch.Tensor] = None
        self.output = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        hidden = 1024
        batch = 64
        seq = 16
        self.model = BaselineOverlapMoE(hidden_dim=hidden, num_experts=4).to(self.device).to(torch.bfloat16)
        self.inputs = torch.randn(batch, seq, hidden, device=self.device, dtype=torch.bfloat16)
        torch.cuda.synchronize(self.device)
        self._verify_tokens = self.inputs[:2, :2].detach()

    def benchmark_fn(self) -> Optional[dict]:
        if self.model is None or self.inputs is None:
            raise RuntimeError("SKIPPED: MoE baseline not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_overlap_baseline", enable=enable_nvtx):
            with torch.no_grad():
                self.output = self.model(self.inputs)
        return {}

    def capture_verification_payload(self) -> None:
        if self.model is None or self.inputs is None or self.output is None or self._verify_tokens is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"tokens": self._verify_tokens},
            output=self.output,
            batch_size=int(self.inputs.shape[0]),
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )
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

def get_benchmark() -> BaseBenchmark:
    return BaselineMoeOverlapBenchmark()
