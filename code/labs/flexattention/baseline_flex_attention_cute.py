"""Baseline FlexAttention CuTe DSL variant (no torch.compile)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.flexattention.flexattention_common import (
    build_qkv_inputs,
    resolve_device,
)

try:
    from flash_attn.cute.interface import _flash_attn_fwd
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "SKIPPED: flash-attn with CuTe DSL support is required (pip install flash-attn)"
    ) from exc


class BaselineFlexAttentionCuteBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """CuTe DSL path without torch.compile."""

    def __init__(self) -> None:
        super().__init__()
        self.dtype = torch.bfloat16
        self.seq_len = 1024
        self.batch = 2
        self.heads = 8
        self.head_dim = 64
        self.block_size = 128
        self.doc_span = 256
        self.q = None
        self.k = None
        self.v = None
        self.output: Optional[torch.Tensor] = None
        # Best-effort: allow attempts on any arch; failures will surface at runtime
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.q, self.k, self.v = build_qkv_inputs(
            batch=self.batch,
            heads=self.heads,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if any(t is None for t in (self.q, self.k, self.v)):
            raise RuntimeError("CuTe FlexAttention inputs are not initialized")

        with self._nvtx_range("flexattention_cute_baseline"):
            with torch.inference_mode():
                result = _flash_attn_fwd(
                    self.q,
                    self.k,
                    self.v,
                )
            self._synchronize()
            output_tensor = result[0] if isinstance(result, (tuple, list)) else result
            self.output = output_tensor.detach().float().clone()
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")
        self._set_verification_payload(
            inputs={"q": self.q.detach(), "k": self.k.detach(), "v": self.v.detach()},
            output=self.output,
            batch_size=self.batch,
            parameter_count=0,
            precision_flags={"bf16": True, "fp16": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.q = None
        self.k = None
        self.v = None
        self.output = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=12, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics."""
        # Estimate problem size for roofline analysis
        n = getattr(self, 'N', 0) or getattr(self, 'hidden_dim', 0) or 4096
        batch = getattr(self, 'batch_size', 1) or getattr(self, 'batch', 1)
        # Simple FLOP estimate for linear layers
        flops = 2.0 * batch * n * n  # Rough estimate
        bytes_moved = batch * n * 4.0  # Input/output bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        return {
            "flex_attention_cute.estimated_flops": flops,
            "flex_attention_cute.estimated_bytes": bytes_moved,
            "flex_attention_cute.arithmetic_intensity": arithmetic_intensity,
        }

    def validate_result(self) -> Optional[str]:
        return None

def get_benchmark() -> BaseBenchmark:
    return BaselineFlexAttentionCuteBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
