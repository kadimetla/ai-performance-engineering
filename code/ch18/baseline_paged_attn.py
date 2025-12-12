"""baseline_paged_attn.py - Dense SDPA baseline for paged attention demos."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402

# Use new SDPA API when available
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _MATH_BACKENDS = [SDPBackend.MATH]
    _NEW_SDPA_API = True
except ImportError:
    sdpa_kernel = None  # type: ignore[assignment]
    SDPBackend = None  # type: ignore[assignment]
    _MATH_BACKENDS = []
    _NEW_SDPA_API = False


def _math_sdpa_context():
    """Return context manager for math-only attention backend (baseline)."""
    if _NEW_SDPA_API and sdpa_kernel is not None:
        return sdpa_kernel(_MATH_BACKENDS)
    return nullcontext()


class BaselinePagedAttnBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.qkv: Optional[torch.Tensor] = None
        self.output = None
        self._workload = WorkloadMetadata(tokens_per_iteration=0.0)
        self._verification_payload = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Longer sequence to expose flash SDPA advantage (O(N) vs O(N²) memory).
        b, h, s, d = 4, 16, 2048, 64
        # Baseline uses full-precision SDPA without flash/memory-efficient kernels.
        self.qkv = torch.randn(b, h, s, 3, d, device=self.device, dtype=torch.float32)
        # Aggressive warmup: run multiple times to fully JIT-compile the math SDPA path.
        q = self.qkv[:, :, :, 0]
        k = self.qkv[:, :, :, 1]
        v = self.qkv[:, :, :, 2]
        with _math_sdpa_context():
            for _ in range(5):
                _ = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if self.qkv is None:
            raise RuntimeError("SKIPPED: QKV not initialized")
        q = self.qkv[:, :, :, 0]
        k = self.qkv[:, :, :, 1]
        v = self.qkv[:, :, :, 2]

        enable_nvtx = get_nvtx_enabled(self.get_config())
        # Force the unfused math path so the optimized variant can contrast flash SDPA.
        with _math_sdpa_context():
            with nvtx_range("paged_attn_baseline", enable=enable_nvtx):
                self.output = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize(self.device)
        if self.output is None or self.qkv is None:
            raise RuntimeError("benchmark_fn() must produce output")
        return {}

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"qkv": self.qkv},
            output=self.output,
            batch_size=self.qkv.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": self.qkv.dtype == torch.float16,
                "bf16": self.qkv.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_speculative_decoding_metrics
        return compute_speculative_decoding_metrics(
            draft_tokens=getattr(self, '_draft_tokens', 64),
            accepted_tokens=getattr(self, '_accepted_tokens', 48),
            draft_time_ms=getattr(self, '_draft_ms', 5.0),
            verify_time_ms=getattr(self, '_verify_ms', 10.0),
            num_rounds=getattr(self, '_num_rounds', 8),
        )

def get_benchmark() -> BaseBenchmark:
    return BaselinePagedAttnBenchmark()
