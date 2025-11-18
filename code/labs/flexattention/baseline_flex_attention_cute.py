"""Baseline FlexAttention CuTe DSL variant (no torch.compile)."""

from __future__ import annotations

from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
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


class BaselineFlexAttentionCuteBenchmark(BaseBenchmark):
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
        # Best-effort: allow attempts on any arch; failures will surface at runtime
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
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
                _flash_attn_fwd(
                    self.q,
                    self.k,
                    self.v,
                )
            self._synchronize()

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.q = None
        self.k = None
        self.v = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=12, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineFlexAttentionCuteBenchmark()
