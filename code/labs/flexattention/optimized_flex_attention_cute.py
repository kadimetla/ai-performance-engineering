"""Optimized FlexAttention CuTe DSL variant (compiled wrapper)."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import enable_tf32
from labs.flexattention.flexattention_common import build_qkv_inputs, resolve_device

try:
    from flash_attn.cute.interface import _flash_attn_fwd
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "SKIPPED: flash-attn with CuTe DSL support is required (pip install flash-attn)"
    ) from exc

DEFAULT_COMPILE_MODE = os.getenv("TORCH_COMPILE_MODE", "reduce-overhead")


class _CompiledCuteAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return _flash_attn_fwd(
            q,
            k,
            v,
        )


class OptimizedFlexAttentionCuteBenchmark(BaseBenchmark):
    """CuTe DSL path wrapped in torch.compile."""

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
        self.module: Optional[nn.Module] = None
        self.compiled = None
        # Best-effort: allow attempts on any arch; failures will surface at runtime
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        enable_tf32()
        torch._inductor.config.triton.cudagraphs = True
        torch._inductor.config.max_autotune = True

        self.q, self.k, self.v = build_qkv_inputs(
            batch=self.batch,
            heads=self.heads,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        self.module = _CompiledCuteAttention().to(self.device)
        self.compiled = torch.compile(self.module, mode=DEFAULT_COMPILE_MODE, fullgraph=True, dynamic=None)

        with torch.inference_mode():
            self.compiled(self.q, self.k, self.v)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if any(t is None for t in (self.q, self.k, self.v)) or self.compiled is None:
            raise RuntimeError("CuTe FlexAttention compiled module is not initialized")

        with self._nvtx_range("flexattention_cute_compiled"):
            with torch.inference_mode():
                _ = self.compiled(
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
        self.module = None
        self.compiled = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=12, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedFlexAttentionCuteBenchmark()
