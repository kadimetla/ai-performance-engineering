"""Baseline FlexAttention lab: uncompiled CuTe DSL path."""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn.attention.flex_attention import flex_attention

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.flexattention.flexattention_common import (
    build_flex_attention_inputs,
    make_relative_bias_score_mod,
    resolve_device,
)


class BaselineFlexAttentionBenchmark(BaseBenchmark):
    """Eager FlexAttention using the DSL score_mod + block_mask."""

    def __init__(self) -> None:
        super().__init__()
        self.dtype = torch.bfloat16
        self.seq_len = 1024
        self.batch = 2
        self.heads = 8
        self.head_dim = 64
        self.block_size = 64
        self.doc_span = 256
        self.inputs = None
        self.score_mod = None
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        self.inputs = build_flex_attention_inputs(
            batch=self.batch,
            heads=self.heads,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            doc_span=self.doc_span,
            block_size=self.block_size,
            dtype=self.dtype,
            device=self.device,
        )
        self.score_mod = make_relative_bias_score_mod(self.inputs.rel_bias)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self.score_mod is None:
            raise RuntimeError("FlexAttention inputs are not initialized")

        with self._nvtx_range("flexattention_baseline"):
            with torch.inference_mode():
                _ = flex_attention(
                    self.inputs.q,
                    self.inputs.k,
                    self.inputs.v,
                    score_mod=self.score_mod,
                    block_mask=self.inputs.block_mask,
                )
            self._synchronize()

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None
        self.score_mod = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=12, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.inputs is None or self.score_mod is None:
            return "FlexAttention inputs are not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineFlexAttentionBenchmark()
