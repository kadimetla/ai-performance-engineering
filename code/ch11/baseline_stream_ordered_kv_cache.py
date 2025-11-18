"""baseline_stream_ordered_kv_cache.py - Single-stream KV cache updates."""

from __future__ import annotations

from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineStreamOrderedKvCacheBenchmark(BaseBenchmark):
    """Baseline: update buffers sequentially on the default stream."""

    def __init__(self):
        super().__init__()
        self.data1: Optional[torch.Tensor] = None
        self.data2: Optional[torch.Tensor] = None
        self.data3: Optional[torch.Tensor] = None
        self.N = 5_000_000

    def setup(self) -> None:
        torch.manual_seed(42)
        self.data1 = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.data2 = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.data3 = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self._synchronize()
        tokens = float(self.N * 3)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=1.0,
        )

    def benchmark_fn(self) -> None:
        assert self.data1 is not None and self.data2 is not None and self.data3 is not None
        with self._nvtx_range("stream_ordered_kv_cache"):
            self.data1 = self.data1 * 2.0
            self.data2 = self.data2 * 2.0
            self.data3 = self.data3 * 2.0
        self._synchronize()

    def teardown(self) -> None:
        self.data1 = None
        self.data2 = None
        self.data3 = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.data1 is None or self.data2 is None or self.data3 is None:
            return "Buffers not initialized"
        return None


def get_benchmark() -> BaselineStreamOrderedKvCacheBenchmark:
    return BaselineStreamOrderedKvCacheBenchmark()
