"""baseline_stream_ordered.py - Serial execution on default stream."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineStreamOrderedBenchmark(BaseBenchmark):
    """Sequential work on the default stream (no overlap)."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.requests: Optional[list[torch.Tensor]] = None
        self.outputs: Optional[list[torch.Tensor]] = None
        self.batch_size = 64
        self.hidden_dim = 1024
        self.num_streams = 8

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ).to(self.device).half().eval()

        self.requests = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            for _ in range(self.num_streams)
        ]
        self.outputs = [torch.empty_like(req) for req in self.requests]
        self._synchronize()
        tokens = float(self.batch_size * self.hidden_dim * self.num_streams)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=float(self.num_streams),
        )

    def benchmark_fn(self) -> None:
        assert self.model is not None
        assert self.requests is not None and self.outputs is not None

        with self._nvtx_range("stream_ordered"):
            with torch.no_grad():
                for request, output in zip(self.requests, self.outputs):
                    output.copy_(self.model(request))
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.requests = None
        self.outputs = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.outputs is None:
            return "Outputs not initialized"
        return None


def get_benchmark() -> BaselineStreamOrderedBenchmark:
    return BaselineStreamOrderedBenchmark()
