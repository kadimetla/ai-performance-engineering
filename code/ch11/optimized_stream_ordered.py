"""optimized_stream_ordered.py - Optimized multi-stream overlap example.

Demonstrates launching work across multiple CUDA streams with explicit events."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class OptimizedStreamOrderedBenchmark(BaseBenchmark):
    """Optimized: Overlap work across multiple CUDA streams."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.requests = None
        self.outputs = None
        self.streams = None
        self.num_streams = 8
        self.hidden_dim = 1024
        self.batch_size = 64
    
    def setup(self) -> None:
        """Setup: initialize lightweight model and per-stream buffers."""
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
        self.outputs = [
            torch.empty_like(req) for req in self.requests
        ]
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        self._synchronize()
        tokens = float(self.batch_size * self.hidden_dim * self.num_streams)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=float(self.num_streams),
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Launch work on dedicated streams to overlap execution."""
        with self._nvtx_range("optimized_stream_ordered"):
            with torch.no_grad():
                for stream, request, output in zip(self.streams, self.requests, self.outputs):
                    with torch.cuda.stream(stream):
                        output.copy_(self.model(request))
                for stream in self.streams:
                    stream.synchronize()
        self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.requests = None
        self.outputs = None
        self.streams = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None

def get_benchmark() -> OptimizedStreamOrderedBenchmark:
    """Factory function for harness discovery."""
    return OptimizedStreamOrderedBenchmark()
