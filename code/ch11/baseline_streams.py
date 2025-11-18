"""baseline_streams.py - Sequential kernel execution (baseline).

Demonstrates sequential kernel execution without streams.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineStreamsBenchmark(BaseBenchmark):
    """Sequential execution - no overlap."""
    
    def __init__(self):
        super().__init__()
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.N = 5_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        self.data1 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.data2 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.data3 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self._synchronize()
        processed = float(self.N * 3)
        self.register_workload_metadata(
            tokens_per_iteration=processed,
            requests_per_iteration=1.0,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential kernel execution."""
        with self._nvtx_range("baseline_streams_sequential"):
            self.data1 = self.data1 * 2.0
            self._synchronize()
            self.data2 = self.data2 * 2.0
            self._synchronize()
            self.data3 = self.data3 * 2.0
            self._synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data1 = None
        self.data2 = None
        self.data3 = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data1 is None:
            return "Data1 tensor not initialized"
        if self.data2 is None:
            return "Data2 tensor not initialized"
        if self.data3 is None:
            return "Data3 tensor not initialized"
        if not torch.isfinite(self.data1).all():
            return "Data1 contains non-finite values"
        if not torch.isfinite(self.data2).all():
            return "Data2 contains non-finite values"
        if not torch.isfinite(self.data3).all():
            return "Data3 contains non-finite values"
        return None


def get_benchmark() -> BaselineStreamsBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineStreamsBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Streams: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

