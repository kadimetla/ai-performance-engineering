"""optimized_occupancy.py - Higher occupancy kernel."""

from __future__ import annotations

from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedOccupancyBenchmark(BaseBenchmark):
    """Higher occupancy by batching work."""
    
    def __init__(self):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.N = 1_000_000
        tokens = self.N
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        self.data = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        assert self.data is not None
        with self._nvtx_range("occupancy_optimized"):
            # Larger chunk to improve occupancy and reduce launch count
            chunk_size = 32_000
            for i in range(0, self.N, chunk_size):
                end = min(i + chunk_size, self.N)
                self.data[i:end] = self.data[i:end] * 2.0
            self._synchronize()
    
    def teardown(self) -> None:
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}, got {self.data.shape[0]}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedOccupancyBenchmark()
