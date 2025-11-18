"""baseline_memory_standard.py - Standard memory access baseline."""

from __future__ import annotations

from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineMemoryStandardBenchmark(BaseBenchmark):
    """Standard memory access patterns without HBM3e optimizations."""
    
    def __init__(self):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.result: Optional[torch.Tensor] = None
        self.size_mb = 100  # 100 MB
        num_elements = (self.size_mb * 1024 * 1024) // 4
        self.num_elements = num_elements
        bytes_per_iter = num_elements * 4 * 2  # read + write
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(num_elements),
            bytes_per_iteration=float(bytes_per_iter),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        self.data = torch.randn(self.num_elements, device=self.device, dtype=torch.float32)
        self.result = torch.zeros_like(self.data)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        assert self.data is not None
        with self._nvtx_range("baseline_memory_standard"):
            self.result = self.data * 2.0 + 1.0
            if self.result is not None:
                self.result += 0.1
            self._synchronize()
    
    def teardown(self) -> None:
        self.data = None
        self.result = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.data is None:
            return "Data not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineMemoryStandardBenchmark()
