"""optimized_streams.py - Concurrent kernel execution with streams (optimized).

Demonstrates overlapping kernel execution using CUDA streams."""

from __future__ import annotations

from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class OptimizedStreamsBenchmark(BaseBenchmark):
    """Concurrent execution - kernels overlap.
    
    Note: For warp specialization examples, see optimized_streams_warp_specialized.cu
    which demonstrates warp specialization with __activemask for efficient stream-ordered execution.
    """
    
    def __init__(self):
        super().__init__()
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.stream1 = None
        self.stream2 = None
        self.stream3 = None
        self.N = 5_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors and streams."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        self.data1 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.data2 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.data3 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        
        # Create separate streams for concurrent execution
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
        self.stream3 = torch.cuda.Stream()
        
        self._synchronize()
        processed = float(self.N * 3)
        self.register_workload_metadata(
            tokens_per_iteration=processed,
            requests_per_iteration=1.0,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Concurrent kernel execution with streams."""
        with self._nvtx_range("streams"):
            # Launch kernels on different streams - they can overlap
            with torch.cuda.stream(self.stream1):
                self.data1 = self.data1 * 2.0
            
            with torch.cuda.stream(self.stream2):
                self.data2 = self.data2 * 2.0
            
            with torch.cuda.stream(self.stream3):
                self.data3 = self.data3 * 2.0
            
            # Synchronize all streams
            self.stream1.synchronize()
            self.stream2.synchronize()
            self.stream3.synchronize()
        self._synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.stream1 = None
        self.stream2 = None
        self.stream3 = None
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


def get_benchmark() -> OptimizedStreamsBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedStreamsBenchmark()

