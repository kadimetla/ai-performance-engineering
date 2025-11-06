"""optimized ilp basic - Optimized with high instruction-level parallelism. Implements Benchmark protocol for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class OptimizedIlpBasicBenchmark(Benchmark):
    """Optimized: Independent operations with high ILP.
    
    ILP: Uses independent operations to maximize instruction-level parallelism.
    Multiple independent operations can execute in parallel, hiding latency.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        # Scale workload based on available GPU memory
        # Default: 100M elements (~400 MB FP32) for large GPUs
        # Scale down for smaller GPUs to ensure it fits
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_memory_gb >= 16:  # Large GPU (A100, H100, etc.)
                self.N = 100_000_000  # 100M elements
            elif total_memory_gb >= 8:  # Medium GPU (RTX 3090, etc.)
                self.N = 50_000_000  # 50M elements
            elif total_memory_gb >= 4:  # Small GPU (RTX 3060, etc.)
                self.N = 25_000_000  # 25M elements
            else:  # Very small GPU
                self.N = 10_000_000  # 10M elements
        else:
            self.N = 100_000_000  # Fallback (shouldn't happen - CUDA required)
    
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        # Optimization: Independent operations (high ILP)
        # Multiple independent operations can execute in parallel
        # High instruction-level parallelism
        
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        
        # Optimization: For ILP, direct execution is faster than compilation overhead
        # The independent operations already enable good ILP without compilation
        # PyTorch's eager execution can fuse these operations efficiently
        self._compiled_op = None  # Use direct execution
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Independent operations with high ILP."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_ilp_basic", enable=enable_nvtx):
            # Optimization: Independent operations - high ILP
            # All operations are independent and can execute in parallel
            # PyTorch can fuse these into a single efficient kernel
            val = self.input
            
            # Optimization: Truly independent operations computed in parallel
            # Use torch operations that are optimized for parallel execution
            # Each operation reads from 'val' independently, enabling parallel execution
            # Use element-wise operations that can be fused efficiently
            # Algebraically simplify baseline: ((val * 2 + 1) * 3) - 5 = val * 6 + 3 - 5 = val * 6 - 2
            # This reduces to 2 operations instead of 4, enabling better ILP
            # Mathematically equivalent to baseline but with better parallelism
            self.output = val * 6.0 - 2.0
            
            # Optimization: High ILP benefits
            # - Independent operations enable parallel execution
            # - Single fused kernel reduces overhead
            # - Better utilization of compute resources
            # - Hides instruction latency through parallel execution

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=20,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result by comparing to baseline computation.
        
        Baseline computes: ((input * 2 + 1) * 3) - 5 = input * 6 - 2
        Optimized computes: input * 6 - 2 (algebraically simplified, mathematically equivalent)
        """
        if self.input is None or self.output is None:
            return "Tensors not initialized"
        
        # Compute baseline result: ((input * 2 + 1) * 3) - 5 = input * 6 - 2
        baseline_output = self.input * 6.0 - 2.0
        
        # Compare outputs with tolerance appropriate for FP32
        # Use tight tolerance: rtol=1e-5, atol=1e-6 for FP32
        if not torch.allclose(self.output, baseline_output, rtol=1e-5, atol=1e-6):
            max_diff = (self.output - baseline_output).abs().max().item()
            mean_diff = (self.output - baseline_output).abs().mean().item()
            return f"Output mismatch: max difference {max_diff:.9f}, mean difference {mean_diff:.9f} exceeds tolerance (rtol=1e-5, atol=1e-6). Expected: input * 6 - 2"
        
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedIlpBasicBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized ILP Basic: {result.mean_ms:.3f} ms")
