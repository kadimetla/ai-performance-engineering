"""optimized_end_to_end_bandwidth.py - Optimized end-to-end bandwidth (optimized).

Optimized end-to-end bandwidth analysis with memory access optimizations.
Uses FP16, better memory layout, and optimized processing.

Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

try:
    import ch20.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass
from typing import Optional

try:
    from common.python.logger import get_logger
    LOGGER = get_logger(__name__)
except ImportError:  # pragma: no cover - logger not available during docs builds
    LOGGER = None

from ch20.inductor_guard import (
    disable_inductor_cudagraph_features,
    restore_inductor_cudagraph_features,
    InductorCudagraphState,
)

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class SimplePipeline(nn.Module):
    """Simple inference pipeline for bandwidth analysis."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OptimizedEndToEndBandwidthBenchmark(Benchmark):
    """Optimized end-to-end bandwidth - optimized processing."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.outputs = None
        self.batch_size = 32
        self.hidden_dim = 1024
        self.num_batches = 10
        self._inductor_cfg_state: InductorCudagraphState = None
        self._used_compiled_model = False
        self._compile_error: Optional[str] = None
    
    def setup(self) -> None:
        """Setup: Initialize optimized model and data."""
        
        self._inductor_cfg_state = disable_inductor_cudagraph_features()
        try:
            # Optimization: Enable cuDNN benchmarking for optimal kernel selection
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            torch.manual_seed(42)
            
            # Optimized: FP16, compiled model
            model = SimplePipeline(hidden_dim=self.hidden_dim).to(self.device).half().eval()

            try:
                compiled_model = torch.compile(model, mode="reduce-overhead")
                self.model = compiled_model
                self._used_compiled_model = True
            except Exception as exc:  # torch.compile can fail on newer architectures/toolchains
                self._compile_error = f"{exc.__class__.__name__}: {exc}"
                self._used_compiled_model = False
                self.model = model  # Fallback to eager execution
                if LOGGER is not None:
                    LOGGER.warning(
                        "torch.compile failed for %s; falling back to eager execution.",
                        self.__class__.__name__,
                        exc_info=exc,
                    )
            # Warmup to trigger compilation (or validate eager fallback) and catch errors early
            test_input = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            for _ in range(3):
                with torch.no_grad():
                    _ = self.model(test_input)
            torch.cuda.synchronize()
            
            # Optimized: FP16, contiguous memory layout
            self.inputs = [
                torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16).contiguous()
                for _ in range(self.num_batches)
            ]
            self.outputs = []
            
            # Warmup
            for inp in self.inputs[:5]:
                with torch.no_grad():
                    _ = self.model(inp)
            torch.cuda.synchronize()
        except Exception:
            restore_inductor_cudagraph_features(self._inductor_cfg_state)
            self._inductor_cfg_state = None
            raise
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - optimized end-to-end."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_end_to_end_bandwidth", enable=enable_nvtx):
            # Optimized processing with better memory access patterns
            torch.cuda.reset_peak_memory_stats()
            
            self.outputs = []
            for inp in self.inputs:
                with torch.no_grad():
                    out = self.model(inp)
                self.outputs.append(out)
            torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.outputs
        torch.cuda.empty_cache()
        restore_inductor_cudagraph_features(self._inductor_cfg_state)
        self._inductor_cfg_state = None
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=True,
            enable_profiling=False,
            use_subprocess=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if len(self.outputs) != self.num_batches:
            return f"Expected {self.num_batches} outputs, got {len(self.outputs)}"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Expose compile mode so results clearly state if a fallback occurred."""
        metrics = {
            "used_torch_compile": self._used_compiled_model,
        }
        if self._compile_error:
            metrics["torch_compile_error"] = self._compile_error
        return metrics


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedEndToEndBandwidthBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized End-to-End Bandwidth: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
