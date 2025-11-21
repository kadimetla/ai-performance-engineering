"""baseline_context_parallelism.py - Baseline sequential processing without context parallelism.

Demonstrates sequential processing of long sequences without context parallelism.
Context parallelism: This baseline processes the entire sequence on a single GPU sequentially.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class BaselineContextParallelismBenchmark(BaseBenchmark):
    """Baseline: Sequential processing without context parallelism (single GPU)."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.input_sequence = None
        self.sequence_length = 8192  # Long sequence for training
        tokens = self.sequence_length
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model and long input sequence."""
        torch.manual_seed(42)
        # Baseline: Sequential processing on single GPU
        # Context parallelism splits long sequences across multiple GPUs
        # This baseline processes the entire sequence sequentially on one GPU
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).eval()
        
        # Long sequence that would benefit from context parallelism
        # Context parallelism splits sequences across GPUs for parallel processing
        self.input_sequence = torch.randn(self.sequence_length, 256, device=self.device)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential processing of long sequence."""
        # Baseline: Process entire sequence sequentially on single GPU
        # No context parallelism - all tokens processed on one device
        with self._nvtx_range("baseline_context_parallelism"):
            with torch.no_grad():
                output = self.model(self.input_sequence)
        self._synchronize()
    
    def teardown(self) -> None:
        """Cleanup: Clear CUDA cache."""
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=3,
            measurement_timeout_seconds=120,
            multi_gpu_required=True,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input_sequence is None:
            return "Input sequence not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineContextParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
