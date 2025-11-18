"""optimized_context_parallelism.py - Optimized context parallelism for long sequences."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class OptimizedContextParallelismBenchmark(BaseBenchmark):
    """Optimized: Context parallelism for long sequences (split across GPUs)."""
    
    def __init__(self):
        super().__init__()
        self.models: Optional[list[nn.Module]] = None
        self.input_sequence: Optional[torch.Tensor] = None
        self.sequence_chunks: Optional[list[torch.Tensor]] = None
        self.sequence_length = 8192  # Long sequence for training
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        tokens = self.sequence_length
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize models on multiple GPUs and split sequence."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        self.models = []
        for gpu_id in range(self.num_gpus):
            model = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
            ).to(torch.device(f"cuda:{gpu_id}")).eval()
            self.models.append(model)
        
        self.input_sequence = torch.randn(self.sequence_length, 256, device=self.device)
        
        tokens_per_gpu = self.sequence_length // self.num_gpus
        self.sequence_chunks = []
        for gpu_id in range(self.num_gpus):
            start_idx = gpu_id * tokens_per_gpu
            end_idx = start_idx + tokens_per_gpu if gpu_id < self.num_gpus - 1 else self.sequence_length
            chunk = self.input_sequence[start_idx:end_idx].to(torch.device(f"cuda:{gpu_id}"))
            self.sequence_chunks.append(chunk)
        
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Context parallelism processing of long sequence."""
        assert self.models is not None and self.sequence_chunks is not None
        
        with self._nvtx_range("optimized_context_parallelism"):
            with torch.no_grad():
                outputs = []
                for model, chunk in zip(self.models, self.sequence_chunks):
                    output = model(chunk)
                    outputs.append(output)
        
        for gpu_id in range(self.num_gpus):
            torch.cuda.synchronize(torch.device(f"cuda:{gpu_id}"))
        self._synchronize()
    
    def teardown(self) -> None:
        self.models = None
        self.sequence_chunks = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        if self.models is None or len(self.models) == 0:
            return "Models not initialized"
        if self.sequence_chunks is None or len(self.sequence_chunks) == 0:
            return "Sequence chunks not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedContextParallelismBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
