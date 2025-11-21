"""optimized_context_parallelism.py - Optimized context parallelism for long sequences."""

from __future__ import annotations

import argparse
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
    
    def __init__(
        self,
        sequence_length: Optional[int] = None,
        cp_ranks: Optional[int] = None,
    ):
        super().__init__()
        self.models: Optional[list[nn.Module]] = None
        self.input_sequence: Optional[torch.Tensor] = None
        self.sequence_chunks: Optional[list[torch.Tensor]] = None
        self._chunk_sizes: list[int] = []
        self.sequence_length = sequence_length if sequence_length is not None else 8192  # Long sequence for training
        max_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if cp_ranks is not None:
            max_devices = max(1, min(max_devices, int(cp_ranks)))
        self.num_gpus = max_devices
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
        self._chunk_sizes = []
        for gpu_id in range(self.num_gpus):
            start_idx = gpu_id * tokens_per_gpu
            end_idx = start_idx + tokens_per_gpu if gpu_id < self.num_gpus - 1 else self.sequence_length
            chunk = self.input_sequence[start_idx:end_idx].to(torch.device(f"cuda:{gpu_id}"))
            self.sequence_chunks.append(chunk)
            self._chunk_sizes.append(chunk.size(0))

        self._synchronize()
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.sequence_length),
        )
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Context parallelism processing of long sequence."""
        assert self.models is not None and self.sequence_chunks is not None
        
        with self._nvtx_range("optimized_context_parallelism"):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
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
        self._chunk_sizes = []
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=3,
            measurement_timeout_seconds=120,
            multi_gpu_required=True,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        if not self.sequence_chunks:
            return None
        return {
            "context_length": self.sequence_length,
            "cp_ranks": self.num_gpus,
            "tokens_per_rank_min": min(self._chunk_sizes) if self._chunk_sizes else 0,
            "tokens_per_rank_max": max(self._chunk_sizes) if self._chunk_sizes else 0,
        }
    
    def validate_result(self) -> Optional[str]:
        if self.models is None or len(self.models) == 0:
            return "Models not initialized"
        if self.sequence_chunks is None or len(self.sequence_chunks) == 0:
            return "Sequence chunks not initialized"
        return None


def get_benchmark(args: Optional[argparse.Namespace] = None) -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    sequence_length = getattr(args, "sequence_length", None) if args else None
    cp_ranks = getattr(args, "cp_ranks", None) if args else None
    return OptimizedContextParallelismBenchmark(
        sequence_length=sequence_length,
        cp_ranks=cp_ranks,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized context parallel benchmark")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Total context length to shard across ranks (default: 8192)",
    )
    parser.add_argument(
        "--cp-ranks",
        type=int,
        default=None,
        help="Number of GPUs (ranks) to use for context parallelism (default: all available)",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    benchmark = get_benchmark(args)
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
