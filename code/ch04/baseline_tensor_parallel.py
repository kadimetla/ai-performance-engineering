#!/usr/bin/env python3
"""Baseline: Tensor Parallelism without communication overlap.

Demonstrates basic tensor parallelism without optimization.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any, Optional
import sys
from pathlib import Path
import time
import os

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from ch04.verification_payload_mixin import VerificationPayloadMixin
from core.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineTensorParallel:
    """Baseline tensor parallelism without overlap."""
    
    def __init__(
        self,
        batch_size: int = 8,
        seq_length: int = 2048,
        hidden_size: int = 4096,
        num_layers: int = 4,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize distributed
        self._init_distributed()
        
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        
        # Hidden size per rank
        self.hidden_per_rank = hidden_size // self.world_size
        
        logger.info(f"TP Rank {self.rank}/{self.world_size}: {self.hidden_per_rank} hidden dims")
    
    def _init_distributed(self):
        """Initialize distributed process group."""
        if not dist.is_initialized():
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                dist.init_process_group(backend='nccl')
            else:
                logger.warning("Running in simulation mode (no distributed)")
                self.rank = 0
                self.world_size = 1
                self.local_rank = 0
                return
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = self.rank % torch.cuda.device_count()
    
    def setup(self):
        """Initialize sharded model."""
        # Column-parallel linear layers (each rank gets a slice)
        self.layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_per_rank, bias=False)
            for _ in range(self.num_layers)
        ]).to(self.device).to(torch.bfloat16)
        
        # Create input (replicated across ranks)
        self.input = torch.randn(
            self.batch_size,
            self.seq_length,
            self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16
        )
        
        logger.info(f"Setup complete (Rank {self.rank})")
    
    def run(self) -> float:
        """Execute baseline tensor parallel forward pass."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        x = self.input
        
        for layer in self.layers:
            # Baseline: Compute local shard
            local_output = layer(x)
            
            # Baseline: AllGather after computation (no overlap)
            if self.world_size > 1 and dist.is_initialized():
                output_list = [torch.empty_like(local_output) for _ in range(self.world_size)]
                dist.all_gather(output_list, local_output)
                x = torch.cat(output_list, dim=-1)
            else:
                x = local_output
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        logger.info(f"Rank {self.rank}: {elapsed*1000:.2f} ms")
        
        return elapsed * 1000
    
    def cleanup(self):
        """Clean up resources."""
        del self.layers, self.input
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 8,
    seq_length: int = 2048,
    hidden_size: int = 4096,
    num_layers: int = 4,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline tensor parallel benchmark."""
    
    benchmark = BaselineTensorParallel(
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    benchmark.setup()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    elapsed_ms = benchmark.run()
    t1.record()
    torch.cuda.synchronize()
    _ = t0.elapsed_time(t1)
    benchmark.cleanup()
    
    return {
        "mean_time_ms": elapsed_ms,
        "world_size": benchmark.world_size,
        "parallelism": "tensor_parallel_baseline",
    }


class _TensorParallelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Wrapper benchmark for tensor parallel - requires multi-GPU."""

    def __init__(self) -> None:
        super().__init__()
        self.register_workload_metadata(requests_per_iteration=1.0)
        probe = torch.zeros(1, device=self.device)
        output = torch.zeros(1, device=self.device)
        self._set_verification_payload(
            inputs={"probe": probe},
            output=output,
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
        )

    def benchmark_fn(self) -> None:
        raise RuntimeError("SKIPPED: baseline_tensor_parallel requires >=2 GPUs")

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5, multi_gpu_required=True)

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count < 2:
        return _TensorParallelBenchmark()
    return _TensorParallelBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
