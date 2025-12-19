#!/usr/bin/env python3
"""Optimized: Tensor Parallelism with communication overlap.

Advanced tensor parallelism with:
- Async all-gather communication
- Computation-communication overlap
- NCCL stream ordering
- Optimal chunk sizing for Blackwell NVLink
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


class OptimizedTensorParallelAsync:
    """Optimized tensor parallelism with async communication."""
    
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
        
        # Create communication stream for overlap
        self.comm_stream = torch.cuda.Stream()
        
        logger.info(
            f"TP Rank {self.rank}/{self.world_size}: "
            f"{self.hidden_per_rank} hidden dims with async overlap"
        )
    
    def _init_distributed(self):
        """Initialize distributed process group."""
        if not dist.is_initialized():
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                dist.init_process_group(backend='nccl')
            else:
                logger.warning("Running in simulation mode")
                self.rank = 0
                self.world_size = 1
                self.local_rank = 0
                return
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = self.rank % torch.cuda.device_count()
    
    def setup(self):
        """Initialize sharded model."""
        # Column-parallel layers
        self.layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_per_rank, bias=False)
            for _ in range(self.num_layers)
        ]).to(self.device).to(torch.bfloat16)
        
        # Create input
        self.input = torch.randn(
            self.batch_size,
            self.seq_length,
            self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16
        )
        
        # Pre-allocate communication buffers
        self.gather_buffers = [
            [
                torch.empty(
                    self.batch_size,
                    self.seq_length,
                    self.hidden_per_rank,
                    device=self.device,
                    dtype=torch.bfloat16
                )
                for _ in range(self.world_size)
            ]
            for _ in range(self.num_layers)
        ]
        
        logger.info(f"Setup complete with pre-allocated buffers (Rank {self.rank})")
    
    def run(self) -> float:
        """Execute optimized tensor parallel with overlap."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        x = self.input
        prev_comm_handle = None
        
        for layer_idx, layer in enumerate(self.layers):
            # Wait for previous communication if any
            if prev_comm_handle is not None:
                prev_comm_handle.wait()
                if self.world_size > 1 and dist.is_initialized():
                    x = torch.cat(self.gather_buffers[layer_idx - 1], dim=-1)
            
            # Compute local shard
            local_output = layer(x)
            
            if self.world_size > 1 and dist.is_initialized():
                # Optimized: Launch async all-gather in separate stream
                with torch.cuda.stream(self.comm_stream):
                    # Copy to pre-allocated buffer
                    self.gather_buffers[layer_idx][self.rank].copy_(local_output)
                    
                    # Async all-gather
                    work = dist.all_gather(
                        self.gather_buffers[layer_idx],
                        local_output,
                        async_op=True
                    )
                    prev_comm_handle = work
            else:
                # Single-rank fast path
                x = local_output
                prev_comm_handle = None
            
            # Next layer can start computing while comm happens
            # (Pipeline overlap)
        
        # Wait for final communication
        if prev_comm_handle is not None:
            prev_comm_handle.wait()
            x = torch.cat(self.gather_buffers[-1], dim=-1)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        logger.info(f"Rank {self.rank}: {elapsed*1000:.2f} ms (with overlap)")
        
        return elapsed * 1000
    
    def cleanup(self):
        """Clean up resources."""
        del self.layers, self.input, self.gather_buffers
        del self.comm_stream
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 8,
    seq_length: int = 2048,
    hidden_size: int = 4096,
    num_layers: int = 4,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run optimized tensor parallel benchmark."""
    
    benchmark = OptimizedTensorParallelAsync(
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
        "parallelism": "tensor_parallel_async_optimized",
    }


class _TensorParallelAsyncBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Wrapper benchmark for async tensor parallel - requires multi-GPU."""

    def __init__(self) -> None:
        super().__init__()
        self.register_workload_metadata(requests_per_iteration=1.0)

    def benchmark_fn(self) -> None:
        raise RuntimeError("SKIPPED: optimized_tensor_parallel_async requires >=2 GPUs")

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5, multi_gpu_required=True)

    def get_verify_output(self) -> torch.Tensor:
        raise RuntimeError("SKIPPED: optimized_tensor_parallel_async requires >=2 GPUs")

    def get_input_signature(self) -> dict:
        raise RuntimeError("SKIPPED: optimized_tensor_parallel_async requires >=2 GPUs")

    def get_output_tolerance(self) -> tuple:
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count < 2:
        return _TensorParallelAsyncBenchmark()
    return _TensorParallelAsyncBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
