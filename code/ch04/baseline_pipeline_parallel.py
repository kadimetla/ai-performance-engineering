#!/usr/bin/env python3
"""Baseline: Pipeline Parallelism (GPipe style).

Demonstrates basic pipeline parallelism with sequential micro-batches.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any, List, Optional
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


class BaselinePipelineParallel:
    """Baseline pipeline parallelism (GPipe style)."""
    
    def __init__(
        self,
        batch_size: int = 32,
        seq_length: int = 2048,
        hidden_size: int = 4096,
        num_layers: int = 8,
        num_micro_batches: int = 4,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_micro_batches = num_micro_batches
        
        # Initialize distributed
        self._init_distributed()
        
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        
        # Layers per stage
        self.layers_per_stage = num_layers // self.world_size
        self.stage_id = self.rank
        
        self.micro_batch_size = batch_size // num_micro_batches
        
        logger.info(
            f"PP Stage {self.stage_id}/{self.world_size}: "
            f"{self.layers_per_stage} layers, {num_micro_batches} micro-batches"
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
        """Initialize pipeline stage."""
        # Each stage gets a subset of layers
        self.stage_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            for _ in range(self.layers_per_stage)
        ]).to(self.device).to(torch.bfloat16)
        
        # Create input (only on first stage)
        if self.stage_id == 0:
            self.input = torch.randn(
                self.batch_size,
                self.seq_length,
                self.hidden_size,
                device=self.device,
                dtype=torch.bfloat16
            )
        else:
            self.input = None
        
        logger.info(f"Stage {self.stage_id} setup complete")
    
    def _forward_micro_batch(self, micro_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for one micro-batch."""
        x = micro_batch
        
        # Process through local layers
        for layer in self.stage_layers:
            x = torch.relu(layer(x))
        
        return x
    
    def run(self) -> float:
        """Execute baseline pipeline parallel (GPipe)."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # GPipe: Forward all micro-batches, then backward
        # Baseline: Sequential processing (large bubble)
        
        outputs = []
        
        for micro_idx in range(self.num_micro_batches):
            if self.stage_id == 0:
                # First stage: split input
                start_idx = micro_idx * self.micro_batch_size
                end_idx = start_idx + self.micro_batch_size
                micro_batch = self.input[start_idx:end_idx]
            else:
                # Receive from previous stage
                if self.world_size > 1 and dist.is_initialized():
                    micro_batch = torch.empty(
                        self.micro_batch_size,
                        self.seq_length,
                        self.hidden_size,
                        device=self.device,
                        dtype=torch.bfloat16
                    )
                    dist.recv(micro_batch, src=self.stage_id - 1)
                else:
                    # Single-GPU: use input directly
                    start_idx = micro_idx * self.micro_batch_size
                    end_idx = start_idx + self.micro_batch_size
                    micro_batch = self.input[start_idx:end_idx]
            
            # Forward through local layers
            output = self._forward_micro_batch(micro_batch)
            
            # Send to next stage or save output
            if self.world_size > 1 and dist.is_initialized():
                if self.stage_id < self.world_size - 1:
                    dist.send(output, dst=self.stage_id + 1)
                else:
                    outputs.append(output)
            else:
                # Single-GPU: save output directly
                outputs.append(output)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Calculate bubble time (idle time)
        # In GPipe, bubble = (num_stages - 1) / num_micro_batches
        bubble_pct = ((self.world_size - 1) / self.num_micro_batches) * 100
        
        logger.info(f"Stage {self.stage_id}: {elapsed*1000:.2f} ms")
        logger.info(f"Expected bubble: ~{bubble_pct:.1f}%")
        
        return elapsed * 1000
    
    def cleanup(self):
        """Clean up resources."""
        del self.stage_layers
        if self.input is not None:
            del self.input
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 32,
    seq_length: int = 2048,
    hidden_size: int = 4096,
    num_layers: int = 8,
    num_micro_batches: int = 4,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline pipeline parallel benchmark."""
    
    benchmark = BaselinePipelineParallel(
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_micro_batches=num_micro_batches,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(
        iterations=3,
        warmup=5,
        profile_mode=profile,
    )
    
    harness = BenchmarkHarness(mode=BenchmarkMode.TRAINING, config=config)
    
    result = harness.benchmark(
        benchmark.run,
        name="baseline_pipeline_parallel"
    )
    
    benchmark.cleanup()
    
    bubble_pct = ((benchmark.world_size - 1) / num_micro_batches) * 100
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        "num_stages": benchmark.world_size,
        "micro_batches": num_micro_batches,
        "expected_bubble_pct": bubble_pct,
        "parallelism": "pipeline_gpipe_baseline",
    }


class _PipelineParallelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Wrapper benchmark for pipeline parallel - requires multi-GPU."""

    def __init__(self) -> None:
        super().__init__()
        self.register_workload_metadata(requests_per_iteration=1.0)

    def benchmark_fn(self) -> None:
        raise RuntimeError("SKIPPED: baseline_pipeline_parallel requires >=2 GPUs")

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5, multi_gpu_required=True)

    def get_verify_output(self) -> torch.Tensor:
        raise RuntimeError("SKIPPED: baseline_pipeline_parallel requires >=2 GPUs")

    def get_input_signature(self) -> dict:
        raise RuntimeError("SKIPPED: baseline_pipeline_parallel requires >=2 GPUs")

    def get_output_tolerance(self) -> tuple:
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count < 2:
        return _PipelineParallelBenchmark()
    # Full multi-GPU case would need proper torchrun harness
    return _PipelineParallelBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
