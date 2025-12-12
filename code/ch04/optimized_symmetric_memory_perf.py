#!/usr/bin/env python3
"""Optimized symmetric-memory perf microbench (SymmetricMemory puts only).

Measures latency/bandwidth of direct peer writes using torch.distributed.nn.SymmetricMemory.
Demonstrates the uplift from using direct GPU-to-GPU memory access vs NCCL collectives.
"""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.metrics import compute_memory_transfer_metrics
from ch04.verification_payload_mixin import VerificationPayloadMixin


def symmetric_memory_available() -> bool:
    """Check if SymmetricMemory is available in this PyTorch build."""
    return hasattr(dist, "nn") and hasattr(dist.nn, "SymmetricMemory")


def init_distributed() -> Tuple[int, int, int]:
    """Initialize process group for a single-node demo."""
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60),
        )
        torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), torch.cuda.current_device()


class OptimizedSymmetricMemoryPerfBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized SymmetricMemory peer-put benchmark for direct GPU memory access."""

    def __init__(self, size_mb: float = 16.0):
        super().__init__()
        self.size_mb = size_mb
        self.numel = int((size_mb * 1024 * 1024) / 4)  # float32
        self.local_tensor: Optional[torch.Tensor] = None
        self.peer_buffer: Optional[torch.Tensor] = None
        self.handle = None
        self.rank = 0
        self.world_size = 1
        self.peer_rank = 0
        self._last_avg_ms = 0.0
        self._last_gbps = 0.0
        self._bytes_transferred = 0.0
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        """Initialize distributed and allocate symmetric memory."""
        if not symmetric_memory_available():
            raise RuntimeError(
                "SKIPPED: SymmetricMemory not available. "
                "Install PyTorch with SymmetricMemory support."
            )

        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: symmetric_memory_perf requires >= 2 GPUs")

        self.rank, self.world_size, device_id = init_distributed()
        
        if self.world_size < 2:
            raise RuntimeError("SKIPPED: SymmetricMemory peer-put requires world_size >= 2")

        device = torch.device("cuda", device_id)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.local_tensor = torch.randn(self.numel, device=device, dtype=torch.float32)
        
        # Create symmetric memory handle for direct peer access
        self.handle = dist.nn.SymmetricMemory(self.local_tensor)
        self.peer_rank = (self.rank + 1) % self.world_size
        self.peer_buffer = self.handle.get_buffer(self.peer_rank)
        
        torch.cuda.synchronize()

    def benchmark_fn(self) -> Optional[Dict[str, float]]:
        """Run direct peer copy via SymmetricMemory and measure performance."""
        if self.local_tensor is None or self.peer_buffer is None:
            raise RuntimeError("Tensors not initialized")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        self.peer_buffer.copy_(self.local_tensor, non_blocking=True)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        
        # Direct peer copy: one traversal of the payload
        bytes_moved = self.size_mb * 1024 * 1024
        gbps = (bytes_moved / (elapsed_ms / 1000.0)) / 1e9 if elapsed_ms > 0 else 0.0

        self._last_avg_ms = elapsed_ms
        self._last_gbps = gbps
        self._bytes_transferred = bytes_moved

        return {
            "symmetric_put.elapsed_ms": elapsed_ms,
            "symmetric_put.gbps": gbps,
            "symmetric_put.size_mb": self.size_mb,
            "symmetric_put.peer_rank": float(self.peer_rank),
        }

    def capture_verification_payload(self) -> None:
        if self.local_tensor is None or self.peer_buffer is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        probe = self.local_tensor[: 256 * 256].view(256, 256)
        output = self.peer_buffer[: 256 * 256].view(256, 256).detach().clone()
        self._set_verification_payload(
            inputs={"tensor": probe},
            output=output,
            batch_size=int(probe.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        """Cleanup distributed resources."""
        self.local_tensor = None
        self.peer_buffer = None
        self.handle = None
        if dist.is_initialized():
            dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        """Return memory transfer metrics for SymmetricMemory peer-put."""
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred,
            elapsed_ms=self._last_avg_ms,
            transfer_type="nvlink",  # SymmetricMemory uses direct NVLink access
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark ran successfully."""
        if self.local_tensor is None:
            return "Local tensor not initialized"
        if self.peer_buffer is None:
            return "Peer buffer not initialized"
        if self._last_avg_ms <= 0:
            return "No timing recorded"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.0, 0.0)


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedSymmetricMemoryPerfBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
