"""NUMA-aware optimization: pinned memory + async copies overlapped with compute.

This benchmark demonstrates efficient memory transfer - using pinned memory
with async copies and double-buffering to overlap data transfer with compute.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


class OptimizedNUMAAwareBenchmark(BaseBenchmark):
    """Uses pinned host memory and overlaps copies with reduction kernels."""

    def __init__(self):
        super().__init__()
        self.host_tensor: Optional[torch.Tensor] = None
        self.device_buffers: list[torch.Tensor] = []
        self.copy_stream = torch.cuda.Stream()
        self.cur_slot = 0
        self.next_slot = 1
        self.output: Optional[torch.Tensor] = None
        # Memory copy benchmark - jitter check not applicable
        self.jitter_exemption_reason = "Memory copy benchmark: input is fixed-size buffer"
        bytes_per_iter = 128_000_000 * 4  # float32 bytes (same as baseline)
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter),
        )

    def setup(self) -> None:
        torch.manual_seed(9)
        # Pinned memory for efficient async H2D transfer (the optimization)
        self.host_tensor = torch.randn(128_000_000, dtype=torch.float32, pin_memory=True)
        # Double-buffering for overlapping copy with compute
        self.device_buffers = [
            torch.empty_like(self.host_tensor, device=self.device),
            torch.empty_like(self.host_tensor, device=self.device),
        ]
        self.cur_slot = 0
        self.next_slot = 1
        # Prefetch first buffer
        self._start_copy(self.cur_slot)
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        # Start prefetching second buffer
        self._start_copy(self.next_slot)

    def _start_copy(self, slot: int) -> None:
        assert self.host_tensor is not None
        with torch.cuda.stream(self.copy_stream):
            self.device_buffers[slot].copy_(self.host_tensor, non_blocking=True)

    def benchmark_fn(self) -> None:
        """Copy data and compute sum - async copy overlapped with compute."""
        assert self.host_tensor is not None
        # Wait for current buffer to be ready
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        with self._nvtx_range("optimized_numa"):
            # Compute on ready buffer while next is being copied
            self.output = torch.sum(self.device_buffers[self.cur_slot]).unsqueeze(0)
        # Start copy for current slot (will be ready next iteration)
        self._start_copy(self.cur_slot)
        # Swap slots
        self.cur_slot, self.next_slot = self.next_slot, self.cur_slot
        self._synchronize()

    def teardown(self) -> None:
        self.host_tensor = None
        self.device_buffers = []
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=15, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 1),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if self.host_tensor is None:
            return "Host tensor not initialized"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return {
            "buffer_size": 128_000_000,
            "dtype": "float32",
        }

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is not None:
            return self.output.detach().clone()
        return torch.tensor([0.0], dtype=torch.float32, device=self.device)
    
    def get_output_tolerance(self) -> tuple:
        """Return custom tolerance for sum output comparison."""
        return (1e-3, 1e-3)


def get_benchmark() -> BaseBenchmark:
    return OptimizedNUMAAwareBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
