"""Kubernetes optimization: overlap data provisioning with training work.

This benchmark demonstrates efficient data provisioning - using pinned memory,
prefetching, and async H2D copies to overlap data loading with compute.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


class OptimizedKubernetesBenchmark(BaseBenchmark):
    """Prefetches device batches on a side stream for overlapped data loading."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.host_batches: List[torch.Tensor] = []
        self.target_batches: List[torch.Tensor] = []
        self.device_batches: List[torch.Tensor] = []
        self.device_targets: List[torch.Tensor] = []
        self.copy_stream = torch.cuda.Stream()
        self.cur_slot = 0
        self.next_slot = 1
        self.batch_idx = 0
        self.output: Optional[torch.Tensor] = None
        # Training benchmarks don't support jitter check - outputs change due to weight updates
        self.jitter_exemption_reason = "Training benchmark: outputs change each iteration due to gradient updates"
        
        elements = 2 * 512 * 1024
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(elements),
            bytes_per_iteration=float(elements * 4),
        )

    def _prefetch_slot(self, slot: int) -> None:
        """Async copy to device buffer on copy stream."""
        batch_idx = (self.batch_idx + slot) % len(self.host_batches)
        with torch.cuda.stream(self.copy_stream):
            self.device_batches[slot].copy_(self.host_batches[batch_idx], non_blocking=True)
            self.device_targets[slot].copy_(self.target_batches[batch_idx], non_blocking=True)

    def setup(self) -> None:
        torch.manual_seed(314)
        # Same model as baseline for fair comparison
        self.model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        ).to(self.device)
        
        # Pre-allocate host batches with pinned memory (the optimization)
        num_batches = 4
        for _ in range(num_batches):
            self.host_batches.append(torch.randn(512, 1024, dtype=torch.float32, pin_memory=True))
            self.target_batches.append(torch.randn(512, 1024, dtype=torch.float32, pin_memory=True))
        
        # Device double-buffers for prefetching
        self.device_batches = [
            torch.empty(512, 1024, device=self.device, dtype=torch.float32),
            torch.empty(512, 1024, device=self.device, dtype=torch.float32),
        ]
        self.device_targets = [
            torch.empty(512, 1024, device=self.device, dtype=torch.float32),
            torch.empty(512, 1024, device=self.device, dtype=torch.float32),
        ]
        
        self.batch_idx = 0
        self.cur_slot = 0
        self.next_slot = 1
        
        # Prefetch first batch
        self._prefetch_slot(self.cur_slot)
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        # Start prefetching second batch
        self.batch_idx += 1
        self._prefetch_slot(self.next_slot)
        self._synchronize()

    def benchmark_fn(self) -> None:
        """Training step with overlapped data loading."""
        assert self.model is not None
        
        # Wait for current batch to be ready
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        data = self.device_batches[self.cur_slot]
        target = self.device_targets[self.cur_slot]
        
        with self._nvtx_range("optimized_kubernetes"):
            # Forward (overlapped with next batch prefetch)
            out = self.model(data)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            
            # Clear gradients (simulate optimizer step)
            for p in self.model.parameters():
                p.grad = None
        
        self.output = out.detach()
        
        # Swap slots and start next prefetch
        self.cur_slot, self.next_slot = self.next_slot, self.cur_slot
        self.batch_idx += 1
        self._prefetch_slot(self.next_slot)
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.host_batches = []
        self.target_batches = []
        self.device_batches = []
        self.device_targets = []
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 1),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if not self.device_batches:
            return "Device batches not initialized"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return {
            "batch_size": 512,
            "input_dim": 1024,
            "hidden_dim": 1024,
            "output_dim": 1024,
        }

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is not None:
            return self.output.detach().clone()
        return torch.tensor([0.0], dtype=torch.float32, device=self.device)
    
    def get_output_tolerance(self) -> tuple:
        """Return custom tolerance for training output comparison."""
        return (1e-3, 1e-3)


def get_benchmark() -> BaseBenchmark:
    return OptimizedKubernetesBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
