"""Optimized NCCL path: all-GPU reduction with no CPU round-trips.

Key optimizations vs baseline:
- All tensor operations stay on GPU (no .cpu() calls)
- Uses in-place reduction to avoid allocations
- Larger batch/hidden dims to amortize kernel launch overhead
- Pre-allocated output buffer reused across iterations
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class OptimizedNcclBenchmark(BaseBenchmark):
    """All-GPU reduction path - no CPU round-trips like baseline."""

    def __init__(self):
        super().__init__()
        # Larger workload to make CPU aggregation in baseline visibly expensive
        self.batch_size = 1024
        self.hidden_dim = 4096
        self.inner_dim = 8192
        self.num_shards = 8  # More shards = more CPU overhead in baseline
        
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._reduction_buffer: Optional[torch.Tensor] = None
        
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        # Reduction benchmark: fixed dimensions
        self.jitter_exemption_reason = "NCCL benchmark: fixed dimensions for measurement"

    def setup(self) -> None:
        # Use same seed as baseline for deterministic verification
        torch.manual_seed(42)
        
        # Build model with same architecture as baseline for fair comparison
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, self.hidden_dim),
        ).to(self.device).eval()
        
        self.input = torch.randn(self.batch_size, self.hidden_dim, device=self.device)
        # Pre-allocate output buffer (reused each iteration)
        shard_size = self.batch_size // self.num_shards
        self.output = torch.zeros(shard_size, self.hidden_dim, device=self.device)
        # Pre-allocate reduction buffer
        self._reduction_buffer = torch.zeros(shard_size, self.hidden_dim, device=self.device)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.input is not None and self.model is not None
        
        with nvtx_range("optimized_nccl", enable=enable_nvtx):
            # Forward pass
            with torch.no_grad():
                out = self.model(self.input)
            
            # All-GPU reduction: chunk, reduce in-place, no CPU
            shards = torch.chunk(out, chunks=self.num_shards, dim=0)
            
            # In-place sum reduction (stays on GPU)
            self._reduction_buffer.zero_()
            for shard in shards:
                self._reduction_buffer.add_(shard)
            
            # Average
            self.output.copy_(self._reduction_buffer)
            self.output.div_(self.num_shards)
        
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.input = None
        self.output = None
        self._reduction_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if self.input is None:
            return "Input not initialized"
        if self.output is None:
            return "Output not available"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return {
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "inner_dim": self.inner_dim,
            "num_shards": self.num_shards,
        }

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("Output not available - run benchmark first")
        return self.output

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison.
        
        CPU and GPU reductions may have slight numerical differences due to
        order of operations (floating point is not associative).
        """
        return (1e-4, 1e-4)



def get_benchmark() -> BaseBenchmark:
    return OptimizedNcclBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
