"""Optimized AI example: fuse blocks into a single inference stack without CPU sync."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.utils.compile_utils import enable_tf32


class TinyBlock(nn.Module):
    """Same architecture as baseline for fair comparison."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class OptimizedAIBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Chains the tiny blocks without CPU sync between them (the optimization)."""

    def __init__(self):
        super().__init__()
        self.blocks: Optional[nn.ModuleList] = None
        self.static_input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.batch = 512
        self.hidden = 1024
        # Inference benchmark - jitter check not applicable
        tokens = self.batch * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        enable_tf32()
        # Initialize model weights after seeding for deterministic comparison
        self.blocks = nn.ModuleList(TinyBlock(1024).to(self.device) for _ in range(4))
        for block in self.blocks:
            block.eval()
        # Use same dtype as baseline (float32)
        self.static_input = torch.randn(self.batch, self.hidden, device=self.device, dtype=torch.float32)
        # Warmup
        with torch.inference_mode():
            out = self.static_input
            for block in self.blocks:
                out = block(out)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.blocks is not None and self.static_input is not None
        with self._nvtx_range("optimized_ai"):
            # The optimization: no CPU sync between blocks
            out = self.static_input
            for block in self.blocks:
                out = block(out)
            self._synchronize()
        self.output = out.detach()

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"inputs": self.static_input},
            output=self.output,
            batch_size=self.batch,
            parameter_count=sum(p.numel() for p in self.blocks.parameters()),
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.static_input = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=40, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_storage_io_metrics
        return compute_storage_io_metrics(
            bytes_read=getattr(self, '_bytes_read', 0.0),
            bytes_written=getattr(self, '_bytes_written', 0.0),
            read_time_ms=getattr(self, '_read_time_ms', 1.0),
            write_time_ms=getattr(self, '_write_time_ms', 1.0),
        )

    def validate_result(self) -> Optional[str]:
        if self.blocks is None or self.static_input is None:
            return "Model/input not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    return OptimizedAIBenchmark()
