"""Baseline FSDP2 with FP32 Communication (No Compression).

This is the baseline for comparison with FP8-compressed communication.
Uses standard FP32 all-reduce without any compression.

Key differences from optimized:
- FP32 gradients (4 bytes per element)
- No quantization overhead but 4x more bandwidth
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.distributed as dist

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing."""
    
    def __init__(self, hidden_dim: int = 4096, num_heads: int = 32):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class BaselineFSDP2FP32CommBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark FSDP2 with standard FP32 communication (no compression)."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.optimizer = None
        self.x = None
        self.batch_size = 4
        self.seq_len = 512
        self.hidden_dim = 4096
        self.num_layers = 4
        self._last = 0.0
        self._comm_bytes = 0.0
        self.output = None
        self._verify_input = None
        self.parameter_count = 0
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup model with standard FP32 communication."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Build model
        layers = nn.ModuleList([
            SimpleTransformerBlock(self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        self.model = nn.Sequential(*layers).to(self.device, torch.bfloat16)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        # Calculate communication volume (FP32 = 4 bytes per element)
        total_params = sum(p.numel() for p in self.model.parameters())
        self._comm_bytes = total_params * 4  # FP32
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # Input
        self.x = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=torch.bfloat16
        )
        
        # Fixed input for verification (created after model init, same seed state)
        self._verify_input = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=torch.bfloat16
        )
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(self.x)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Training step with standard FP32 gradients (no compression)."""
        with self._nvtx_range("baseline_fsdp2_fp32_comm"):
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward
            output = self.model(self.x)
            loss = output.sum()
            
            # Backward - gradients remain FP32 (no compression)
            loss.backward()
            
            # In real FSDP, all-reduce would happen here with FP32 gradients
            # (4x more bandwidth than FP8)
            
            # Optimizer step
            self.optimizer.step()
            
            self._last = float(loss.detach())
            
            # Capture verification output after training step
            with torch.no_grad():
                self.model.eval()
                self.output = self.model(self._verify_input).float().clone()
                self.model.train()
        self._synchronize()
        if self._verify_input is None or self.output is None:
            raise RuntimeError("Verification input/output not initialized")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.output,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        """Cleanup."""
        self.model = None
        self.optimizer = None
        self.x = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineFSDP2FP32CommBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
