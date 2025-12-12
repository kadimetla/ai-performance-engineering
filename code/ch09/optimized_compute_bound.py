"""optimized_compute_bound.py - Optimized compute-bound kernel (same math as baseline)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class OptimizedComputeBoundBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Compute-bound kernel - uses torch.compile to fuse the same math as baseline."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.repeats = 16
        self.N = 4096
        tokens = self.N * self.repeats
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize compiled model and inputs."""
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(self.N, self.N * 2),
            nn.ReLU(),
            nn.Linear(self.N * 2, self.N),
        ).to(self.device, dtype=torch.float16).eval()
        # Use torch.compile to fuse and cut Python overhead for repeated matmuls.
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True, dynamic=False)
        except Exception:
            # Fail-fast: if compilation unsupported (e.g., env mismatch), keep eager to preserve correctness.
            self.model = self.model
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: repeated matmul chain identical to baseline."""
        if self.model is None or self.input is None:
            raise RuntimeError("Model/input not initialized")
        out = self.input
        with self._nvtx_range("optimized_compute_bound"):
            for _ in range(self.repeats):
                out = self.model(out)
            self.output = out
        self._synchronize()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output.detach().clone(),
            batch_size=self.input.shape[0],
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-1, 1e-1),
        )
    
    def teardown(self) -> None:
        self.model = None
        self.input = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return compute-bound analysis metrics using the centralized helper."""
        from core.benchmark.metrics import compute_roofline_metrics
        # Same FLOP/byte estimates as baseline model.
        layer1_flops = 2 * self.N * (self.N * 2) * self.N
        layer2_flops = 2 * self.N * self.N * (self.N * 2)
        total_flops = (layer1_flops + layer2_flops) * self.repeats
        element_size = 2  # FP16
        total_bytes = (self.N + self.N) * element_size * self.repeats
        return compute_roofline_metrics(
            total_flops=total_flops,
            total_bytes=total_bytes,
            elapsed_ms=getattr(self, "_last_elapsed_ms", 1.0),
            precision="fp16",
        )
    
    def validate_result(self) -> Optional[str]:
        if self.input is None or self.model is None:
            return "Model/input not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedComputeBoundBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
