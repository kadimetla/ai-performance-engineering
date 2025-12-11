"""Baseline TMEM epilogue (separate matmul + bias + SiLU kernels)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineMatmulTCGen05EpilogueBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: matmul followed by separate bias addition and SiLU activation.
    
    This is the naive approach that requires:
    1. Matmul kernel (write result to global memory)
    2. Bias addition kernel (read from global, write to global)
    3. SiLU activation kernel (read from global, write to global)
    
    The optimized version fuses these into a single kernel with TMEM-resident epilogue.
    """

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda")
        # Match other matmul benchmarks (baseline_matmul.py uses n=8192)
        self.n = 8192
        self.size = self.n
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(bytes_per_iteration=float(self.n * self.n * 2 * 4))

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        dtype = torch.float16
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        self.bias = torch.randn(self.size, device=self.device, dtype=dtype)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.A is not None and self.B is not None and self.bias is not None
        with self._nvtx_range("baseline_matmul_tcgen05_bias_silu"):
            with torch.no_grad():
                # Step 1: Matrix multiplication (uses cuBLAS)
                C = torch.matmul(self.A, self.B)
                # Step 2: Add bias (separate kernel launch)
                C = C + self.bias
                # Step 3: SiLU activation (separate kernel launch)
                self.output = F.silu(C)
        self._synchronize()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B, "bias": self.bias},
            output=self.output.detach().float().clone(),
            batch_size=self.size,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        self.A = None
        self.B = None
        self.bias = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaselineMatmulTCGen05EpilogueBenchmark:
    return BaselineMatmulTCGen05EpilogueBenchmark()
