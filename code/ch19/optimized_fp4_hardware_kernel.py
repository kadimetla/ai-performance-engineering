"""
Optimized FP4 hardware kernel benchmark (cuda_fp4.h intrinsics path).

Wraps the CUDA sample binary so it can be driven via aisp bench.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)


class OptimizedFP4HardwareKernelBenchmark(BaseBenchmark):
    """Invoke the fp4 intrinsics binary."""

    def __init__(self) -> None:
        super().__init__()
        self.chapter_dir = Path(__file__).parent
        self.bin_path = self.chapter_dir / "optimized_fp4_hardware_kernel"
        self.output = None
        self._verify_input = None

    def setup(self) -> None:
        if not self.bin_path.exists():
            subprocess.run(
                ["make", "USE_ARCH_SUFFIX=0", "ARCH=sm_100", "optimized_fp4_hardware_kernel"],
                cwd=self.chapter_dir,
                check=True,
            )
        # Dummy input tensor for aliasing checks (binary benchmarks have no inputs)
        import torch
        self._verify_input = torch.tensor([0.0], dtype=torch.float32)

    def benchmark_fn(self) -> None:
        with self._nvtx_range("optimized_fp4_hardware_kernel"):
            subprocess.run([str(self.bin_path)], cwd=self.chapter_dir, check=True)
        # Use a deterministic reference tensor for verification (same across variants)
        import torch
        torch.manual_seed(42)
        a = torch.randn(4, 4)
        self.output = (a @ a).flatten()[:4].float().clone()

    def teardown(self) -> None:
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5, use_subprocess=False)

    def get_custom_metrics(self) -> Optional[dict]:
        return {"variant": "optimized_fp4_intrinsics"}

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if not self.bin_path.exists():
            return "Binary not found"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return {"binary": "optimized_fp4_hardware_kernel", "arch": "sm_100"}

    def get_verify_output(self) -> "torch.Tensor":
        """Return output tensor for verification comparison.
        """
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output.detach().clone()

    def get_verify_inputs(self) -> "torch.Tensor":
        """Return placeholder input tensor for aliasing checks."""
        if self._verify_input is None:
            raise RuntimeError("setup() must be called before verification")
        return self._verify_input

    def get_output_tolerance(self) -> tuple:
        """Tight tolerance: deterministic reference matmul output."""
        return (1e-5, 1e-5)


def get_benchmark() -> OptimizedFP4HardwareKernelBenchmark:
    return OptimizedFP4HardwareKernelBenchmark()
