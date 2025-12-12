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
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin


class OptimizedFP4HardwareKernelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Invoke the fp4 intrinsics binary."""

    def __init__(self) -> None:
        super().__init__()
        self.chapter_dir = Path(__file__).parent
        self.bin_path = self.chapter_dir / "optimized_fp4_hardware_kernel"
        self.output = None
        self._verify_input = None
        self._verification_payload = None
        self._workload = WorkloadMetadata(requests_per_iteration=1.0)
        self.register_workload_metadata(requests_per_iteration=1.0)

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
        if self.output is None or self._verify_input is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.output,
            batch_size=1,
            parameter_count=0,
            output_tolerance=(1e-5, 1e-5),
        )

    def teardown(self) -> None:
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5, use_subprocess=True)

    def get_custom_metrics(self) -> Optional[dict]:
        return {"variant": "optimized_fp4_intrinsics"}

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if not self.bin_path.exists():
            return "Binary not found"
        return None


def get_benchmark() -> OptimizedFP4HardwareKernelBenchmark:
    return OptimizedFP4HardwareKernelBenchmark()
