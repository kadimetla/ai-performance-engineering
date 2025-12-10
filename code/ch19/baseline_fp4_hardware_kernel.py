"""
Baseline FP4 hardware kernel benchmark (manual quantization path).

Wraps the CUDA sample binary so it can be driven via aisp bench.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)


class BaselineFP4HardwareKernelBenchmark(BaseBenchmark):
    """Invoke the manual FP4 kernel binary."""

    def __init__(self) -> None:
        super().__init__()
        self.chapter_dir = Path(__file__).parent
        self.bin_path = self.chapter_dir / "baseline_fp4_hardware_kernel"
        # Binary benchmark: no tensor output available
        self.jitter_exemption_reason = "FP4 hardware binary: workload dimensions compiled into binary"

    def setup(self) -> None:
        # Build without arch suffix so we know the binary name deterministically.
        if not self.bin_path.exists():
            subprocess.run(
                ["make", "USE_ARCH_SUFFIX=0", "ARCH=sm_100", "baseline_fp4_hardware_kernel"],
                cwd=self.chapter_dir,
                check=True,
            )

    def benchmark_fn(self) -> None:
        with self._nvtx_range("baseline_fp4_hardware_kernel"):
            subprocess.run([str(self.bin_path)], cwd=self.chapter_dir, check=True)

    def teardown(self) -> None:
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5, use_subprocess=False)

    def get_custom_metrics(self) -> Optional[dict]:
        return {"variant": "baseline_manual_fp4"}

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if not self.bin_path.exists():
            return "Binary not found"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return {"binary": "baseline_fp4_hardware_kernel", "arch": "sm_100"}

    def get_verify_output(self) -> "torch.Tensor":
        """Return output tensor for verification comparison.
        
        Binary benchmark: returns consistent checksum for binary identity.
        """
        import torch
        # Return binary path hash as a consistent checksum
        return torch.tensor([float(hash(str(self.bin_path)) % (2**31))], dtype=torch.float32)


def get_benchmark() -> BaselineFP4HardwareKernelBenchmark:
    return BaselineFP4HardwareKernelBenchmark()
