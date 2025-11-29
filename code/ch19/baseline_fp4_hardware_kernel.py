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

    def skip_output_verification(self) -> bool:
        # Binary prints its own stats; we don't collect outputs for comparison.
        return True

    def get_custom_metrics(self) -> Optional[dict]:
        return {"variant": "baseline_manual_fp4"}


def get_benchmark() -> BaselineFP4HardwareKernelBenchmark:
    return BaselineFP4HardwareKernelBenchmark()
