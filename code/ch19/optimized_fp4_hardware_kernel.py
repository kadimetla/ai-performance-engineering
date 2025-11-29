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

    def setup(self) -> None:
        if not self.bin_path.exists():
            subprocess.run(
                ["make", "USE_ARCH_SUFFIX=0", "ARCH=sm_100", "optimized_fp4_hardware_kernel"],
                cwd=self.chapter_dir,
                check=True,
            )

    def benchmark_fn(self) -> None:
        with self._nvtx_range("optimized_fp4_hardware_kernel"):
            subprocess.run([str(self.bin_path)], cwd=self.chapter_dir, check=True)

    def teardown(self) -> None:
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5, use_subprocess=False)

    def skip_output_verification(self) -> bool:
        return True

    def get_custom_metrics(self) -> Optional[dict]:
        return {"variant": "optimized_fp4_intrinsics"}


def get_benchmark() -> OptimizedFP4HardwareKernelBenchmark:
    return OptimizedFP4HardwareKernelBenchmark()
