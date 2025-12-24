"""Python harness wrapper for baseline_fp4_hardware_kernel.cu (Chapter 19).

This benchmark participates in the harness as a *comparable* baseline for the
FP4 hardware-kernel example.

Pairs with: optimized_fp4_hardware_kernel.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature
from core.harness.benchmark_harness import BaseBenchmark

_M = 1024
_N = 1024
_K = 1024


class BaselineFP4HardwareKernelBenchmark(CudaBinaryBenchmark):
    """Runs baseline_fp4_hardware_kernel.cu (manual NVFP4 GEMM)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_fp4_hardware_kernel",
            friendly_name="Baseline FP4 Hardware Kernel (Manual GEMM)",
            iterations=1,
            warmup=1,
            timeout_seconds=300,
            run_args=[str(_M), str(_N), str(_K)],
            workload_params={
                "batch_size": 1,
                "dtype": "nvfp4_e2m1",
                "M": _M,
                "N": _N,
                "K": _K,
            },
        )
        self.register_workload_metadata(requests_per_iteration=1.0)

    def get_custom_metrics(self) -> Optional[dict]:
        if self.last_time_ms is None:
            raise RuntimeError("Benchmark did not capture TIME_MS output")
        flops = 2.0 * _M * _N * _K
        tflops = flops / (float(self.last_time_ms) * 1e9)
        return {"tflops": float(tflops), "variant": "baseline_manual_fp4"}

    def get_input_signature(self) -> dict:
        return simple_signature(batch_size=1, dtype="nvfp4_e2m1", M=_M, N=_N, K=_K).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        # Verification compares scalar checksums derived from FP16 output. Allow small
        # numerical drift between the manual kernel and cuBLASLt NVFP4 tensor cores.
        return (1e-2, 1e-2)


def get_benchmark() -> BaseBenchmark:
    return BaselineFP4HardwareKernelBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
