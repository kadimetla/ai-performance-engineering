"""Python harness wrapper for optimized_cutlass_gemm.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedCutlassGemmBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUTLASS GEMM driver."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        m = n = k = 1024
        micro_batches = 32
        iterations = 5
        bytes_a = m * k * 4
        bytes_b = k * n * 4
        bytes_c = m * n * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_cutlass_gemm",
            friendly_name="Optimized CUTLASS GEMM",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "M": m,
                "N": n,
                "K": k,
                "micro_batches": micro_batches,
                "iterations": iterations,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(
            bytes_per_iteration=float(bytes_a + bytes_b + bytes_c),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline metrics for cutlass_gemm."""
        return None  # Metrics computed by CUDA binary



def get_benchmark() -> OptimizedCutlassGemmBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedCutlassGemmBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
