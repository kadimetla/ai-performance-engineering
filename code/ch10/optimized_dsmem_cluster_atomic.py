"""Python harness wrapper for optimized_dsmem_cluster_atomic.cu.

DSMEM Cluster Atomic: Uses map_shared_rank() + atomicAdd for cross-CTA aggregation.

BOOK REFERENCE (Ch10): DSMEM (Distributed Shared Memory) allows CTAs within
a cluster to communicate through shared memory without global memory round-trips.

Key pattern:
  1. Each CTA performs block-level reduction
  2. Each CTA atomically adds its result to the cluster leader's smem via DSMEM
  3. Cluster leader writes final result to global memory

This is faster than two-pass reduction because:
  - No intermediate global memory writes between passes
  - Cluster sync is cheaper than kernel launch overhead
"""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class OptimizedDSMEMClusterAtomicBenchmark(CudaBinaryBenchmark):
    """Wraps DSMEM cluster reduction using atomic aggregation."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_dsmem_cluster_atomic",
            friendly_name="DSMEM Cluster Reduction (Atomic Aggregation)",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "batch_size": 1024,
                "dtype": "float32",
                "N": 16 * 1024 * 1024,
                "cluster_size": 4,
                "block_elems": 4096,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=1024 * 1024)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def get_input_signature(self) -> dict:
        """Signature for DSMEM cluster atomic reduction."""
        return simple_signature(
            batch_size=1024,
            dtype="float32",
            N=16 * 1024 * 1024,
            cluster_size=4,
            block_elems=4096,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedDSMEMClusterAtomicBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
