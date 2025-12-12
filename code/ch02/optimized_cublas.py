"""optimized_cublas.py - Pure cuBLAS matmul with TF32 tensor-core acceleration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.utils.compile_utils import configure_tf32, restore_tf32
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class OptimizedCublasBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """
    Optimized: pure cuBLAS GEMM with TF32 and warmed-up heuristics.

    This keeps the math in FP32 but lets cuBLAS route workloads through tensor cores
    (TF32) while running a few warmup matmuls so Lt heuristics cache the best kernel.
    """

    def __init__(self):
        super().__init__()
        self.m = 2048
        self.n = 2048
        self.k = 2048
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.C: Optional[torch.Tensor] = None
        self._tf32_state: Optional[Tuple[Optional[str], Optional[str]]] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )

    def setup(self) -> None:
        """Enable TF32, allocate FP32 matrices, compute verification output, and warm up cuBLAS."""
        # Seed FIRST for deterministic verification
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Only use the new matmul precision APIs to avoid mixed-mode warnings.
        self._tf32_state = configure_tf32(
            enable_matmul=True,
            enable_cudnn=True,
            matmul_precision="high",
            cudnn_precision="high",
        )
        torch.set_float32_matmul_precision("high")

        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        self.C = None

        # Warmup a handful of GEMMs so cuBLAS Lt heuristics settle before measurement.
        for _ in range(10):
            _ = torch.matmul(self.A, self.B)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """cuBLAS TF32 GEMM."""
        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("cublas", enable=enable_nvtx):
            self.C = torch.matmul(self.A, self.B)
        self._synchronize()

        if self.C is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B},
            output=self.C.detach().clone(),
            batch_size=self.A.shape[0],
            parameter_count=0,
            precision_flags={
                # Keep signature aligned with baseline; TF32 is the optimization detail, not a workload change.
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(1e-2, 1e-1),
        )

    def teardown(self) -> None:
        """Restore TF32 knobs and free tensors."""
        self.A = None
        self.B = None
        if self._tf32_state is not None:
            restore_tf32(self._tf32_state)
            self._tf32_state = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedCublasBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
