"""Optimized GEMM using a 2:4 structured sparse weight matrix via cuSPARSELt."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.utils.structured_sparsity import prune_2_4


class OptimizedStructuredSparsityBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Sparse GEMM optimized with 2:4 structured sparsity."""

    def __init__(self) -> None:
        super().__init__()
        self.m = 40960
        self.k = 40960
        self.n = 40960
        self.input: Optional[torch.Tensor] = None
        self.input_t: Optional[torch.Tensor] = None
        self.weight: Optional[torch.Tensor] = None
        self.weight_compressed: Optional[torch.Tensor] = None
        self.alg_id: Optional[int] = None
        self.split_k: Optional[int] = None
        self.split_k_mode: Optional[int] = None
        self.output: Optional[torch.Tensor] = None
        tokens = self.m * self.n
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.m),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.m),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if not torch.backends.cusparselt.is_available():
            raise RuntimeError("cuSPARSELt is required for structured sparsity benchmarks")
        self.input = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.input_t = self.input.t().contiguous()
        dense_weight = torch.randn(self.n, self.k, device=self.device, dtype=torch.float16)
        self.weight = prune_2_4(dense_weight)
        self.weight_compressed = torch._cslt_compress(self.weight)
        if not hasattr(torch._C, "_cusparselt"):
            raise RuntimeError("torch._C._cusparselt is required for cuSPARSELt mm_search")
        alg_id, split_k, split_k_mode, _ = torch._C._cusparselt.mm_search(
            self.weight_compressed,
            self.input_t,
            None,
            None,
            None,
            True,
        )
        self.alg_id = int(alg_id)
        self.split_k = int(split_k)
        self.split_k_mode = int(split_k_mode)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if (
            self.input_t is None
            or self.weight_compressed is None
            or self.alg_id is None
            or self.split_k is None
            or self.split_k_mode is None
        ):
            raise RuntimeError("Benchmark not initialized")
        with self._nvtx_range("optimized_structured_sparsity"):
            self.output = torch._cslt_sparse_mm(
                self.weight_compressed,
                self.input_t,
                transpose_result=True,
                alg_id=self.alg_id,
                split_k=self.split_k,
                split_k_mode=self.split_k_mode,
            )
        self._synchronize()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.input is None or self.weight is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        verify_output = self.output[:256, :256]
        self._set_verification_payload(
            inputs={"input": self.input, "weight": self.weight},
            output=verify_output.detach().clone(),
            batch_size=self.input.shape[0],
            parameter_count=self.weight.numel(),
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-2, 1e-2),
            signature_overrides={
                "pruning_enabled": True,
                "sparsity_ratio": 0.5,
            },
        )

    def teardown(self) -> None:
        self.input = None
        self.input_t = None
        self.weight = None
        self.weight_compressed = None
        self.alg_id = None
        self.split_k = None
        self.split_k_mode = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.input is None or self.weight_compressed is None:
            return "Inputs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedStructuredSparsityBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
