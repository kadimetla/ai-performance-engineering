"""Baseline dense GEMM for a max-size 2:4-pruned weight matrix."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.utils.structured_sparsity import prune_2_4


class BaselineStructuredSparsityMaxBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Dense GEMM baseline for a max-size 2:4-pruned weight matrix."""

    def __init__(self) -> None:
        super().__init__()
        self.m = 49152
        self.k = 49152
        self.n = 49152
        self.input: Optional[torch.Tensor] = None
        self.weight: Optional[torch.Tensor] = None
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
        self.input = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        dense_weight = torch.randn(self.n, self.k, device=self.device, dtype=torch.float16)
        self.weight = prune_2_4(dense_weight)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.input is None or self.weight is None:
            raise RuntimeError("Benchmark not initialized")
        with self._nvtx_range("baseline_structured_sparsity_max"):
            self.output = F.linear(self.input, self.weight)
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
        self.weight = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.input is None or self.weight is None:
            return "Inputs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineStructuredSparsityMaxBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
