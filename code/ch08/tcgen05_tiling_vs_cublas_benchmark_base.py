"""Shared base for tcgen05 tiling vs cuBLAS matmul benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class Tcgen05TilingVsCublasBase(VerificationPayloadMixin, BaseBenchmark):
    """Common setup and verification for tcgen05 tiling vs cuBLAS matmuls."""

    nvtx_label = "tcgen05_tiling_vs_cublas"
    matrix_rows = 8192
    matrix_cols = 8192
    shared_dim = 512
    tensor_dtype = torch.float16
    output_tolerance = (0.1, 1.0)

    def __init__(self) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for ch08 tcgen05 tiling vs cuBLAS benchmark")
        self.device = torch.device("cuda")
        self.matrix_a: Optional[torch.Tensor] = None
        self.matrix_b: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.parameter_count = 0
        bytes_per_iter = (
            (self.matrix_rows * self.shared_dim)
            + (self.shared_dim * self.matrix_cols)
            + (self.matrix_rows * self.matrix_cols)
        ) * 2.0
        self.register_workload_metadata(bytes_per_iteration=float(bytes_per_iter))

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.matrix_a = torch.randn(
            self.matrix_rows,
            self.shared_dim,
            device=self.device,
            dtype=self.tensor_dtype,
        ).contiguous()
        self.matrix_b = torch.randn(
            self.shared_dim,
            self.matrix_cols,
            device=self.device,
            dtype=self.tensor_dtype,
        ).contiguous()
        self.output = None
        self.parameter_count = int(self.matrix_a.numel() + self.matrix_b.numel())
        self._synchronize()

    def capture_verification_payload(self) -> None:
        if self.matrix_a is None or self.matrix_b is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"matrix_a": self.matrix_a, "matrix_b": self.matrix_b},
            output=self.output.detach().clone(),
            batch_size=self.matrix_rows,
            parameter_count=int(self.parameter_count),
            precision_flags={
                "fp16": self.tensor_dtype == torch.float16,
                "bf16": self.tensor_dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=self.output_tolerance,
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        return None

    def teardown(self) -> None:
        self.matrix_a = None
        self.matrix_b = None
        self.output = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=48,
            warmup=8,
            timing_method="wall_clock",
        )
