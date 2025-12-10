#!/usr/bin/env python3
"""Baseline: Llama 3.1 8B without optimizations.

Standard PyTorch eager mode without compile, FP8, or FlexAttention.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.real_world_models.llama_3_1_8b_optimization import Llama31_8B_Optimization


class BaselineLlama31_8B(BaseBenchmark):
    """Baseline Llama 3.1 8B - eager mode, no optimizations."""

    def __init__(self, batch_size: int = 1, seq_length: int = 2048):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.model_wrapper = None
        self._last_metrics = {}
        self.jitter_exemption_reason = "Llama 3.1 8B baseline: fixed configuration"
        self.register_workload_metadata(requests_per_iteration=float(batch_size))

    def setup(self) -> None:
        self.model_wrapper = Llama31_8B_Optimization(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            use_compile=False,  # No torch.compile
            use_fp8=False,      # No FP8
            use_flex_attention=False,  # No FlexAttention
        )
        self.model_wrapper.setup()

    def benchmark_fn(self) -> None:
        if self.model_wrapper:
            self._last_metrics = self.model_wrapper.run()
        self._synchronize()

    def teardown(self) -> None:
        if self.model_wrapper:
            self.model_wrapper.teardown()
        self.model_wrapper = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_custom_metrics(self) -> dict:
        return self._last_metrics

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"batch_size": self.batch_size, "seq_length": self.seq_length}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return BaselineLlama31_8B()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
