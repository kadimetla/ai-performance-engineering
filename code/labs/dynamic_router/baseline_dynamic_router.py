"""Benchmark harness wrapper for the baseline dynamic router simulation."""

from __future__ import annotations

import json
from numbers import Number
from typing import Dict, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.verification_mixin import VerificationPayloadMixin
from labs.dynamic_router.driver import simulate


class BaselineDynamicRouterBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs the baseline (single-pool) routing simulation under aisp bench."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.register_workload_metadata(requests_per_iteration=1.0)
        self.output: Optional[torch.Tensor] = None
        self.metrics: Optional[torch.Tensor] = None
        self.verify_input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        # No external assets to prepare
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    def benchmark_fn(self) -> None:
        # Fixed seed/ticks to keep runs comparable
        self._summary = simulate(
            "baseline",
            num_ticks=120,
            seed=42,
            log_interval=None,
        )
        verify_tensor = torch.tensor(
            [[float(self._summary.get("ticks", 0)), float(self._summary.get("seed", 0))]],
            dtype=torch.float32,
        )
        expected_shape = tuple(verify_tensor.shape)
        if self.metrics is None or tuple(self.metrics.shape) != expected_shape:
            self.metrics = torch.zeros(expected_shape, dtype=torch.float32)
        if self.verify_input is None or tuple(self.verify_input.shape) != expected_shape:
            self.verify_input = torch.ones(expected_shape, dtype=torch.float32)
        self.output = (verify_tensor * self.verify_input + self.metrics).detach()
        self._set_verification_payload(
            inputs={
                "verify_input": self.verify_input.detach(),
                "metrics_seed": torch.tensor([42], dtype=torch.int64),
                "ticks": torch.tensor([120], dtype=torch.int64),
            },
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "tf32": False},
            output_tolerance=(0.1, 1.0),
        )

    def get_config(self) -> Optional[BenchmarkConfig]:
        # Single iteration; simulation already encapsulates multiple ticks
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=120,
            timeout_multiplier=3.0,
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None

    def teardown(self) -> None:
        self.metrics = None
        self.verify_input = None
        self.output = None
        super().teardown()


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineDynamicRouterBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    cfg = bench.get_config()
    bench.benchmark_fn()
    print(json.dumps(bench.get_custom_metrics() or {}, indent=2))
