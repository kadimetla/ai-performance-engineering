"""Benchmark harness wrapper for the topology probe (GPU↔NUMA mapping)."""

from __future__ import annotations

import json
from numbers import Number
from typing import Dict, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.verification_mixin import VerificationPayloadMixin
from labs.dynamic_router.topology_probe import TopologyProbeBenchmark


class BaselineTopologyProbeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs the topology probe under aisp bench."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.metrics: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.verify_input: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    def benchmark_fn(self) -> None:
        bench = TopologyProbeBenchmark()
        bench.benchmark_fn()
        self._summary = bench.get_custom_metrics() or {}
        metric_values = [float(v) for v in self._summary.values() if isinstance(v, Number)]
        if not metric_values:
            metric_values = [0.0]
        summary_tensor = torch.tensor(metric_values, dtype=torch.float32).unsqueeze(0)
        expected_shape = tuple(summary_tensor.shape)
        if self.metrics is None or tuple(self.metrics.shape) != expected_shape:
            self.metrics = torch.zeros(expected_shape, dtype=torch.float32)
        if self.verify_input is None or tuple(self.verify_input.shape) != expected_shape:
            self.verify_input = torch.ones(expected_shape, dtype=torch.float32)
        self.output = (summary_tensor * self.verify_input + self.metrics).detach()

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={
                "verify_input": self.verify_input.detach(),
                "num_gpus": torch.tensor([len(self._summary)], dtype=torch.int64),
            },
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "tf32": False},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.metrics = None
        self.verify_input = None
        self.output = None
        super().teardown()

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None


def get_benchmark() -> BaseBenchmark:
    return BaselineTopologyProbeBenchmark()


if __name__ == "__main__":
    b = get_benchmark()
    b.benchmark_fn()
    print(json.dumps(b.get_custom_metrics() or {}, indent=2))
