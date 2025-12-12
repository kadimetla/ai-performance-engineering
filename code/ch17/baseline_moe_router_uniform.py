"""baseline_moe_router_uniform.py

Topology-agnostic MoE router: selects experts uniformly at random without
respecting NVLink/NVSwitch locality. Serves as a baseline for the topology-
aware router.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineMoERouterUniformBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Uniform expert selection ignoring fabric distance."""

    def __init__(self):
        super().__init__()
        self.num_experts = 16
        self.tokens = 4096
        self.experts: List[int] = list(range(self.num_experts))
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.tokens),
            tokens_per_iteration=float(self.tokens),
        )
        self._last_assignment: Dict[int, int] = {}
        self.output: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verification_payload = None

    def setup(self) -> None:
        torch.manual_seed(42)
        random.seed(42)
        self._verify_input = torch.tensor(self.tokens, device=self.device, dtype=torch.int64)
        self._synchronize()

    def benchmark_fn(self) -> None:
        with self._nvtx_range("baseline_moe_router_uniform"):
            assignment: Dict[int, int] = {}
            for token_id in range(self.tokens):
                expert = random.choice(self.experts)
                assignment[token_id] = expert
            self._last_assignment = assignment
            self.output = torch.tensor(
                [assignment[t] for t in range(self.tokens)],
                device=self.device,
                dtype=torch.int32,
            )
            self._synchronize()

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"tokens": self._verify_input},
            output=self.output,
            batch_size=self.tokens,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self._last_assignment = {}

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        if not self._last_assignment:
            return "No assignments produced"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineMoERouterUniformBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(BaselineMoERouterUniformBenchmark)
