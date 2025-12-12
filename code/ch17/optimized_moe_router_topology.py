"""optimized_moe_router_topology.py

Topology-aware MoE router: groups experts by NVSwitch island and routes tokens
to local experts first, falling back to nearby islands only when local capacity
is exhausted. Uses simple round-robin within each island.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedMoERouterTopologyBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Fabric-aware expert routing with island preference."""

    def __init__(self):
        super().__init__()
        self.islands: Dict[int, List[int]] = {}
        self.capacity_per_expert = 256
        self.tokens = 4096
        self._last_assignment: Dict[int, int] = {}
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.tokens),
            tokens_per_iteration=float(self.tokens),
        )
        self.output: Optional[torch.Tensor] = None
        self._verify_tokens: Optional[torch.Tensor] = None
        self._verification_payload = None

    def setup(self) -> None:
        # 4 islands × 4 experts each
        expert_id = 0
        for island in range(4):
            self.islands[island] = []
            for _ in range(4):
                self.islands[island].append(expert_id)
                expert_id += 1
        self._verify_tokens = torch.tensor(self.tokens, device=self.device)
        self._synchronize()

    def _route(self, token_id: int, local_island: int, loads: Dict[int, int]) -> int:
        """Pick an expert preferring the local island, then nearest neighbor."""
        # Local island first
        for exp in self.islands[local_island]:
            if loads[exp] < self.capacity_per_expert:
                loads[exp] += 1
                return exp

        # Fall back to nearest other island (simple linear distance)
        island_ids = sorted(self.islands.keys(), key=lambda i: abs(i - local_island))
        for isl in island_ids:
            for exp in self.islands[isl]:
                if loads[exp] < self.capacity_per_expert:
                    loads[exp] += 1
                    return exp
        # If everything is full, just return the first expert
        fallback = self.islands[local_island][0]
        loads[fallback] += 1
        return fallback

    def benchmark_fn(self) -> None:
        with self._nvtx_range("optimized_moe_router_topology"):
            assignment: Dict[int, int] = {}
            loads: Dict[int, int] = {exp: 0 for experts in self.islands.values() for exp in experts}
            for token in range(self.tokens):
                local_island = token % len(self.islands)  # cheap locality hint
                expert = self._route(token, local_island, loads)
                assignment[token] = expert
            self._last_assignment = assignment
            self.output = torch.tensor(
                [assignment[t] for t in range(self.tokens)],
                device=self.device,
                dtype=torch.int32,
            )

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_tokens is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"tokens": self._verify_tokens},
            output=self.output,
            batch_size=int(self.tokens),
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
    return OptimizedMoERouterTopologyBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(OptimizedMoERouterTopologyBenchmark)
