#!/usr/bin/env python3
"""Baseline: Simple MoE routing without topology awareness.

Basic MoE routing that doesn't consider GPU topology or NVLink locality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from core.utils.logger import get_logger
from ch15.verification_payload_mixin import VerificationPayloadMixin

logger = get_logger(__name__)


class BaselineMoERoutingSimple:
    """Baseline MoE routing without topology awareness."""
    
    def __init__(
        self,
        batch_size: int = 16,
        seq_length: int = 2048,
        hidden_size: int = 4096,
        num_experts: int = 64,
        top_k: int = 2,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_output = None  # For verification
        
        logger.info(f"Baseline MoE Routing")
        logger.info(f"  Experts: {num_experts}, Top-K: {top_k}")
    
    def setup(self):
        """Initialize router."""
        # Simple linear router
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False).to(self.device)
        
        # Create input
        self.input = torch.randn(
            self.batch_size,
            self.seq_length,
            self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16
        )
        
        logger.info("Router initialized")
    
    def run(self) -> Dict[str, float]:
        """Execute baseline routing."""
        import time
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Compute routing scores
        routing_logits = self.router(self.input)  # [batch, seq, num_experts]
        
        # Top-K selection (no topology consideration)
        routing_weights, selected_experts = torch.topk(routing_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Capture output for verification
        self.last_output = routing_weights.detach()
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Calculate load imbalance
        expert_counts = torch.bincount(
            selected_experts.view(-1),
            minlength=self.num_experts
        ).float()
        
        ideal_count = expert_counts.sum() / self.num_experts
        load_variance = torch.var(expert_counts).item()
        max_imbalance = (expert_counts.max() / ideal_count).item()
        
        logger.info(f"Load variance: {load_variance:.2f}")
        logger.info(f"Max imbalance: {max_imbalance:.2f}×")
        
        return {
            "latency_ms": elapsed * 1000,
            "load_variance": load_variance,
            "max_imbalance": max_imbalance,
            "topology_aware": False,
        }
    
    def cleanup(self):
        """Clean up."""
        del self.router, self.input
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 16,
    seq_length: int = 2048,
    hidden_size: int = 4096,
    num_experts: int = 64,
    top_k: int = 2,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline MoE routing benchmark."""
    
    benchmark = BaselineMoERoutingSimple(
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(iterations=10, warmup=5, profile_mode=profile)
    harness = BenchmarkHarness(mode=BenchmarkMode.INFERENCE, config=config)
    
    result = harness.benchmark(benchmark.run, name="baseline_moe_routing")
    
    metrics = benchmark.run()
    benchmark.cleanup()
    
    return {"mean_time_ms": result.timing.mean_ms, **metrics}


class _MoERoutingSimpleBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Wrapper benchmark for simple MoE routing."""

    def __init__(self) -> None:
        super().__init__()
        self._impl = BaselineMoERoutingSimple()
        self._metrics = {}
        self.output = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        self._impl.setup()
        probe = torch.zeros(1, device=self.device, dtype=torch.bfloat16)
        param_count = sum(p.numel() for p in self._impl.router.parameters()) if hasattr(self._impl, "router") else 0
        self._set_verification_payload(
            inputs={"probe": probe},
            output=torch.zeros(1, device=self.device, dtype=torch.bfloat16),
            batch_size=1,
            parameter_count=param_count,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def benchmark_fn(self) -> None:
        self._metrics = self._impl.run()
        self.output = self._impl.last_output
        self._synchronize()

    def teardown(self) -> None:
        self._impl.cleanup()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        return super().get_output_tolerance()


def get_benchmark() -> BaseBenchmark:
    return _MoERoutingSimpleBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
