"""optimized_routing_static.py - Dynamic routing optimization."""

from __future__ import annotations

from typing import Optional

import random

import torch
import torch.nn as nn

from core.utils.compile_utils import enable_tf32
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleModel(nn.Module):
    """Simple model with configurable size."""
    
    def __init__(self, hidden_dim=2048, num_layers=24):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, 10)
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


class OptimizedRoutingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Dynamic routing that mixes model sizes."""
    
    def __init__(self):
        super().__init__()
        self.small_model = None
        self.medium_model = None
        self.large_model = None
        self.x_small = None
        self.x_medium = None
        self.x_large = None
        # Match baseline dimensions for fair comparison
        self.batch_size = 16
        self.hidden_dim = 2048
        self.num_layers = 24
        self.routing_order = ["small"] * 5 + ["medium"] * 3 + ["large"] * 2
        self._schedule_index = 0
        tokens = self.batch_size * self.hidden_dim * len(self.routing_order)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(len(self.routing_order)),
            tokens_per_iteration=float(tokens),
        )
        self.output: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(len(self.routing_order)),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        random.seed(42)
        self.small_model = SimpleModel(hidden_dim=1024, num_layers=8).to(self.device).eval()
        self.medium_model = SimpleModel(hidden_dim=1536, num_layers=16).to(self.device).eval()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.large_model = SimpleModel(hidden_dim=2048, num_layers=24).to(self.device).eval()

        if self.device.type == "cuda":
            self.small_model = self.small_model.half()
            self.medium_model = self.medium_model.half()
            self.large_model = self.large_model.half()
        
        dtype_small = next(self.small_model.parameters()).dtype
        dtype_medium = next(self.medium_model.parameters()).dtype
        dtype_large = next(self.large_model.parameters()).dtype

        self.x_small = torch.randn(self.batch_size, 1024, device=self.device, dtype=dtype_small)
        self.x_medium = torch.randn(self.batch_size, 1536, device=self.device, dtype=dtype_medium)
        self.x_large = torch.randn(self.batch_size, 2048, device=self.device, dtype=dtype_large)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._verify_input = torch.randn(self.batch_size, 2048, device=self.device, dtype=dtype_large)
        self.parameter_count = sum(p.numel() for p in self.small_model.parameters()) + \
            sum(p.numel() for p in self.medium_model.parameters()) + \
            sum(p.numel() for p in self.large_model.parameters())
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if any(v is None for v in (self.small_model, self.medium_model, self.large_model, self.x_small, self.x_medium, self.x_large)):
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("routing"):
            with torch.no_grad():
                idx = self._schedule_index
                order_len = len(self.routing_order)
                for _ in range(order_len):
                    tier = self.routing_order[idx]
                    if tier == "small":
                        _ = self.small_model(self.x_small)
                    elif tier == "medium":
                        _ = self.medium_model(self.x_medium)
                    else:
                        _ = self.large_model(self.x_large)
                    idx = (idx + 1) % order_len
                self._schedule_index = idx
            if self._verify_input is not None:
                with torch.no_grad():
                    self.output = self.large_model(self._verify_input).detach().float().clone()
        self._synchronize()
        if self.output is None or self._verify_input is None:
            raise RuntimeError("benchmark_fn() must produce output")
        dtype = self.output.dtype
        self._payload_dtype = dtype

    def capture_verification_payload(self) -> None:
        dtype = self._payload_dtype
        self._set_verification_payload(
            inputs={"verify_input": self._verify_input},
            output=self.output,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1.0, 10.0),
        )

    def teardown(self) -> None:
        self.small_model = None
        self.medium_model = None
        self.large_model = None
        self.x_small = None
        self.x_medium = None
        self.x_large = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
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
        return None


def get_benchmark() -> OptimizedRoutingBenchmark:
    return OptimizedRoutingBenchmark()
