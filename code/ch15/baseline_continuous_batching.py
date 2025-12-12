"""baseline_continuous_batching.py - Baseline static batching in disaggregated inference context."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch15.verification_payload_mixin import VerificationPayloadMixin


class BaselineContinuousBatchingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: static batching, sequential processing."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.batches: Optional[list[torch.Tensor]] = None
        self.samples: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self.batch_size = 12
        self.hidden_dim = 1024
        self.num_batches = 12
        self.num_samples = self.batch_size * self.num_batches  # 144 total samples for signature matching
        tokens = self.batch_size * self.hidden_dim * self.num_batches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_batches),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.num_batches),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize model and static batches."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ).to(self.device).eval()

        # Generate a single shared sample buffer so baseline/optimized can verify
        # equivalent workloads under identical inputs.
        self.samples = torch.randn(self.num_samples, self.hidden_dim, device=self.device)
        self.batches = [
            self.samples[i * self.batch_size:(i + 1) * self.batch_size]
            for i in range(self.num_batches)
        ]

        # Keep probe as a view into the timed inputs so jitter checks are meaningful.
        self._verify_input = self.samples[:2].detach()
        self._synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: static batches processed sequentially."""
        assert self.model is not None and self.batches is not None
        with self._nvtx_range("baseline_continuous_batching"):
            with torch.no_grad():
                outputs = []
                for batch in self.batches:
                    outputs.append(self.model(batch))
                self.output = torch.cat(outputs, dim=0)

    def capture_verification_payload(self) -> None:
        if self.model is None or self._verify_input is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"probe": self._verify_input},
            output=self.output,
            batch_size=self.num_samples,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.5, 5.0),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.batches = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
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
        if self.model is None:
            return "Model not initialized"
        if self.batches is None:
            return "Batches not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineContinuousBatchingBenchmark()
