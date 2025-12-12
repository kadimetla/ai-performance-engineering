"""Baseline guided decoding - standard decoding without guidance."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import ch01.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


class BaselineGuidedDecodingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: standard decoding without schema/structure guidance."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.TransformerDecoder] = None
        self.input_ids: Optional[torch.Tensor] = None
        self.embedded_input: Optional[torch.Tensor] = None
        self.memory: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output: Optional[torch.Tensor] = None
        self.max_length = 20
        self.batch_size = 4
        self.seq_len = 10
        self.hidden_dim = 256
        self.parameter_count = 0
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model, fixed inputs, and verification output."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        vocab_size = 1000

        # On GB10 (sm_12x), flash SDP routes to sm80-only kernels; keep this baseline stable by using math SDP.
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(self.device)
            if major >= 12:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_cudnn_sdp(False)
        
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, batch_first=True),
            num_layers=2,
        ).to(self.device).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.input_ids = torch.randint(0, vocab_size, (self.batch_size, self.seq_len), device=self.device)
        
        # Create FIXED inputs for deterministic verification
        self.embedded_input = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device)
        self.memory = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: standard decoding without guidance."""
        assert self.model is not None and self.embedded_input is not None and self.memory is not None
        with self._nvtx_range("baseline_guided_decoding"):
            with torch.no_grad():
                # Use fixed inputs for deterministic verification
                output = self.model(self.embedded_input, self.memory)
                self.output = output
                _ = output.sum()
            self._synchronize()

    def capture_verification_payload(self) -> None:
        if self.embedded_input is None or self.memory is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"embedded_input": self.embedded_input, "memory": self.memory},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=int(self.parameter_count),
            output_tolerance=(1e-4, 1e-4),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input_ids = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_environment_metrics
        return compute_environment_metrics(
            gpu_count=getattr(self, 'gpu_count', 1),
            gpu_memory_gb=getattr(self, 'gpu_memory_gb', 80.0),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineGuidedDecodingBenchmark()
