"""baseline_kv_cache_management.py - Baseline KV cache without management."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch15.verification_payload_mixin import VerificationPayloadMixin


class BaselineKVCacheManagementBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: recomputes KV every step (no cache reuse)."""
    
    def __init__(self):
        super().__init__()
        self.q_proj: Optional[nn.Linear] = None
        self.k_proj: Optional[nn.Linear] = None
        self.v_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None
        self.inputs: Optional[list[torch.Tensor]] = None
        self.hidden_dim = 256
        self.num_heads = 8
        self.batch_size = 8
        self.steps = 32
        tokens = self.batch_size * self.steps
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verify_input: Optional[torch.Tensor] = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Initialize model without KV cache management."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            module.eval()
        
        self.inputs = [
            torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
            for _ in range(self.steps)
        ]
        self._synchronize()
        self._verify_input = torch.randn(1, 1, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
    
    def benchmark_fn(self) -> None:
        """Benchmark: KV cache without management."""
        assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None and self.out_proj is not None
        assert self.inputs is not None
        with self._nvtx_range("baseline_kv_cache_management"):
            with torch.no_grad():
                queries = torch.cat(self.inputs, dim=1)
                all_inputs = torch.cat(self.inputs, dim=1)

                q = self.q_proj(queries)
                k = self.k_proj(all_inputs)
                v = self.v_proj(all_inputs)

                head_dim = self.hidden_dim // self.num_heads
                q = q.view(self.batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
                k = k.view(self.batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
                v = v.view(self.batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

                attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn = attn.transpose(1, 2).contiguous().view(self.batch_size, -1, self.hidden_dim)
                output = self.out_proj(attn)
                self.output = output.detach().clone()
                _ = output[:, -1, :].sum()
            self._synchronize()
        if self.output is not None and self._verify_input is not None:
            param_count = 0
            for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                if layer is not None:
                    param_count += sum(p.numel() for p in layer.parameters())
            self._set_verification_payload(
                inputs={"tokens": self._verify_input},
                output=self.output,
                batch_size=int(self._verify_input.shape[0]),
                parameter_count=param_count,
                precision_flags={
                    "fp16": False,
                    "bf16": True,
                    "fp8": False,
                    "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
                },
                output_tolerance=(1e-3, 1e-3),
            )
        if self.output is None:
            raise RuntimeError("No output computed during benchmark")
    
    def teardown(self) -> None:
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.out_proj = None
        self.inputs = None
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
        if any(layer is None for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj)):
            return "Projection layers not initialized"
        if self.inputs is None:
            return "Inputs not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return super().get_output_tolerance()


def get_benchmark() -> BaseBenchmark:
    return BaselineKVCacheManagementBenchmark()
