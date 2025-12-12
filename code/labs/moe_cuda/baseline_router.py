"""labs.moe_cuda/baseline_router.py - Dense MoE router baseline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range
from core.harness.benchmark_harness import WorkloadMetadata
from core.utils.compile_utils import enable_tf32
from labs.moe_cuda.optimized_router_vectorized import VectorizedTopKMoE


class Expert(nn.Module):
    """Simple feed-forward expert."""

    def __init__(self, hidden_size: int, expansion: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expansion),
            nn.GELU(),
            nn.Linear(hidden_size * expansion, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - exercised via benchmark
        return self.net(x)


class DenseRouterMoE(nn.Module):
    """Runs every expert for each input token."""

    def __init__(self, hidden_size: int, num_experts: int, expansion: int = 2) -> None:
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(hidden_size, expansion) for _ in range(num_experts)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover - benchmarked
        output = torch.zeros_like(tokens)
        for expert in self.experts:
            output += expert(tokens)
        return output / len(self.experts)


class BaselineRouterDenseBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline top-k MoE router using an eager VectorizedTopKMoE forward."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024
        self.num_experts = 32
        self.top_k = 2
        self.batch_size = 4096
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.hidden_size),
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("labs.moe_cuda requires CUDA for fair comparison")

        import gc
        
        # Clean up CUDA graph state from previous benchmarks
        # to prevent "Offset increment outside graph capture" errors
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        try:
            if hasattr(torch.cuda, 'graph_pool_trim'):
                torch.cuda.graph_pool_trim()
        except Exception:
            pass
        
        # Reset CUDA RNG state to prevent graph capture errors
        try:
            device_idx = torch.cuda.current_device()
            gen = torch.cuda.default_generators[device_idx]
            gen.set_offset(0)
            gen.manual_seed(42)
        except Exception:
            pass
        
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        
        try:
            torch._inductor.cudagraph_trees.reset_cudagraph_trees()
        except Exception:
            pass

        enable_tf32()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = VectorizedTopKMoE(self.hidden_size, self.num_experts, self.top_k, expansion=2)
        model = model.to(self.device, dtype=torch.bfloat16)
        model.eval()
        self.model = model

        # Use CPU randn + to(device) to avoid CUDA RNG graph capture issues
        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_size,
            dtype=torch.bfloat16,
        ).to(self.device)
        torch.cuda.synchronize(self.device)
        self.output = None

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Model not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_cuda_dense_router", enable=enable_nvtx):
            with torch.inference_mode():
                out = self.model(self.inputs)
                self.output = out.detach().float().clone()
        torch.cuda.synchronize(self.device)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.inputs.detach()},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={"bf16": True, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        self.inputs = None
        self.output = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)  # Min warmup for CUDA

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics."""
        # Estimate problem size for roofline analysis
        n = getattr(self, 'N', 0) or getattr(self, 'hidden_dim', 0) or 4096
        batch = getattr(self, 'batch_size', 1) or getattr(self, 'batch', 1)
        # Simple FLOP estimate for linear layers
        flops = 2.0 * batch * n * n  # Rough estimate
        bytes_moved = batch * n * 4.0  # Input/output bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        return {
            "router.estimated_flops": flops,
            "router.estimated_bytes": bytes_moved,
            "router.arithmetic_intensity": arithmetic_intensity,
        }

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Dense MoE model missing"
        if self.inputs is None:
            return "Inputs missing"
        return None

def get_benchmark() -> BaseBenchmark:
    return BaselineRouterDenseBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
