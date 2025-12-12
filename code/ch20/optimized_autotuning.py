"""optimized_autotuning.py - Lightweight autotuning demo benchmark."""

from __future__ import annotations

from typing import Optional
from contextlib import nullcontext

import torch
import torch.nn as nn

try:
    from arch_config import prefer_sdpa_backends  # type: ignore
    from core.utils.compile_utils import enable_tf32  # type: ignore
except Exception:  # pragma: no cover - defensive import
    prefer_sdpa_backends = None  # type: ignore
    enable_tf32 = None  # type: ignore

try:
    import ch20.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.utils.compile_utils import compile_model


class AutotuneModel(nn.Module):
    """Tiny MLP to exercise autotuning paths."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        return self.fc2(x)


class OptimizedAutotuningBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs a small model with torch.compile to validate autotune plumbing."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.batch = 16
        self.hidden_dim = 1024
        tokens = self.batch * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._verify_input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if enable_tf32 is not None:
            enable_tf32(set_global_precision=True)
        else:
            try:
                torch.set_float32_matmul_precision("high")
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Failed to set float32_matmul_precision='high': {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        model = AutotuneModel(self.hidden_dim).to(self.device, dtype=torch.bfloat16).eval()
        # compile_model skips safely on unsupported architectures
        self.model = compile_model(
            model,
            mode="max-autotune",
            fullgraph=False,
            dynamic=False,
        )
        self.inputs = torch.randn(self.batch, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        self._verify_input = self.inputs[0:1].clone()
        # Warm a couple runs to trigger compile/autotune caches
        for _ in range(3):
            sdpa_ctx = prefer_sdpa_backends() if prefer_sdpa_backends is not None else nullcontext()
            with torch.no_grad(), sdpa_ctx:
                _ = self.model(self.inputs)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None
        sdpa_ctx = prefer_sdpa_backends() if prefer_sdpa_backends is not None else nullcontext()
        with self._nvtx_range("optimized_autotuning"):
            with torch.no_grad(), sdpa_ctx:
                _ = self.model(self.inputs)
            self._synchronize()

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.model is None:
            raise RuntimeError("setup() must prepare verify input before verification")
        with torch.no_grad():
            self.output = self.model(self._verify_input).float().clone()
        self._set_verification_payload(
            inputs={"verify_input": self._verify_input},
            output=self.output,
            batch_size=int(self._verify_input.shape[0]),
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            use_subprocess=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_ai_optimization_metrics
        return compute_ai_optimization_metrics(
            original_time_ms=getattr(self, '_original_ms', 10.0),
            ai_optimized_time_ms=getattr(self, '_optimized_ms', 5.0),
            suggestions_applied=getattr(self, '_suggestions_applied', 1),
            suggestions_total=getattr(self, '_suggestions_total', 1),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.inputs is None:
            return "Input tensor not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()


def get_benchmark() -> BaseBenchmark:
    return OptimizedAutotuningBenchmark()
