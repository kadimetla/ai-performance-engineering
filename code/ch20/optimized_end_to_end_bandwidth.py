"""optimized_end_to_end_bandwidth.py - Optimized end-to-end bandwidth."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

try:
    import ch20.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

try:
    from core.utils.logger import get_logger
    LOGGER = get_logger(__name__)
except ImportError:  # pragma: no cover
    LOGGER = None

from ch20.inductor_guard import (
    disable_inductor_cudagraph_features,
    restore_inductor_cudagraph_features,
    InductorCudagraphState,
)

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.utils.compile_utils import is_torch_compile_supported_on_device


class SimplePipeline(nn.Module):
    """Simple inference pipeline for bandwidth analysis."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OptimizedEndToEndBandwidthBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized end-to-end bandwidth - FP16 + optional compile."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[list[torch.Tensor]] = None
        self.outputs: Optional[list[torch.Tensor]] = None
        self.output: Optional[torch.Tensor] = None
        self.batch_size = 32
        self.hidden_dim = 1024
        self.num_batches = 10
        self._inductor_cfg_state: Optional[InductorCudagraphState] = None
        self._used_compiled_model = False
        self._compile_error: Optional[str] = None
        tokens = self.batch_size * self.num_batches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(tokens),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        self._inductor_cfg_state = disable_inductor_cudagraph_features()
        try:
            if torch.cuda.is_available():
                # Warm the primary context and cuBLAS handle before any heavy ops.
                torch.cuda.init()
                warmup_device = self.device
                if warmup_device.index is None:
                    warmup_device = torch.device("cuda", torch.cuda.current_device())
                torch.cuda.set_device(warmup_device)
                torch.ones((1, 1), device=warmup_device).matmul(torch.ones((1, 1), device=warmup_device))
                torch.cuda.synchronize()

            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            torch.manual_seed(42)
            
            model = SimplePipeline(hidden_dim=self.hidden_dim).to(self.device).half().eval()
            compile_supported, compile_reason = is_torch_compile_supported_on_device()
            disable_compile = bool(os.environ.get("PYTEST_CURRENT_TEST"))
            if compile_supported and not disable_compile:
                try:
                    compiled_model = torch.compile(model, mode="reduce-overhead")
                    self.model = compiled_model
                    self._used_compiled_model = True
                except Exception as exc:
                    self._compile_error = f"{exc.__class__.__name__}: {exc}"
                    self._used_compiled_model = False
                    self.model = model
                    if LOGGER is not None:
                        LOGGER.warning(
                            "torch.compile failed for %s; falling back to eager.",
                            self.__class__.__name__,
                            exc_info=exc,
                        )
            else:
                reason = compile_reason or "torch.compile unsupported on this GPU"
                if disable_compile:
                    reason = "torch.compile disabled under pytest"
                self._compile_error = reason
                self._used_compiled_model = False
                self.model = model

            test_input = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            for _ in range(3):
                with torch.no_grad():
                    _ = self.model(test_input)
            torch.cuda.synchronize()
            
            self.inputs = [
                torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16).contiguous()
                for _ in range(self.num_batches)
            ]
            self.outputs = []
            self.output = None
            
            for inp in self.inputs[:5]:
                with torch.no_grad():
                    _ = self.model(inp)
            self._synchronize()
        except Exception:
            restore_inductor_cudagraph_features(self._inductor_cfg_state)
            self._inductor_cfg_state = None
            raise
    
    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None
        with self._nvtx_range("optimized_end_to_end_bandwidth"):
            torch.cuda.reset_peak_memory_stats()
            self.outputs = []
            with torch.no_grad():
                for inp in self.inputs:
                    out = self.model(inp)
                    self.outputs.append(out)
            if self.outputs:
                self.output = torch.stack(self.outputs)
            else:
                self.output = None
            self._synchronize()

    def capture_verification_payload(self) -> None:
        if self.model is None or self.inputs is None or self.output is None:
            raise RuntimeError("capture_verification_payload() requires completed benchmark run")
        stacked_inputs = torch.stack(self.inputs)
        self._set_verification_payload(
            inputs={"inputs": stacked_inputs},
            output=self.output,
            batch_size=int(stacked_inputs.shape[0]),
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.outputs = None
        torch.cuda.empty_cache()
        restore_inductor_cudagraph_features(self._inductor_cfg_state)
        self._inductor_cfg_state = None
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=True,
            enable_profiling=False,
            use_subprocess=False,
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
        if self.outputs is None or len(self.outputs) != self.num_batches:
            return f"Expected {self.num_batches} outputs, got {len(self.outputs) if self.outputs else 0}"
        return None

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()


def get_benchmark() -> BaseBenchmark:
    return OptimizedEndToEndBandwidthBenchmark()
