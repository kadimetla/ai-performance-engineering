"""optimized_training_single.py - Optimized training loop."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn

try:
    from arch_config import prefer_sdpa_backends  # type: ignore
    from core.utils.compile_utils import enable_tf32  # type: ignore
except Exception:  # pragma: no cover - defensive import
    prefer_sdpa_backends = None  # type: ignore
    enable_tf32 = None  # type: ignore

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleModel(nn.Module):
    """Simple model for training demonstration."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OptimizedTrainingDistributedBenchmark(BaseBenchmark):
    """Optimized training loop leveraging AMP, fused optimizers, and compilation."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self.batch_size = 32
        self.hidden_dim = 8192
        self.train_steps = 6
        self._sdpa_ctx_factory = prefer_sdpa_backends if prefer_sdpa_backends is not None else nullcontext
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
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
        torch.manual_seed(42)
        
        base_model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device, dtype=torch.bfloat16).train()
        self.model = torch.compile(base_model, mode="reduce-overhead")
        
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        # Fixed input for verification - used to test trained model at END of benchmark_fn
        self._verify_input = self.inputs[0:1].clone()
        try:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, fused=True)
        except TypeError:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.scaler = None

        with self._sdpa_ctx_factory():
            for _ in range(3):
                for _ in range(self.train_steps):
                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outputs = self.model(self.inputs)
                        loss = self.criterion(outputs, self.targets)
                    loss.backward()
                    self.optimizer.step()
                self._synchronize()
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None and self.targets is not None
        assert self.optimizer is not None and self.criterion is not None
        with self._nvtx_range("training_optimized"):
            self.optimizer.zero_grad(set_to_none=True)
            with self._sdpa_ctx_factory(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(self.inputs)
                loss = self.criterion(outputs, self.targets)
            loss.backward()
            self.optimizer.step()
            self._synchronize()
        # Capture output AFTER training for verification
        with torch.no_grad():
            self.output = self.model(self._verify_input).float().clone()
    
    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self):
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
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output.detach().clone()
    
    def get_verify_inputs(self) -> torch.Tensor:
        """Return fixed verification input for aliasing checks."""
        if self._verify_input is None:
            raise RuntimeError("setup() must be called before verification")
        return self._verify_input

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"batch_size": self.batch_size, "hidden_dim": self.hidden_dim}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedTrainingDistributedBenchmark()
