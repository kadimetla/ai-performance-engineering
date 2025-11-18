"""optimized_training_standard.py - Gradient checkpointing optimization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch13.workload_config import WORKLOAD


class DeepModel(nn.Module):
    """Deep model with gradient checkpointing."""
    
    def __init__(self, hidden_dim=2048, num_layers=20, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(lambda y: torch.relu(layer(y)), x, use_reentrant=False)
            else:
                x = torch.relu(layer(x))
        return x


class OptimizedCheckpointBenchmark(BaseBenchmark):
    """Gradient checkpointing: memory-efficient, slightly slower."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer = None
        self.criterion = None
        self.workload = WORKLOAD
        self.hidden_dim = self.workload.training_hidden_dim
        self.num_layers = self.workload.training_layers_optimized
        self.global_batch = self.workload.global_batch_size
        self.micro_batch = self.workload.micro_batch_size
        self.accum_steps = self.global_batch // self.micro_batch
        self.dtype = torch.bfloat16
        tokens = self.global_batch * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.accum_steps),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        model = DeepModel(hidden_dim=self.hidden_dim, num_layers=self.num_layers, use_checkpoint=True)
        self.model = model.to(self.device, dtype=self.dtype).train()
        self.inputs = torch.randn(self.global_batch, self.hidden_dim, device=self.device, dtype=self.dtype)
        self.targets = torch.randn(self.global_batch, self.hidden_dim, device=self.device, dtype=self.dtype)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        if any(v is None for v in (self.model, self.inputs, self.targets, self.optimizer, self.criterion)):
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("training_standard_checkpoint"):
            self.optimizer.zero_grad(set_to_none=True)
            for inputs, targets in zip(
                self.inputs.view(self.accum_steps, self.micro_batch, self.hidden_dim),
                self.targets.view(self.accum_steps, self.micro_batch, self.hidden_dim),
            ):
                with autocast("cuda", dtype=self.dtype):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets) / self.accum_steps
                loss.backward()
            self.optimizer.step()
        self._synchronize()

    def teardown(self) -> None:
        """Cleanup."""
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.inputs is None:
            return "Input tensor not initialized"
        if self.targets is None:
            return "Target tensor not initialized"
        try:
            with torch.no_grad():
                test_output = self.model(self.inputs)
                if test_output.shape != self.targets.shape:
                    return f"Output shape mismatch: expected {self.targets.shape}, got {test_output.shape}"
                if not torch.isfinite(test_output).all():
                    return "Output contains non-finite values"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> OptimizedCheckpointBenchmark:
    """Factory function for harness discovery."""
    return OptimizedCheckpointBenchmark()
