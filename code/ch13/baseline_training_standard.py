"""baseline_training_standard.py - Standard training without checkpointing (baseline)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch13.workload_config import WORKLOAD


class DeepModel(nn.Module):
    """Deep model for demonstrating checkpoint benefits."""
    
    def __init__(self, hidden_dim=2048, num_layers=20):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class BaselineTrainingBenchmark(BaseBenchmark):
    """Standard training that stores all activations (memory heavy)."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.workload = WORKLOAD
        self.hidden_dim = self.workload.training_hidden_dim
        self.num_layers = self.workload.training_layers_baseline
        self.global_batch = self.workload.global_batch_size
        self.micro_batch = self.workload.micro_batch_size
        self.accum_steps = self.global_batch // self.micro_batch
        tokens = self.global_batch * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.accum_steps),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        self.model = DeepModel(hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        self.model = self.model.to(self.device).train()
        
        self.inputs = torch.randn(self.global_batch, self.hidden_dim, device=self.device)
        self.targets = torch.randn(self.global_batch, self.hidden_dim, device=self.device)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        if any(v is None for v in (self.model, self.inputs, self.targets, self.optimizer, self.criterion)):
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("baseline_training_standard"):
            self.optimizer.zero_grad(set_to_none=True)
            for start in range(0, self.global_batch, self.micro_batch):
                end = start + self.micro_batch
                inputs = self.inputs[start:end]
                targets = self.targets[start:end]
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


def get_benchmark() -> BaselineTrainingBenchmark:
    """Factory function for harness discovery."""
    return BaselineTrainingBenchmark()
