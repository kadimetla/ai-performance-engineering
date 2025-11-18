"""optimized_training_single.py - Optimized training loop."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


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
        self.batch_size = 8
        self.hidden_dim = 4096
        self.train_steps = 4
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        
        base_model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().train()
        self.model = torch.compile(base_model, mode="reduce-overhead")
        
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        try:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, fused=True)
        except TypeError:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        for _ in range(3):
            for _ in range(self.train_steps):
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(self.inputs)
                    loss = self.criterion(outputs, self.targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self._synchronize()
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None and self.targets is not None
        assert self.optimizer is not None and self.criterion is not None and self.scaler is not None
        with self._nvtx_range("training_optimized"):
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = self.model(self.inputs)
                loss = self.criterion(outputs, self.targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._synchronize()
    
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

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedTrainingDistributedBenchmark()
