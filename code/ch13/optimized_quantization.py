"""optimized_quantization.py - Optimized FP16 quantization for faster inference."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedQuantizationBenchmark(BaseBenchmark):
    """Optimized: FP16 quantization for faster inference with reduced memory."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.quantized_model = None
        self.data = None
        self.N = 65536
        tokens = self.N * 256
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ).to(self.device).to(torch.float32)
        
        self.model.eval()
        self.quantized_model = self.model.to(torch.float16)
        self.data = torch.randn(self.N, 256, device=self.device, dtype=torch.float16)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if self.quantized_model is None or self.data is None:
            raise RuntimeError("Model/data not initialized")
        with self._nvtx_range("optimized_quantization"):
            with torch.no_grad():
                _ = self.quantized_model(self.data)
        self._synchronize()
    
    def teardown(self) -> None:
        self.model = None
        self.quantized_model = None
        self.data = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def validate_result(self) -> Optional[str]:
        if self.quantized_model is None:
            return "Quantized model not initialized"
        return None


def get_benchmark() -> OptimizedQuantizationBenchmark:
    """Factory function for harness discovery."""
    return OptimizedQuantizationBenchmark()
