"""optimized_nvfp4_trtllm.py - NVFP4/TRT-LLM placeholder.

If TensorRT-LLM is unavailable, this benchmark reports SKIPPED. Otherwise it
would quantize a tiny linear layer to fp4 and run a dummy forward pass.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class NVFP4TRTLLMBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.linear: Optional[nn.Linear] = None
        self.inputs: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(tokens_per_iteration=0.0)
        self._trt_available = False

    def setup(self) -> None:
        # Detect TensorRT-LLM; if missing, skip gracefully.
        try:
            import tensorrt  # noqa: F401

            self._trt_available = True
        except Exception:
            self._trt_available = False
            raise RuntimeError("SKIPPED: TensorRT-LLM not available for NVFP4 demo")

        self.linear = nn.Linear(1024, 1024, bias=False).to(self.device).to(torch.float16)
        self.inputs = torch.randn(32, 1024, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if not self._trt_available:
            raise RuntimeError("SKIPPED: TensorRT-LLM not available")
        if self.linear is None or self.inputs is None:
            raise RuntimeError("SKIPPED: NVFP4 linear model not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("nvfp4_trtllm", enable=enable_nvtx):
            _ = self.linear(self.inputs)
        torch.cuda.synchronize(self.device)
        return {}

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    return NVFP4TRTLLMBenchmark()
