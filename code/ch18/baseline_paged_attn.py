"""baseline_paged_attn.py - Dense SDPA baseline for paged attention demos."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class BaselinePagedAttnBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.qkv: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(tokens_per_iteration=0.0)

    def setup(self) -> None:
        torch.manual_seed(0)
        b, h, s, d = 1, 8, 256, 64
        self.qkv = torch.randn(b, h, s, 3, d, device=self.device, dtype=torch.bfloat16)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if self.qkv is None:
            raise RuntimeError("SKIPPED: QKV not initialized")
        q = self.qkv[:, :, :, 0]
        k = self.qkv[:, :, :, 1]
        v = self.qkv[:, :, :, 2]

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("paged_attn_baseline", enable=enable_nvtx):
            _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize(self.device)
        return {}

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    return BaselinePagedAttnBenchmark()
