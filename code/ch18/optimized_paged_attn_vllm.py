"""optimized_paged_attn_vllm.py - Chunked/paged attention stand-in.

Imitates vLLM-style paged attention by processing KV in blocks and reusing a
small cache tensor. Keeps dependencies minimal so it can run in the harness.
"""

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


class OptimizedPagedAttnBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.qkv: Optional[torch.Tensor] = None
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None
        self.block_size = 64
        self._workload = WorkloadMetadata(tokens_per_iteration=0.0)

    def setup(self) -> None:
        torch.manual_seed(1)
        b, h, s, d = 1, 8, 256, 64
        self.qkv = torch.randn(b, h, s, 3, d, device=self.device, dtype=torch.bfloat16)
        # Allocate a small KV cache block and reuse it across blocks.
        self.cache_k = torch.empty(b, h, self.block_size, d, device=self.device, dtype=torch.bfloat16)
        self.cache_v = torch.empty_like(self.cache_k)
        torch.cuda.synchronize(self.device)

    def _paged_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        outputs = []
        seq_len = q.size(2)
        for start in range(0, seq_len, self.block_size):
            end = min(start + self.block_size, seq_len)
            block_k = k[:, :, start:end].contiguous()
            block_v = v[:, :, start:end].contiguous()
            # Emulate residency by copying into a reusable cache.
            self.cache_k[:, :, : block_k.size(2)] = block_k
            self.cache_v[:, :, : block_v.size(2)] = block_v
            out = F.scaled_dot_product_attention(q[:, :, start:end], self.cache_k[:, :, : end - start], self.cache_v[:, :, : end - start])
            outputs.append(out)
        return torch.cat(outputs, dim=2)

    def benchmark_fn(self) -> Optional[dict]:
        if self.qkv is None or self.cache_k is None or self.cache_v is None:
            raise RuntimeError("SKIPPED: paged attention buffers not initialized")
        q = self.qkv[:, :, :, 0]
        k = self.qkv[:, :, :, 1]
        v = self.qkv[:, :, :, 2]

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("paged_attn_vllm", enable=enable_nvtx):
            _ = self._paged_attention(q, k, v)
        torch.cuda.synchronize(self.device)
        return {}

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    return OptimizedPagedAttnBenchmark()
