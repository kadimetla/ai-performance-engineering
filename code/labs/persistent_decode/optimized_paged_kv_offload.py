"""Optimized paged KV-cache benchmark with pinned staging + FP8 KV.

- Uses pinned staging buffers with direct H2D copies.
- Enables FP8 KV only when a fused FlashAttention path is available on B200/GB200.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathlib import Path
import sys

import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.persistent_decode.paged_kv_offload_common import PagedKVConfig, PagedKVOffloadBenchmark


def get_benchmark() -> PagedKVOffloadBenchmark:
    cfg = PagedKVConfig(
        batch_size=4,
        num_heads=16,
        head_dim=128,
        max_seq_len=32768,
        page_tokens=2048,
        decode_tokens=64,
        repeat_pages=32,
        use_pinned_stage=True,
        use_async_stream=False,
        use_memmap=True,
        prefer_fp8=True,
        require_fused_fp8=True,
        fallback_dtype=torch.float16,
        prefetch_next_page=False,
        use_direct_h2d=True,
    )
    return PagedKVOffloadBenchmark(cfg, label="paged_kv_offload_optimized")


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
