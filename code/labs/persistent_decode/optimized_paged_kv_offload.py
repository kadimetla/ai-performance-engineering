"""Optimized paged KV-cache benchmark with pinned staging + async H2D copies.

- Uses pinned staging buffers and an async copy stream.
- Prefetches the next page to overlap staging with compute.
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
        batch_size=2,
        num_heads=16,
        head_dim=128,
        max_seq_len=8192,
        page_tokens=1024,
        decode_tokens=128,
        use_pinned_stage=True,
        use_async_stream=True,
        use_memmap=False,
        prefer_fp8=True,
        require_fused_fp8=False,
        fallback_dtype=torch.float16,
        prefetch_next_page=True,
    )
    return PagedKVOffloadBenchmark(cfg, label="paged_kv_offload_optimized")


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
