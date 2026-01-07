"""Baseline paged KV-cache prefetch benchmark (pageable sync copy, no prefetch).

- Uses pageable staging buffers and a synchronous copy path.
- Does not prefetch the next page, so H2D copies block the iteration.
- Uses a pageable host cache (memmap disabled) to isolate overlap effects.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from labs.persistent_decode.paged_kv_offload_common import PagedKVConfig, PagedKVOffloadBenchmark


def get_benchmark() -> PagedKVOffloadBenchmark:
    cfg = PagedKVConfig(
        batch_size=4,
        num_heads=16,
        head_dim=128,
        max_seq_len=65536,
        page_tokens=8192,
        decode_tokens=8,
        repeat_pages=64,
        use_pinned_stage=False,
        use_async_stream=False,
        use_memmap=False,
        prefer_fp8=False,
        require_fused_fp8=False,
        fallback_dtype=torch.float16,
        prefetch_next_page=False,
        use_direct_h2d=False,
    )
    return PagedKVOffloadBenchmark(cfg, label="paged_kv_offload_prefetch_baseline")


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
