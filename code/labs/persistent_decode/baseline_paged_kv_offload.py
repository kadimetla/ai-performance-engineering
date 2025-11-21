"""Baseline paged KV-cache benchmark without fusion checks or async copies.

- Stores cold KV pages in pageable CPU memory.
- Tries FP8 KV even when no fused attention path is present (may fall back).
- Uses blocking H2D copies and no prefetch, so TTFT-style latency is higher.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
        page_tokens=512,
        decode_tokens=64,
        use_pinned_stage=False,
        use_async_stream=False,
        use_memmap=False,
        prefer_fp8=True,  # naive: request FP8 even if fused path is absent
        require_fused_fp8=False,
        fallback_dtype=torch.float16,
        prefetch_next_page=False,
    )
    return PagedKVOffloadBenchmark(cfg, label="paged_kv_offload_baseline")


if __name__ == "__main__":
    benchmark = get_benchmark()
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"\npaged_kv_offload_baseline: {mean:.3f} ms/iter")
