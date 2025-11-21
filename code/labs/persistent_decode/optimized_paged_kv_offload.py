"""Optimized paged KV-cache benchmark with fused FP8 gating and NVMe-style offload.

- Uses pinned staging buffers and an async copy stream.
- Backs cold pages by a memmap file to mimic NVMe/SSD offload.
- Enables FP8 KV only when a fused FlashAttention path is available on B200/GB200.
- Prefetches the next page to hide a portion of TTFT on long contexts.
"""

from __future__ import annotations

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
        max_seq_len=16384,
        page_tokens=512,
        decode_tokens=96,
        use_pinned_stage=True,
        use_async_stream=True,
        use_memmap=True,  # mimic NVMe/SSD backing
        prefer_fp8=True,
        require_fused_fp8=True,  # Only use FP8 KV when fused kernels are likely present
        fallback_dtype=torch.float16,
        prefetch_next_page=True,
    )
    return PagedKVOffloadBenchmark(cfg, label="paged_kv_offload_optimized")


if __name__ == "__main__":
    benchmark = get_benchmark()
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"\npaged_kv_offload_optimized: {mean:.3f} ms/iter")
