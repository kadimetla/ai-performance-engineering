"""Optimized KV offload benchmark using pinned host memory and async copies."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.persistent_decode.nvlink_offload_common import NvlinkOffloadBenchmark, OffloadConfig


def get_benchmark() -> NvlinkOffloadBenchmark:
    cfg = OffloadConfig(
        use_pinned=True,
        non_blocking=True,
        use_copy_stream=True,
        batch_size=4,
        num_layers=4,
        num_heads=16,
        head_dim=64,
        max_seq_len=4096,
        chunk_tokens=4096,
    )
    return NvlinkOffloadBenchmark(cfg, label="nvlink_offload_optimized")


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
