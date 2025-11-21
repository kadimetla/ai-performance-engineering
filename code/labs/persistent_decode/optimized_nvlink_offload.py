"""Optimized KV offload benchmark using pinned host memory and async copies."""

from __future__ import annotations

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
        batch_size=2,
        num_layers=2,
        num_heads=8,
        head_dim=64,
        max_seq_len=4096,
        chunk_tokens=1024,
    )
    return NvlinkOffloadBenchmark(cfg, label="nvlink_offload_optimized")


if __name__ == "__main__":
    benchmark = get_benchmark()
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"\nnvlink_offload_optimized: {mean:.3f} ms/iter")
