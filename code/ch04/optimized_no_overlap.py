"""Optimized wrapper for the overlap-enabled DDP demo.

This matches the chapter narrative and expectations by exposing the
overlapped training path as `optimized_no_overlap.py`, reusing the
implementation from `optimized_ddp_overlap.py`.
"""

from __future__ import annotations

from pathlib import Path
import sys

CHAPTER_DIR = Path(__file__).parent
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from optimized_ddp_overlap import OptimizedOverlapDdpBenchmark


def get_benchmark() -> OptimizedOverlapDdpBenchmark:
    """Factory used by the harness."""
    return OptimizedOverlapDdpBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
