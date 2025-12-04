"""Baseline wrapper for the no-overlap DDP demo.

This module exists to match the chapter text and expectations file.  It
simply re-exports the BaselineNoOverlapBenchmark defined in
`baseline_ddp_no_overlap.py`, which performs a single-process DDP-like step
without communication/compute overlap.
"""

from __future__ import annotations

from pathlib import Path
import sys

CHAPTER_DIR = Path(__file__).parent
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from baseline_ddp_no_overlap import BaselineNoOverlapBenchmark


def get_benchmark() -> BaselineNoOverlapBenchmark:
    """Factory used by the harness."""
    return BaselineNoOverlapBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
