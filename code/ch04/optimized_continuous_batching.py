"""Optimized continuous batching shim for ch04 multi-GPU wrappers.

Reuses the chapter 15 optimized implementation so the ch04 harness can discover
the benchmark and gracefully skip on single-GPU hosts.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch15.optimized_continuous_batching import (
    OptimizedContinuousBatchingBenchmark as _OptimizedContinuousBatchingBenchmark,
)


class OptimizedContinuousBatchingBenchmark(_OptimizedContinuousBatchingBenchmark):
    """Alias to chapter 15 optimized; requires >=2 GPUs for ch04 runs."""

    allow_cpu = True

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: requires >=2 GPUs")
        super().setup()


def get_benchmark() -> OptimizedContinuousBatchingBenchmark:
    return OptimizedContinuousBatchingBenchmark()

