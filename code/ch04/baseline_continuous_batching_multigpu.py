"""Multi-GPU wrapper for continuous batching baseline; skips when GPUs < 2."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch04.baseline_continuous_batching import BaselineContinuousBatchingBenchmark


def get_benchmark() -> BaselineContinuousBatchingBenchmark:
    return BaselineContinuousBatchingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)

