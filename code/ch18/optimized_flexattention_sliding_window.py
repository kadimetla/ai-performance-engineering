"""Optimized sliding-window attention using FlexAttention block masks."""

from __future__ import annotations

from ch18.flexattention_sliding_window_common import SlidingWindowAttentionBenchmark
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedSlidingWindowAttentionBenchmark(SlidingWindowAttentionBenchmark):
    def __init__(self) -> None:
        super().__init__(use_flex=True)


def get_benchmark() -> BaseBenchmark:
    return OptimizedSlidingWindowAttentionBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
