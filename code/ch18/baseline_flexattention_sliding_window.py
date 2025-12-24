"""Baseline sliding-window attention using dense masking."""

from __future__ import annotations

from ch18.flexattention_sliding_window_common import SlidingWindowAttentionBenchmark
from core.harness.benchmark_harness import BaseBenchmark


class BaselineSlidingWindowAttentionBenchmark(SlidingWindowAttentionBenchmark):
    def __init__(self) -> None:
        super().__init__(use_flex=False)


def get_benchmark() -> BaseBenchmark:
    return BaselineSlidingWindowAttentionBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
