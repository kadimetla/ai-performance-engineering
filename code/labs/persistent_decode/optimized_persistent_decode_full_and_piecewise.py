"""Graph-mode sweep for persistent decode: full vs piecewise vs fallback."""

from __future__ import annotations

from labs.persistent_decode.optimized_persistent_decode_graphs import (
    GraphMode,
    OptimizedPersistentDecodeGraphsBenchmark,
)
from common.python.benchmark_harness import BaseBenchmark


class OptimizedPersistentDecodeFullAndPiecewiseBenchmark(OptimizedPersistentDecodeGraphsBenchmark):
    """Default to FULL_AND_PIECEWISE to mirror vLLM graph heuristics."""

    def __init__(self) -> None:
        super().__init__(graph_mode=GraphMode.FULL_AND_PIECEWISE)


def get_benchmark() -> BaseBenchmark:
    return OptimizedPersistentDecodeFullAndPiecewiseBenchmark()
