"""optimized_multiple_all_techniques.py - Alias to the combined techniques benchmark."""

from __future__ import annotations

try:
    import ch20.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from ch20.optimized_multiple_unoptimized import OptimizedAllTechniquesBenchmark as _OptimizedAllTechniquesBenchmark
from common.python.benchmark_harness import BaseBenchmark


class OptimizedAllTechniquesBenchmark(_OptimizedAllTechniquesBenchmark):
    """Thin alias kept for backwards compatibility with integration tests."""


def get_benchmark() -> BaseBenchmark:
    return OptimizedAllTechniquesBenchmark()
