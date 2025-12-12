"""Alias to reuse the optimized MoE inference benchmark implemented in ch18."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch18.run_vllm_decoder import VLLMMoEInferenceBenchmark  # noqa: E402


class Ch15VLLMMoEInferenceBenchmark(VLLMMoEInferenceBenchmark):
    """Chapter-local wrapper for the ch18 MoE inference benchmark."""


def get_benchmark():
    return Ch15VLLMMoEInferenceBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
