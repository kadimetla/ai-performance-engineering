"""Python harness wrapper for baseline_tma_bulk_tensor_2d.cu."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


def _require_torch_2_10() -> None:
    """Ensure PyTorch 2.10+ for consistency across Blackwell targets."""
    version = torch.__version__.split("+")[0]
    parts = version.split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    if (major, minor) < (2, 10):
        raise RuntimeError(f"PyTorch >=2.10 required (found {torch.__version__})")


class BaselineTMABulkTensor2D(CudaBinaryBenchmark):
    """Wraps the manual 2D bulk copy (no TMA)."""

    def __init__(self) -> None:
        _require_torch_2_10()
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_tma_bulk_tensor_2d",
            friendly_name="Baseline 2D tensor copy (manual)",
            iterations=3,
            warmup=1,
            timeout_seconds=120,
        )


def get_benchmark() -> BaselineTMABulkTensor2D:
    """Factory for discover_benchmarks()."""
    return BaselineTMABulkTensor2D()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    timing = result.timing.mean_ms if result.timing else 0.0
    print(f"\nBaseline 2D tensor copy (manual): {timing:.3f} ms")
