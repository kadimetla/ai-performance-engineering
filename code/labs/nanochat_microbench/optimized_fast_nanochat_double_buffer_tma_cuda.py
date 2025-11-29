"""Non-Triton warp-specialized/TMA-backed variant using the moe_cuda optimized kernel."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.moe_cuda.optimized_decode_kernel import OptimizedDecodeKernelBenchmark  # noqa: E402


class NanoChatWarpSpecializedCudaBenchmark(OptimizedDecodeKernelBenchmark):
    """Reuse the moe_cuda TMA double-buffered kernel as a non-Triton WS/TMA path."""

    def __init__(self) -> None:
        super().__init__()
        # Use a slightly smaller shape to match nanochat hidden size better
        self.rows = 4096
        self.cols = 4096


def get_benchmark() -> NanoChatWarpSpecializedCudaBenchmark:
    return NanoChatWarpSpecializedCudaBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode  # noqa: E402

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"\noptimized_fast_nanochat_warp_specialized_cuda: {mean:.3f} ms/iter")
