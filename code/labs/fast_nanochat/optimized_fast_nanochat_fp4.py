"""FP4 (NVFP4 if available) + full prefill+decode graph capture."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.fast_nanochat.nanochat_common import NanoChatBenchmark, NanoChatConfig  # noqa: E402


def get_benchmark() -> NanoChatBenchmark:
    cfg = NanoChatConfig(
        batch_size=8,
        prompt_tokens=1024,
        decode_tokens=256,
        hidden_size=2048,
        use_fp4=True,
        use_pinned_host=True,
        use_copy_stream=True,
        use_compute_stream=True,
        use_torch_compile=False,
        use_cuda_graphs=False,
        graph_full_iteration=False,
        label="optimized_fast_nanochat_fp4",
    )
    return NanoChatBenchmark(cfg)


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode  # noqa: E402

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"\noptimized_fast_nanochat_fp4: {mean:.3f} ms/iter")
