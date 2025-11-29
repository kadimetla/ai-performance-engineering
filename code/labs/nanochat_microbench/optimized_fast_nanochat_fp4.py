"""FP4 (NVFP4 if available) + full prefill+decode graph capture."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.nanochat_microbench.nanochat_common import NanoChatBenchmark, NanoChatConfig, attach_benchmark_metadata  # noqa: E402
import os


def get_benchmark() -> NanoChatBenchmark:
    os.environ.setdefault(
        "PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:512"
    )
    mem_cap = os.getenv("NANOCHAT_FP4_MEM_CAP", "0")
    if mem_cap == "2":
        batch_size, prompt_tokens, decode_tokens, hidden_size = 8, 1024, 256, 2048
    elif mem_cap == "1":
        batch_size, prompt_tokens, decode_tokens, hidden_size = 16, 2048, 512, 4096
    else:
        batch_size, prompt_tokens, decode_tokens, hidden_size = 32, 4096, 1024, 6144
    cfg = NanoChatConfig(
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        decode_tokens=decode_tokens,
        hidden_size=hidden_size,
        use_fp4=True,
        use_pinned_host=True,
        use_copy_stream=True,
        use_compute_stream=True,
        use_torch_compile=False,
        use_cuda_graphs=False,
        graph_full_iteration=False,
        label="optimized_fast_nanochat_fp4",
    )
    return attach_benchmark_metadata(NanoChatBenchmark(cfg), __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode  # noqa: E402

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"\noptimized_fast_nanochat_fp4: {mean:.3f} ms/iter")
