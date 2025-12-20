"""Optimized: Transformer Engine FP4 (Blackwell NVFP4) for prefill/TTFT.

FP4 benefits show up most clearly in the prefill (TTFT) phase where GEMMs are
large (batch * prompt_tokens rows). Small-batch decode tends to be overhead-bound
for FP4, so this benchmark uses a prefill-only workload and compares against
`baseline_decode_fp4.py` to keep the workload equivalent.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata  # noqa: E402


def get_benchmark() -> DecodeBenchmark:
    """FP4 decode path using Transformer Engine.

    Workload matches `baseline_decode_fp4.py` exactly; only precision changes.
    """
    cfg = DecodeConfig(
        batch_size=64,
        prompt_tokens=1024,
        decode_tokens=0,
        hidden_size=8192,
        use_fp4=True,
        use_pinned_host=False,
        use_copy_stream=False,
        use_compute_stream=False,
        use_torch_compile=False,  # TE FP4 not compatible with torch.compile
        use_cuda_graphs=False,
        label="optimized_decode_fp4",
        iterations=12,
        warmup=15,
    )
    return attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
