"""Optimized: Transformer Engine FP8 (Blackwell MXFP8) for prefill/TTFT.

FP8 benefits show up most clearly in the prefill (TTFT) phase where GEMMs are
large (batch * prompt_tokens rows). Small-batch decode tends to be overhead-bound
for FP8, so this benchmark uses a prefill-only workload and compares against
`baseline_decode_fp8.py` to keep the workload equivalent.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata  # noqa: E402


def get_benchmark() -> DecodeBenchmark:
    """FP8 decode path using Transformer Engine.

    Workload matches `baseline_decode_fp8.py` exactly; only precision changes.
    """
    cfg = DecodeConfig(
        batch_size=128,
        prompt_tokens=1024,
        decode_tokens=0,
        hidden_size=8192,
        use_fp8=True,
        use_pinned_host=False,
        use_copy_stream=False,
        use_compute_stream=False,
        use_torch_compile=False,  # TE FP8 not compatible with torch.compile
        use_cuda_graphs=False,
        label="optimized_decode_fp8",
        iterations=12,
        warmup=15,
    )
    bench = attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)
    bench.signature_equivalence_group = "labs_decode_fp8_precision"
    bench.signature_equivalence_ignore_fields = ("precision_flags",)
    return bench


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
