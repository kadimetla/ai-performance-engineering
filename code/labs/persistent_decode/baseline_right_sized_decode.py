"""Baseline wrapper for right-sized decode (flag-driven).

This is the harness-comparable baseline for `optimized_right_sized_decode.py`.

Baseline behavior:
- Always runs the naive per-token decode loop (one step at a time).
- Still honors the `--tier/--quantization` knobs so the *workload* matches the
  optimized variant, but does not select optimized backends (Triton/graphs).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from typing import List, Tuple

from core.harness.benchmark_harness import BaseBenchmark
from labs.persistent_decode.baseline_persistent_decode import BaselinePersistentDecodeBenchmark
from labs.persistent_decode.persistent_decode_common import DecodeOptions, set_decode_options


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Right-sized decode baseline flags", add_help=False)
    p.add_argument("--tier", choices=["small", "medium", "large"], default="medium", help="Decode tier sizing preset")
    p.add_argument("--quantization", choices=["fp32", "fp16", "int4"], default="fp32", help="Decode quantization mode")
    p.add_argument("--block-k", type=int, default=None, help="Override BLOCK_K (defaults depend on tier)")
    p.add_argument("--num-programs", type=int, default=None, help="Override number of Triton programs")
    return p


def _parse_cli() -> Tuple[argparse.Namespace, List[str]]:
    parser = _build_parser()
    args, unknown = parser.parse_known_args()
    return args, unknown


_CLI_ARGS, _CLI_UNKNOWN = _parse_cli()

set_decode_options(
    DecodeOptions(
        tier=_CLI_ARGS.tier,
        quantization=_CLI_ARGS.quantization,
        block_k=_CLI_ARGS.block_k,
        num_programs=_CLI_ARGS.num_programs,
    )
)


def get_benchmark() -> BaseBenchmark:
    return BaselinePersistentDecodeBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
