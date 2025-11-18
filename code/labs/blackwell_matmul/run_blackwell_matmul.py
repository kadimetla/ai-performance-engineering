"""Quick CLI to benchmark the Blackwell matmul variants."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.blackwell_matmul import is_cluster_launch_supported
from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

_VARIANT_MODULES = {
    "baseline": "labs.blackwell_matmul.baseline_blackwell_matmul",
    "tma": "labs.blackwell_matmul.optimized_blackwell_matmul_tma",
    "pipeline": "labs.blackwell_matmul.optimized_blackwell_matmul_pipeline",
    "cluster": "labs.blackwell_matmul.optimized_blackwell_matmul_cluster",
}


def _ensure_cluster_available(requested_variant: str) -> None:
    if requested_variant != "cluster":
        return
    if is_cluster_launch_supported():
        return
    raise SystemExit(
        "[labs/blackwell_matmul] Cluster launch unsupported on this GPU."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(_VARIANT_MODULES.keys()),
        default="baseline",
        help="Which Modular.org part to replay",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2048,
        help="Square dimension for the GEMM (must be divisible by 64)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override BenchmarkConfig.iterations",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override BenchmarkConfig.warmup",
    )
    args = parser.parse_args(argv)

    module_name = _VARIANT_MODULES[args.variant]
    module = importlib.import_module(module_name)
    benchmark = module.get_benchmark(size=args.size)
    config = benchmark.get_config()
    if args.iterations is not None and config is not None:
        config.iterations = args.iterations
    if args.warmup is not None and config is not None:
        config.warmup = args.warmup
    if config is not None:
        config.setup_timeout_seconds = max(180, config.setup_timeout_seconds or 0)
        config.measurement_timeout_seconds = max(180, config.measurement_timeout_seconds)
        config.profiling_timeout_seconds = max(300, config.profiling_timeout_seconds or 0)
        config.ncu_timeout_seconds = max(300, config.ncu_timeout_seconds)
        config.nsys_timeout_seconds = max(300, config.nsys_timeout_seconds)

    _ensure_cluster_available(args.variant)

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(
        f"\nBlackwell matmul ({args.variant}, size={args.size}) : {mean_ms:.3f} ms"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - manual tool
    raise SystemExit(main())
