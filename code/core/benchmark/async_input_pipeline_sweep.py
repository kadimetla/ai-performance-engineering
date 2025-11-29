#!/usr/bin/env python3
"""Sweep pin_memory/non_blocking/num_workers for async input pipelines."""

from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from pathlib import Path
from typing import Iterable, List

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(f"PyTorch is required to run this sweep: {exc}") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.common.async_input_pipeline import AsyncInputPipelineBenchmark, PipelineConfig


def _parse_bool_list(values: Iterable[str], default: List[bool]) -> List[bool]:
    parsed: List[bool] = []
    for value in values:
        val = value.strip().lower()
        if val in {"1", "true", "t", "yes", "y"}:
            parsed.append(True)
        elif val in {"0", "false", "f", "no", "n"}:
            parsed.append(False)
        elif val:
            raise argparse.ArgumentTypeError(f"Invalid boolean '{value}'")
    return parsed or default


def _parse_int_list(value: str, default: List[int]) -> List[int]:
    tokens = [tok.strip() for tok in value.split(",") if tok.strip()]
    if not tokens:
        return default
    result: List[int] = []
    for tok in tokens:
        try:
            result.append(int(tok))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid integer '{tok}'") from exc
    return result


def run_trial(cfg: PipelineConfig, iterations: int, warmup: int, use_copy_stream: bool) -> dict:
    cfg = PipelineConfig(**{**cfg.__dict__, "use_copy_stream": use_copy_stream})
    bench = AsyncInputPipelineBenchmark(cfg, label="async_input_sweep")
    bench.setup()

    try:
        for _ in range(warmup):
            bench.benchmark_fn()
        torch.cuda.synchronize(bench.device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times_ms: List[float] = []

        for _ in range(iterations):
            start_event.record()
            bench.benchmark_fn()
            end_event.record()
            end_event.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))

        mean_ms = statistics.mean(times_ms)
        median_ms = statistics.median(times_ms)
        samples_per_s = cfg.batch_size / (mean_ms / 1000.0)
        return {
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "samples_per_s": samples_per_s,
        }
    finally:
        bench.teardown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for the synthetic pipeline (default: 128)",
    )
    parser.add_argument(
        "--feature-shape",
        type=str,
        default="3,224,224",
        help="Feature shape as comma-separated dims (default: 3,224,224)",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=512,
        help="Number of samples in the synthetic dataset (default: 512)",
    )
    parser.add_argument(
        "--num-workers",
        type=str,
        default="0,4,8",
        help="Comma-separated worker counts to sweep (default: 0,4,8)",
    )
    parser.add_argument(
        "--pin-memory",
        type=str,
        nargs="*",
        default=["false", "true"],
        help="Boolean values to sweep for pin_memory (default: false true)",
    )
    parser.add_argument(
        "--non-blocking",
        type=str,
        nargs="*",
        default=["false", "true"],
        help="Boolean values to sweep for non_blocking (default: false true)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Prefetch factor when num_workers > 0 (default: 4)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Measurement iterations per config (default: 50)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations per config (default: 10)",
    )
    parser.add_argument(
        "--copy-stream",
        action="store_true",
        help="Stage H2D transfers on a dedicated copy stream when non_blocking is enabled",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this sweep.")

    try:
        c, h, w = (int(x) for x in args.feature_shape.split(","))
    except ValueError as exc:
        raise SystemExit(f"Invalid --feature-shape '{args.feature_shape}': {exc}") from exc

    pin_memory_vals = _parse_bool_list(args.pin_memory, default=[False, True])
    non_blocking_vals = _parse_bool_list(args.non_blocking, default=[False, True])
    worker_vals = _parse_int_list(args.num_workers, default=[0, 4, 8])

    combos = itertools.product(pin_memory_vals, non_blocking_vals, worker_vals)
    results = []

    print("pin_memory non_blocking workers copy_stream mean_ms median_ms samples/s")
    for pin_mem, non_block, workers in combos:
        cfg = PipelineConfig(
            batch_size=args.batch_size,
            feature_shape=(c, h, w),
            dataset_size=args.dataset_size,
            num_workers=workers,
            prefetch_factor=args.prefetch_factor if workers > 0 else None,
            pin_memory=pin_mem,
            non_blocking=non_block,
            use_copy_stream=args.copy_stream and non_block,
        )
        metrics = run_trial(cfg, iterations=args.iterations, warmup=args.warmup, use_copy_stream=cfg.use_copy_stream)
        results.append((cfg, metrics))
        print(
            f"{str(pin_mem):>9} {str(non_block):>12} {workers:7d} {str(cfg.use_copy_stream):>11} "
            f"{metrics['mean_ms']:7.3f} {metrics['median_ms']:9.3f} {metrics['samples_per_s']:10.1f}"
        )


if __name__ == "__main__":
    main()
