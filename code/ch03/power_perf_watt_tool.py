#!/usr/bin/env python3
"""Estimate GEMM performance per watt using NVML power sampling."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.utils.power_sampling import PowerSampler, ensure_nvml_initialized

try:
    import pynvml  # type: ignore
except ImportError as exc:  # pragma: no cover - required dependency
    raise RuntimeError(
        "power_perf_watt_tool requires pynvml (nvidia-ml-py) when CUDA is available."
    ) from exc


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {sorted(mapping)}.")
    return mapping[name]


def _set_power_limit(handle, limit_watts: float) -> None:
    min_limit, max_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
    limit_mw = int(limit_watts * 1000)
    if limit_mw < min_limit or limit_mw > max_limit:
        raise ValueError(
            f"Requested power limit {limit_watts:.1f} W outside supported range "
            f"{min_limit / 1000:.1f}-{max_limit / 1000:.1f} W."
        )
    pynvml.nvmlDeviceSetPowerManagementLimit(handle, limit_mw)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for perf-per-watt estimation.")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    dtype = _parse_dtype(args.dtype)
    a = torch.randn(args.m, args.k, device=device, dtype=dtype)
    b = torch.randn(args.k, args.n, device=device, dtype=dtype)

    ensure_nvml_initialized()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    original_limit: Optional[int] = None
    if args.power_limit_watts is not None:
        original_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        _set_power_limit(handle, args.power_limit_watts)

    sampler: Optional[PowerSampler] = None
    try:
        sampler = PowerSampler([args.device], interval=args.sample_interval)
        for _ in range(args.warmup):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize(device)

        sampler.start()
        start = time.perf_counter()
        for _ in range(args.iterations):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize(device)
        elapsed_s = time.perf_counter() - start
        power = sampler.stop()
    finally:
        if original_limit is not None:
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, int(original_limit))
        if sampler is not None:
            sampler.close()

    flops_per_iter = 2.0 * args.m * args.n * args.k
    total_flops = flops_per_iter * args.iterations
    tflops = (total_flops / elapsed_s) / 1e12
    avg_power = float(power["avg_watts"])
    perf_per_watt = (total_flops / elapsed_s) / avg_power

    print("Perf-per-watt report")
    print(f"  Device: cuda:{args.device}")
    print(f"  GEMM: {args.m}x{args.k} @ {args.k}x{args.n} ({args.dtype})")
    print(f"  Iterations: {args.iterations} (warmup {args.warmup})")
    print(f"  Time: {elapsed_s:.4f} s")
    print(f"  Throughput: {tflops:.2f} TFLOP/s")
    print(f"  Power: avg {avg_power:.1f} W, max {float(power['max_watts']):.1f} W")
    print(f"  Perf/Watt: {perf_per_watt / 1e12:.3f} TFLOP/J")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate GEMM perf-per-watt via NVML sampling.")
    parser.add_argument("--m", type=int, default=4096, help="GEMM M dimension.")
    parser.add_argument("--n", type=int, default=4096, help="GEMM N dimension.")
    parser.add_argument("--k", type=int, default=4096, help="GEMM K dimension.")
    parser.add_argument("--dtype", type=str, default="fp16", help="Data type: fp16, bf16, fp32.")
    parser.add_argument("--iterations", type=int, default=50, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.1,
        help="Power sampling interval in seconds.",
    )
    parser.add_argument(
        "--power-limit-watts",
        type=float,
        default=None,
        help="Optional power limit to apply during the run.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
