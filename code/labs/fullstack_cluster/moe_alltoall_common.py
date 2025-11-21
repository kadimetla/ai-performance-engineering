"""Shared utilities for MoE all-to-all readiness probes."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist


def parse_size(text: str) -> int:
    """Parse human-friendly size strings like 64k or 4m into bytes."""
    cleaned = text.strip().lower()
    if cleaned.endswith("k"):
        return int(float(cleaned[:-1]) * 1024)
    if cleaned.endswith("m"):
        return int(float(cleaned[:-1]) * 1024 * 1024)
    if cleaned.endswith("g"):
        return int(float(cleaned[:-1]) * 1024 * 1024 * 1024)
    return int(cleaned)


def parse_size_list(spec: str) -> List[int]:
    return [parse_size(x) for x in spec.split(",") if x.strip()]


def parse_float_list(spec: str) -> List[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f}M"
    return f"{num_bytes / 1024:.0f}K"


def percentile(t: torch.Tensor, q: float) -> float:
    q = max(0.0, min(100.0, q))
    k = (q / 100.0) * (t.numel() - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(t[f].item())
    d0 = t[f] * (c - k)
    d1 = t[c] * (k - f)
    return float((d0 + d1).item())


def make_zipf_counts(world_size: int, total_elems: int, alpha: float) -> List[int]:
    """Zipf-like routing counts that sum to total_elems."""
    ranks = torch.arange(1, world_size + 1, dtype=torch.float64)
    weights = 1.0 / (ranks**alpha)
    probs = weights / weights.sum()
    expected = probs * total_elems
    counts = torch.floor(expected).to(torch.int64)
    remainder = int(total_elems - int(counts.sum()))
    if remainder > 0:
        frac = (expected - counts.to(expected.dtype))
        _, idx = torch.sort(frac, descending=True)
        counts[idx[:remainder]] += 1
    return [int(c.item()) for c in counts]


def gini_coefficient(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    x_sorted, _ = torch.sort(x)
    n = x_sorted.numel()
    index = torch.arange(1, n + 1, dtype=torch.float64, device=x.device)
    x = x_sorted.to(torch.float64)
    gini = (2.0 * (index * x).sum() / (n * x.sum())) - (n + 1.0) / n
    return float(gini)


@dataclass
class AllToAllResult:
    msg_bytes: int
    skew_alpha: float
    gini: float
    max_over_mean: float
    p50_ms: float
    p99_ms: float
    bw_p50_gbps: float
    bw_p99_gbps: float


def init_distributed(backend: str = "nccl") -> Tuple[int, int, int, bool]:
    """Initialize torch.distributed if world_size>1; return (rank, world_size, local_rank, initialized)."""
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(world_size, 1)))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return rank, world_size, local_rank, False

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank, True


def _collect_latencies(latencies_ms: List[float], world_size: int) -> Tuple[float, float]:
    lat_tensor = torch.tensor(latencies_ms, dtype=torch.float32, device="cuda")
    gather_list = [torch.zeros_like(lat_tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, lat_tensor)
    global_lats = torch.cat([t.cpu() for t in gather_list], dim=0)
    global_lats_sorted, _ = torch.sort(global_lats)
    p50 = percentile(global_lats_sorted, 50.0)
    p99 = percentile(global_lats_sorted, 99.0)
    return p50, p99


def run_alltoall_single(
    *,
    msg_bytes: int,
    skew_alpha: float,
    num_iters: int,
    world_size: int,
    dtype: torch.dtype,
) -> AllToAllResult:
    elem_size = torch.tensor([], dtype=dtype).element_size()
    total_elems = msg_bytes // elem_size
    if total_elems == 0:
        raise ValueError(f"msg_bytes {msg_bytes} too small for dtype {dtype}")

    send_counts = make_zipf_counts(world_size, total_elems, skew_alpha)
    recv_counts = send_counts

    send_splits = torch.tensor(send_counts, dtype=torch.int64, device="cuda")
    recv_splits = torch.tensor(recv_counts, dtype=torch.int64, device="cuda")

    send_buf = torch.randn(total_elems, dtype=dtype, device="cuda")
    recv_buf = torch.empty_like(send_buf)

    counts_t = torch.tensor(send_counts, dtype=torch.float32)
    gini = gini_coefficient(counts_t)
    max_over_mean = float(counts_t.max().item() / (counts_t.mean().item() + 1e-6))

    for _ in range(5):
        dist.all_to_all_single(recv_buf, send_buf, recv_splits=recv_splits, send_splits=send_splits)
    torch.cuda.synchronize()

    latencies_ms: List[float] = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_to_all_single(recv_buf, send_buf, recv_splits=recv_splits, send_splits=send_splits)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50, p99 = _collect_latencies(latencies_ms, world_size)
    bw_p50 = (msg_bytes / (p50 / 1000.0)) / 1e9
    bw_p99 = (msg_bytes / (p99 / 1000.0)) / 1e9

    return AllToAllResult(
        msg_bytes=msg_bytes,
        skew_alpha=skew_alpha,
        gini=gini,
        max_over_mean=max_over_mean,
        p50_ms=p50,
        p99_ms=p99,
        bw_p50_gbps=bw_p50,
        bw_p99_gbps=bw_p99,
    )


def dump_json(results: Dict[Tuple[int, float], AllToAllResult], path: Path, *, iters: int, world_size: int) -> None:
    serializable = [asdict(v) for (_, v) in sorted(results.items(), key=lambda kv: (kv[0][0], kv[0][1]))]
    payload = {"world_size": world_size, "iters": iters, "results": serializable}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run_preflight(
    commands: Iterable[Tuple[str, List[str]]],
    *,
    rank: int,
    timeout_s: float = 5.0,
) -> Dict[str, str]:
    """Run shell commands for quick health checks; swallow failures."""
    outputs: Dict[str, str] = {}
    if rank != 0:
        return outputs

    for label, cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
            )
            text = proc.stdout.strip()
            outputs[label] = "\n".join(text.splitlines()[-20:]) if text else ""
        except FileNotFoundError:
            outputs[label] = "missing (binary not found)"
        except subprocess.SubprocessError as exc:
            outputs[label] = f"error: {exc}"
    return outputs


def add_base_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--message-sizes", type=str, default="64k,1m", help="Comma-separated per-rank message sizes.")
    parser.add_argument("--skews", type=str, default="1.0", help="Comma-separated Zipf alpha values.")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations per point.")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Payload dtype.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory for reports.")
    parser.add_argument("--max-p99-ms", type=float, default=6.0, help="Latency budget for pass/fail.")
    parser.add_argument("--min-bw-gbps", type=float, default=15.0, help="Bandwidth floor for pass/fail.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip quick preflight shell checks.")


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[name]


def summarize_results(
    results: Dict[Tuple[int, float], AllToAllResult],
    *,
    max_p99_ms: float,
    min_bw_gbps: float,
    rank: int,
    header: str,
) -> None:
    if rank != 0:
        return
    print(f"\n{header}")
    overall_pass = True
    for (msg_bytes, alpha) in sorted(results.keys(), key=lambda k: (k[0], k[1])):
        r = results[(msg_bytes, alpha)]
        pass_latency = r.p99_ms <= max_p99_ms
        pass_bw = r.bw_p99_gbps >= min_bw_gbps
        passed = pass_latency and pass_bw
        overall_pass = overall_pass and passed
        status = "PASS" if passed else "FAIL"
        size_str = format_size(msg_bytes)
        print(
            f"[{status}] size={size_str:>5} skew={alpha:>4.2f} "
            f"p50={r.p50_ms:6.2f} ms p99={r.p99_ms:6.2f} ms "
            f"BW_p99={r.bw_p99_gbps:6.2f} GB/s Gini={r.gini:5.3f} "
            f"max/mean={r.max_over_mean:5.2f}"
        )
    verdict = "PASS" if overall_pass else "FAIL"
    print(f"\nVerdict: {verdict} (p99 <= {max_p99_ms} ms and BW_p99 >= {min_bw_gbps} GB/s)")


def try_make_heatmaps(
    results: Dict[Tuple[int, float], AllToAllResult],
    output_dir: Path,
    *,
    rank: int,
) -> Optional[List[Path]]:
    """Render heatmaps if matplotlib/numpy exist; return written paths."""
    if rank != 0 or not results:
        return None
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    sizes = sorted({r.msg_bytes for r in results.values()})
    skews = sorted({r.skew_alpha for r in results.values()})
    size_to_idx = {s: i for i, s in enumerate(sizes)}
    skew_to_idx = {a: i for i, a in enumerate(skews)}

    p99_matrix = np.zeros((len(skews), len(sizes)), dtype=float)
    gini_matrix = np.zeros((len(skews), len(sizes)), dtype=float)

    for (msg_bytes, alpha), r in results.items():
        i = skew_to_idx[alpha]
        j = size_to_idx[msg_bytes]
        p99_matrix[i, j] = r.p99_ms
        gini_matrix[i, j] = r.gini

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    def _plot(matrix, title, fname, xlabels, ylabels, xlabel, ylabel):
        plt.figure(figsize=(8, 4))
        im = plt.imshow(matrix, aspect="auto", origin="lower")
        plt.colorbar(im)
        plt.xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
        plt.yticks(range(len(ylabels)), ylabels)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        out_path = output_dir / fname
        plt.savefig(out_path)
        plt.close()
        paths.append(out_path)

    pretty_sizes = [format_size(s) for s in sizes]
    pretty_skews = [f"{a:.2f}" for a in skews]
    _plot(p99_matrix, "All-to-All p99 Latency (ms)", "heatmap_p99_latency.png", pretty_sizes, pretty_skews,
          "Message size per rank", "Skew alpha")
    _plot(gini_matrix, "Send Count Gini (Skew)", "heatmap_gini_skew.png", pretty_sizes, pretty_skews,
          "Message size per rank", "Skew alpha")
    return paths
