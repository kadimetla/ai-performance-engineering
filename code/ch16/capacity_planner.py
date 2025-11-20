"""Little's Law-based GPU capacity planner for Chapter 16 workloads.

This helper consumes observed or target traffic stats (QPS, prompt/gen token
percentiles) along with measured throughput numbers (prefill tokens/sec,
decode tokens/sec, sustained tokens/sec per GPU). It then estimates how many
GPUs are needed to satisfy the workload at different percentiles and how many
to provision once headroom is included.

The CLI can ingest JSON emitted by ``ch16/inference_server_load_test.py`` or
accept metrics directly via command-line flags.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CapacityEstimate:
    """Result for a single percentile."""

    percentile: str
    qps: float
    service_time_prefill_s: float
    service_time_decode_s: float
    total_service_time_s: float
    required_gpus: float
    required_gpus_with_headroom: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class LittleLawCapacityPlanner:
    """Implements the cheat-sheet described in the Chapter 16 writeup."""

    def __init__(
        self,
        *,
        prefill_tokens_per_s: float,
        decode_tokens_per_s: float,
        tokens_per_gpu: float,
        headroom_ratio: float,
    ) -> None:
        self.prefill_tokens_per_s = prefill_tokens_per_s
        self.decode_tokens_per_s = decode_tokens_per_s
        self.tokens_per_gpu = tokens_per_gpu
        self.headroom_ratio = max(0.0, headroom_ratio)

    def _service_time(self, prompt_tokens: float, generated_tokens: float) -> Dict[str, float]:
        prefill = (
            prompt_tokens / self.prefill_tokens_per_s
            if self.prefill_tokens_per_s > 0.0
            else 0.0
        )
        decode = (
            generated_tokens / self.decode_tokens_per_s
            if self.decode_tokens_per_s > 0.0
            else 0.0
        )
        return {
            "prefill": prefill,
            "decode": decode,
            "total": prefill + decode,
        }

    def estimate_percentile(
        self,
        *,
        percentile: str,
        qps: float,
        prompt_tokens: float,
        generated_tokens: float,
    ) -> Optional[CapacityEstimate]:
        if qps <= 0.0 or self.tokens_per_gpu <= 0.0:
            return None
        service = self._service_time(prompt_tokens, generated_tokens)
        total = service["total"]
        if total <= 0.0:
            return None
        required = (qps * total) / self.tokens_per_gpu
        required_with_headroom = required * (1.0 + self.headroom_ratio)
        return CapacityEstimate(
            percentile=percentile,
            qps=qps,
            service_time_prefill_s=service["prefill"],
            service_time_decode_s=service["decode"],
            total_service_time_s=total,
            required_gpus=required,
            required_gpus_with_headroom=required_with_headroom,
        )

    def estimate_all(
        self,
        *,
        qps: float,
        prompt_token_stats: Dict[str, float],
        generated_token_stats: Dict[str, float],
    ) -> List[CapacityEstimate]:
        estimates: List[CapacityEstimate] = []
        for percentile in ("p50", "p95"):
            prompt = prompt_token_stats.get(percentile)
            generated = generated_token_stats.get(percentile)
            if prompt is None or generated is None:
                continue
            est = self.estimate_percentile(
                percentile=percentile,
                qps=qps,
                prompt_tokens=prompt,
                generated_tokens=generated,
            )
            if est is not None:
                estimates.append(est)
        return estimates


def _load_results(path: Path) -> Dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("results JSON must contain an object")
    return data


def _resolve_stats(source: Dict, key: str) -> Dict[str, float]:
    raw = source.get(key, {})
    if not isinstance(raw, dict):
        return {}
    stats = {}
    for percentile in ("p50", "p95"):
        value = raw.get(percentile)
        if value is not None:
            stats[percentile] = float(value)
    return stats


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Little's Law capacity planner")
    parser.add_argument("--results", type=Path, help="Optional JSON from inference_server_load_test.py")
    parser.add_argument("--qps", type=float, help="Observed or target QPS")
    parser.add_argument("--prefill-tokens-per-s", type=float, default=None)
    parser.add_argument("--decode-tokens-per-s", type=float, default=None)
    parser.add_argument(
        "--tokens-per-gpu",
        type=float,
        help="Sustained tokens/sec per GPU (overrides value saved in results JSON)",
    )
    parser.add_argument("--headroom", type=float, default=None)
    parser.add_argument("--installed-gpus", type=float, help="Optional installed GPU count for warnings")
    parser.add_argument("--prompt-p50", type=float, help="Prompt tokens p50 if no results JSON")
    parser.add_argument("--prompt-p95", type=float, help="Prompt tokens p95 if no results JSON")
    parser.add_argument("--gen-p50", type=float, help="Generated tokens p50 if no results JSON")
    parser.add_argument("--gen-p95", type=float, help="Generated tokens p95 if no results JSON")
    return parser


def _stats_from_args(args: argparse.Namespace, existing: Dict[str, float]) -> Dict[str, float]:
    stats = dict(existing)
    if args.prompt_p50 is not None:
        stats["p50"] = float(args.prompt_p50)
    if args.prompt_p95 is not None:
        stats["p95"] = float(args.prompt_p95)
    return stats


def _gen_stats_from_args(args: argparse.Namespace, existing: Dict[str, float]) -> Dict[str, float]:
    stats = dict(existing)
    if args.gen_p50 is not None:
        stats["p50"] = float(args.gen_p50)
    if args.gen_p95 is not None:
        stats["p95"] = float(args.gen_p95)
    return stats


def _resolve_qps(args: argparse.Namespace, data: Optional[Dict]) -> Optional[float]:
    if args.qps is not None:
        return float(args.qps)
    if data is None:
        return None
    cap = data.get("capacity_plan", {})
    qps = cap.get("qps_observed")
    if qps is not None:
        return float(qps)
    completed = data.get("completed_requests")
    elapsed = data.get("elapsed")
    if completed is None or elapsed in (None, 0):
        return None
    return float(completed) / float(elapsed)


def _resolve_tokens_per_gpu(args: argparse.Namespace, data: Optional[Dict]) -> Optional[float]:
    if args.tokens_per_gpu is not None:
        return float(args.tokens_per_gpu)
    if data is None:
        return None
    cap = data.get("capacity_plan", {})
    value = cap.get("tokens_per_gpu")
    if value is not None:
        return float(value)
    throughput = data.get("throughput_tok_per_s")
    world_size = data.get("world_size", 1) or 1
    if throughput is None:
        return None
    return float(throughput) / float(world_size)


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    data = _load_results(args.results) if args.results else None
    prompt_stats = _resolve_stats(data, "prompt_token_stats") if data else {}
    gen_stats = _resolve_stats(data, "generated_token_stats") if data else {}
    prompt_stats = _stats_from_args(args, prompt_stats)
    gen_stats = _gen_stats_from_args(args, gen_stats)

    qps = _resolve_qps(args, data)
    tokens_per_gpu = _resolve_tokens_per_gpu(args, data)
    if qps is None or tokens_per_gpu is None:
        raise SystemExit("QPS and tokens-per-gpu must be provided via --qps/--tokens-per-gpu or the results JSON")

    cap = data.get("capacity_plan") if data else None

    def _resolve_rate(arg_value: Optional[float], cap_key: str, fallback: float) -> float:
        if arg_value is not None:
            return float(arg_value)
        if cap and cap.get(cap_key) is not None:
            return float(cap[cap_key])
        return fallback

    prefill_rate = _resolve_rate(args.prefill_tokens_per_s, "prefill_tokens_per_s", 40000.0)
    decode_rate = _resolve_rate(args.decode_tokens_per_s, "decode_tokens_per_s", 2000.0)
    headroom = _resolve_rate(args.headroom, "headroom_ratio", 0.3)

    planner = LittleLawCapacityPlanner(
        prefill_tokens_per_s=prefill_rate,
        decode_tokens_per_s=decode_rate,
        tokens_per_gpu=tokens_per_gpu,
        headroom_ratio=headroom,
    )
    estimates = planner.estimate_all(
        qps=qps,
        prompt_token_stats=prompt_stats,
        generated_token_stats=gen_stats,
    )
    if not estimates:
        raise SystemExit("Insufficient percentile data to compute capacity. Provide prompt/gen token percentiles.")

    print(
        "Little's Law capacity plan | QPS={:.2f} | Prefill={:.0f} tok/s | Decode={:.0f} tok/s | Tokens/GPU={:.0f} tok/s | Headroom={:.0%}".format(
            qps,
            prefill_rate,
            decode_rate,
            tokens_per_gpu,
            headroom,
        )
    )
    for est in estimates:
        print(
            "  {percentile}: service={service_ms:.1f} ms (prefill={prefill_ms:.1f} ms, decode={decode_ms:.1f} ms)"
            " → GPUs={gpus:.2f}, GPUs+headroom={gpus_headroom:.2f}".format(
                percentile=est.percentile.upper(),
                service_ms=est.total_service_time_s * 1000.0,
                prefill_ms=est.service_time_prefill_s * 1000.0,
                decode_ms=est.service_time_decode_s * 1000.0,
                gpus=est.required_gpus,
                gpus_headroom=est.required_gpus_with_headroom,
            )
        )

    if args.installed_gpus is not None:
        for est in estimates:
            if est.required_gpus_with_headroom > args.installed_gpus:
                print(
                    "  ⚠ {} requirement ({:.2f}) exceeds installed GPU count ({:.2f}).".format(
                        est.percentile.upper(),
                        est.required_gpus_with_headroom,
                        args.installed_gpus,
                    )
                )


if __name__ == "__main__":
    main()
