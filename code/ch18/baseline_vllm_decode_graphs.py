"""Baseline decode loop that triggers CUDA graph churn, allocator growth, and eager KV compaction."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
import threading
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ch18.decode_kernels import DEVICE, build_decode_kernel  # noqa: E402


def default_trace(num_steps: int = 24, seed: int = 0) -> List[int]:
    """Generate a ragged decode schedule to mimic continuous batching."""
    rng = random.Random(seed)
    candidates = (3, 5, 7, 9, 12, 15, 18, 24, 28)
    return [rng.choice(candidates) for _ in range(num_steps)]


@dataclass
class KVBlock:
    capacity: int
    used: int = 0

    @property
    def free(self) -> int:
        return max(0, self.capacity - self.used)

    def fill(self, tokens: int) -> None:
        self.used = min(self.capacity, self.used + tokens)

    def compact(self) -> None:
        self.used = 0


class NaiveKVLayout:
    """Compacts every step, which models the "micro-motion" overhead."""

    def __init__(self, blocks: Sequence[int] = (64, 64, 128)) -> None:
        self.blocks: List[KVBlock] = [KVBlock(b) for b in blocks]

    def simulate_step(self, tokens_written: int) -> int:
        """Write tokens into blocks, then compact all blocks eagerly."""
        compactions = 0
        for block in self.blocks:
            block.fill(tokens_written)
            block.compact()
            compactions += 1
        return compactions


@dataclass
class DecodeMetrics:
    steps: int = 0
    tokens: int = 0
    graph_recaptures: int = 0
    allocator_bytes: int = 0
    compactions: int = 0


class BaselineDecodeDriver:
    """
    Decode loop without buckets or preallocation.

    Every ragged batch forces a new graph capture, fresh allocator activity, and
    aggressive KV compaction.
    """

    def __init__(self, trace: Iterable[int] | None = None, hidden: int = 128) -> None:
        self.trace = list(trace) if trace is not None else default_trace()
        self.hidden = hidden
        self.decode_kernel = build_decode_kernel(hidden=self.hidden, max_batch=max(self.trace or [1]))
        self.kv_layout = NaiveKVLayout()
        self.captured_shapes: set[Tuple[int, int]] = set()

    def run(self) -> DecodeMetrics:
        metrics = DecodeMetrics()
        dtype = torch.float16 if getattr(self.decode_kernel, "backend", "") == "vllm" else torch.float32
        for batch_size in self.trace:
            tokens = torch.randn(batch_size, self.hidden, device=DEVICE, dtype=dtype)
            kv = torch.randn(batch_size, self.hidden, device=DEVICE, dtype=dtype)
            logits = self.decode_kernel(tokens, kv, None)

            shape_key = (logits.shape[0], logits.shape[1])
            if shape_key not in self.captured_shapes:
                metrics.graph_recaptures += 1
                self.captured_shapes.add(shape_key)

            metrics.allocator_bytes += logits.numel() * logits.element_size()
            metrics.compactions += self.kv_layout.simulate_step(tokens_written=batch_size)
            metrics.tokens += batch_size
            metrics.steps += 1

        return metrics


def format_metrics(label: str, metrics: DecodeMetrics, backend: str = "torch") -> str:
    mb = metrics.allocator_bytes / (1024 * 1024)
    return (
        f"[{label}] backend={backend}, steps={metrics.steps}, tokens={metrics.tokens}, "
        f"graph_captures={metrics.graph_recaptures}, allocator_mb={mb:.1f}, "
        f"compactions={metrics.compactions}"
    )


def export_prom_metrics(
    label: str,
    metrics: DecodeMetrics,
    backend: str,
    port: int,
    duration_s: int,
) -> None:
    """Expose graph/allocator counters alongside vLLM metrics naming style."""
    try:
        from prometheus_client import Counter, Gauge, start_http_server
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[warn] prometheus_client unavailable, skipping export: {exc}")
        return

    graph_recaptures = Counter(
        "vllm:decode_graph_recaptures_total",
        "Decode graph recaptures (bucket drift) observed by the demo driver.",
        ["variant", "backend"],
    )
    allocator_bytes = Gauge(
        "vllm:decode_allocator_bytes",
        "Bytes attributed to decode workspaces in the demo driver.",
        ["variant", "backend"],
    )
    kv_compactions = Counter(
        "vllm:decode_kv_compactions_total",
        "Number of KV compactions triggered by the demo driver.",
        ["variant", "backend"],
    )

    start_http_server(port)
    graph_recaptures.labels(variant=label, backend=backend).inc(metrics.graph_recaptures)
    allocator_bytes.labels(variant=label, backend=backend).set(metrics.allocator_bytes)
    kv_compactions.labels(variant=label, backend=backend).inc(metrics.compactions)

    if duration_s > 0:
        print(f"[metrics] exporting on :{port} for {duration_s}s")
        threading.Event().wait(timeout=duration_s)
        print(f"[metrics] Prometheus window elapsed on :{port}")
        return

    print(f"[metrics] exported once on :{port} (process will exit)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline ragged decode loop (no buckets/prealloc).")
    parser.add_argument("--steps", type=int, default=24, help="Decode iterations to run.")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size for the mock decode op.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the ragged trace.")
    parser.add_argument("--prom-port", type=int, default=None, help="Optional Prometheus port to export metrics.")
    parser.add_argument(
        "--prom-duration",
        type=int,
        default=0,
        help="If --prom-port is set, keep the server alive this many seconds (0 = fire-and-exit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = default_trace(num_steps=args.steps, seed=args.seed)
    driver = BaselineDecodeDriver(trace=trace, hidden=args.hidden)
    metrics = driver.run()
    backend = getattr(driver.decode_kernel, "backend", "torch")
    print(format_metrics("baseline", metrics, backend=backend))

    if args.prom_port is not None:
        export_prom_metrics("baseline", metrics, backend=backend, port=args.prom_port, duration_s=args.prom_duration)


if __name__ == "__main__":
    main()
