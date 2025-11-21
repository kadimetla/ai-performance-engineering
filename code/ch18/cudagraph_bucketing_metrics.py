"""Prometheus counters for the CUDA graph bucketing simulator."""

from __future__ import annotations

from typing import Optional

try:
    from prometheus_client import Counter, start_http_server  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Counter = None  # type: ignore
    start_http_server = None  # type: ignore

from ch18.cudagraph_bucketing_common import GraphStats


CAPTURE_COUNTER_NAME = "cudagraph_sim_captures_total"
REPLAY_COUNTER_NAME = "cudagraph_sim_replays_total"

if Counter is not None:
    CG_CAPTURE_COUNTER = Counter(
        CAPTURE_COUNTER_NAME,
        "Simulated CUDA graph captures by padded batch + seq bucket",
        ["region", "model", "padded_bs", "seqlen"],
    )
    CG_REPLAY_COUNTER = Counter(
        REPLAY_COUNTER_NAME,
        "Simulated CUDA graph replays by padded batch + seq bucket",
        ["region", "model", "padded_bs", "seqlen"],
    )
else:  # pragma: no cover - optional dependency fallback
    CG_CAPTURE_COUNTER = None
    CG_REPLAY_COUNTER = None


def export_stats_to_prometheus(
    stats: GraphStats,
    *,
    region: str,
    model: str,
    start_port: Optional[int] = None,
) -> None:
    """
    Push GraphStats into Prometheus counters. Optionally start an HTTP server.
    No-op when prometheus_client is unavailable.
    """
    if CG_CAPTURE_COUNTER is None or CG_REPLAY_COUNTER is None:
        return

    if start_port is not None and start_http_server is not None:
        start_http_server(start_port)

    for (padded_bs, seqlen), count in stats.captures_per_key.items():
        CG_CAPTURE_COUNTER.labels(
            region=region,
            model=model,
            padded_bs=str(padded_bs),
            seqlen=str(seqlen),
        ).inc(count)

    for (padded_bs, seqlen), count in stats.replays_per_key.items():
        CG_REPLAY_COUNTER.labels(
            region=region,
            model=model,
            padded_bs=str(padded_bs),
            seqlen=str(seqlen),
        ).inc(count)
