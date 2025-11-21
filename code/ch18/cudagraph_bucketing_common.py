"""Shared helpers for CUDA Graph Tree bucketing demos."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Default capture sizes mimic vLLM's descending cudagraph buckets.
DEFAULT_CAPTURE_BATCH_SIZES: List[int] = [1, 2, 4, 8, 16, 32]
DEFAULT_BATCH_BUCKETS: List[int] = [1, 2, 4, 8, 16, 32]
DEFAULT_SEQLEN_BUCKETS: List[int] = [64, 128, 256, 512, 1024]

# A small, skewed decode traffic pattern: mixed batches and lengths with repeats.
DEFAULT_DECODE_TRAFFIC: List[Tuple[int, int]] = [
    (1, 72),
    (2, 96),
    (4, 192),
    (8, 192),
    (16, 256),
    (4, 220),
    (8, 260),
    (3, 64),
    (6, 144),
    (12, 320),
    (4, 192),
    (8, 192),
    (2, 140),
    (1, 88),
]


@dataclass(frozen=True)
class BucketBands:
    batch_buckets: Sequence[int]
    seqlen_buckets: Sequence[int]

    def bucket(self, batch: int, seqlen: int) -> Tuple[int, int]:
        return self._round(batch, self.batch_buckets), self._round(seqlen, self.seqlen_buckets)

    @staticmethod
    def _round(value: int, bands: Sequence[int]) -> int:
        if not bands:
            return value
        for band in bands:
            if value <= band:
                return band
        return bands[-1]


@dataclass
class GraphStats:
    captures: int = 0
    prewarm_captures: int = 0
    replays: int = 0
    skipped: int = 0
    key_hits: Counter = field(default_factory=Counter)
    captures_per_key: Counter = field(default_factory=Counter)
    replays_per_key: Counter = field(default_factory=Counter)

    def record_capture(self, key: Tuple[int, int], *, prewarm: bool) -> None:
        self.captures += 1
        if prewarm:
            self.prewarm_captures += 1
        self.key_hits[key] += 1
        self.captures_per_key[key] += 1

    def record_replay(self, key: Tuple[int, int]) -> None:
        self.replays += 1
        self.key_hits[key] += 1
        self.replays_per_key[key] += 1

    def record_skip(self) -> None:
        self.skipped += 1

    def summary(self) -> Dict[str, object]:
        hot = self.key_hits.most_common(6)
        return {
            "captures": self.captures,
            "prewarm_captures": self.prewarm_captures,
            "replays": self.replays,
            "skipped": self.skipped,
            "unique_keys": len(self.key_hits),
            "hot_keys": [(str(k), v) for k, v in hot],
        }


class GraphTreeSimulator:
    """
    Lightweight stand-in for CUDA Graph Trees.

    It buckets shapes, rounds batch sizes to graph capture bins, and counts
    capture vs replay events so we can see how many graphs a decode workload
    would stabilize on.
    """

    def __init__(
        self,
        *,
        bucket_bands: BucketBands,
        capture_batch_sizes: Sequence[int],
        name: str,
        pad_fn: Callable[[int], int | None] | None = None,
    ) -> None:
        self.bands = bucket_bands
        self.capture_batch_sizes = sorted(set(int(x) for x in capture_batch_sizes))
        self.name = name
        self.stats = GraphStats()
        self._seen: set[Tuple[int, int]] = set()
        self._pad_fn = pad_fn

    def _pad_batch(self, batch: int) -> int | None:
        return pad_batch_to_capture(batch, self.capture_batch_sizes, self._pad_fn)

    def observe(self, batch: int, seqlen: int, *, prewarm: bool = False) -> None:
        b_bucket, s_bucket = self.bands.bucket(batch, seqlen)
        padded_batch = self._pad_batch(b_bucket)
        if padded_batch is None:
            self.stats.record_skip()
            return
        key = (padded_batch, s_bucket)
        if key in self._seen:
            self.stats.record_replay(key)
            return
        self._seen.add(key)
        self.stats.record_capture(key, prewarm=prewarm)

    def prewarm(self, shapes: Iterable[Tuple[int, int]]) -> None:
        for batch, seqlen in shapes:
            self.observe(batch, seqlen, prewarm=True)

    def run(self, traffic: Iterable[Tuple[int, int]]) -> GraphStats:
        for batch, seqlen in traffic:
            self.observe(batch, seqlen, prewarm=False)
        return self.stats

    def format_summary(self) -> str:
        summary = self.stats.summary()
        lines = [
            f"[{self.name}] captures={summary['captures']} "
            f"(prewarm={summary['prewarm_captures']}), "
            f"replays={summary['replays']}, skipped={summary['skipped']}, "
            f"unique_keys={summary['unique_keys']}",
        ]
        hot_keys = summary["hot_keys"]
        if hot_keys:
            hot_str = ", ".join(f"{k}: {v}" for k, v in hot_keys)  # type: ignore[arg-type]
            lines.append(f"[{self.name}] hot keys -> {hot_str}")
        return "\n".join(lines)


def default_bucket_bands() -> BucketBands:
    return BucketBands(batch_buckets=DEFAULT_BATCH_BUCKETS, seqlen_buckets=DEFAULT_SEQLEN_BUCKETS)


def demo_traffic() -> List[Tuple[int, int]]:
    return list(DEFAULT_DECODE_TRAFFIC)


def capture_bins_from_vllm_config(vllm_config: object) -> List[int]:
    """
    Extract cudagraph capture batch sizes from a VllmConfig instance.
    Falls back to DEFAULT_CAPTURE_BATCH_SIZES if unavailable.
    """
    try:
        comp_cfg = getattr(vllm_config, "compilation_config", None)
        sizes = getattr(comp_cfg, "cudagraph_capture_sizes", None)
        if sizes:
            return list(sizes)
    except Exception:
        pass
    return list(DEFAULT_CAPTURE_BATCH_SIZES)


def pad_fn_from_vllm_config(vllm_config: object) -> Callable[[int], int | None] | None:
    """
    Return VllmConfig.pad_for_cudagraph if present; otherwise None.
    """
    pad_fn = getattr(vllm_config, "pad_for_cudagraph", None)
    if callable(pad_fn):
        return pad_fn
    return None


def pad_batch_to_capture(
    batch: int,
    capture_sizes: Sequence[int],
    pad_fn: Callable[[int], int | None] | None = None,
) -> int | None:
    """
    Pad/round a batch size to the next capture size (or pad_for_cudagraph).
    Returns None if no capture size can hold the batch.
    """
    if pad_fn is not None:
        try:
            return pad_fn(batch)
        except Exception:
            pass
    for size in sorted(set(int(x) for x in capture_sizes)):
        if batch <= size:
            return size
    return None


def load_vllm_config(model_name: Optional[str]) -> object | None:
    """
    Construct a VllmConfig via EngineArgs using the provided model name.
    Returns None if vLLM is unavailable or the config cannot be created.
    """
    if not model_name:
        return None
    try:
        from vllm.engine.arg_utils import EngineArgs  # type: ignore
    except Exception:
        return None

    try:
        engine_args = EngineArgs(
            model=model_name,
            tensor_parallel_size=1,
            enforce_eager=False,
        )
        return engine_args.create_engine_config()
    except Exception:
        return None
