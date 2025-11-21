"""Optimized prefill vs. decode microbench with simple TMA burst shaping.

What it demonstrates for Nsight Systems:
- Prefill: double-buffered-ish pipeline using multiple streams + max_in_flight
  guard to show how shaping reduces contention.
- Decode: graph-captured token loop to trim host launch gaps.
"""

from __future__ import annotations

from enum import Enum
import os

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    get_stream_priorities,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)
from common.python.blackwell_requirements import ensure_blackwell_tma_supported


class GraphMode(Enum):
    FULL = "full"
    PIECEWISE = "piecewise"
    FULL_AND_PIECEWISE = "full_and_piecewise"

    @classmethod
    def from_str(cls, raw: str | None) -> "GraphMode":
        normalized = (raw or cls.FULL_AND_PIECEWISE.value).strip().lower().replace("-", "_")
        for mode in cls:
            if normalized == mode.value:
                return mode
        return cls.FULL_AND_PIECEWISE


class TmaBurstConfig:
    """Simple config holder for burst shaping knobs."""

    def __init__(self, chunk_k: int = 128, max_in_flight: int = 2, tma_sleep_cycles: int = 50_000) -> None:
        self.chunk_k = chunk_k
        self.max_in_flight = max_in_flight
        # torch.cuda._sleep argument is in cycles; keep high enough to visualize overlap.
        self.tma_sleep_cycles = tma_sleep_cycles


class OptimizedTmaPrefillDecodeBenchmark(BaseBenchmark):
    """Prefill with shaped pseudo-TMA + decode hosted in a CUDA Graph."""

    def __init__(self, *, graph_mode: "GraphMode | None" = None, max_capture_seq: int | None = None) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.prefill_chunks = 8
        self.prefill_chunk_elems = 128 * 128
        self.cfg = TmaBurstConfig()
        self._prio_low, self._prio_high = get_stream_priorities()
        self.prefill_streams = [torch.cuda.Stream(priority=self._prio_low) for _ in range(self.cfg.max_in_flight)]
        self.decode_stream = torch.cuda.Stream(priority=self._prio_high)
        self.decode_graph = torch.cuda.CUDAGraph()
        self.full_graph: torch.cuda.CUDAGraph | None = None
        self.graph_q = None
        self.graph_k = None
        self.graph_v = None
        self.graph_out = None
        self.graph_mode = graph_mode or GraphMode.from_str(os.getenv("PD_GRAPH_MODE"))
        self.max_capture_seq = max_capture_seq or int(os.getenv("PD_MAX_CAPTURE_SEQ", self.seq_len))
        self._history: dict[str, list[float]] = {}
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        ensure_blackwell_tma_supported("optimized_tma_prefill_decode")
        self.inputs = build_inputs(self.device)
        # Skip on GPUs without TMA support to avoid false regressions.
        if not torch.cuda.get_device_capability(self.device) >= (12, 0):
            raise RuntimeError("SKIP: TMA not supported on this GPU")
        self.prefill_src = torch.randn(
            self.prefill_chunks, self.prefill_chunk_elems, device=self.device
        )
        self.prefill_dst = torch.zeros_like(self.prefill_src)

        # Graph-captured decode loop to cut host gaps during profiling.
        self.graph_q = self.inputs.q.clone()
        self.graph_k = self.inputs.k.clone()
        self.graph_v = self.inputs.v.clone()
        self.graph_out = torch.zeros_like(self.inputs.out)

        torch.cuda.synchronize()
        with torch.cuda.graph(self.decode_graph, stream=self.decode_stream):
            self._decode_body(self.graph_q, self.graph_k, self.graph_v, self.graph_out)
        torch.cuda.synchronize()
        self._capture_full_graph()

    def _capture_full_graph(self) -> None:
        if self.graph_mode == GraphMode.PIECEWISE:
            return
        self.full_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.full_graph, stream=self.decode_stream):
            # Simplified full-iteration capture: single-stream prefill analog + captured decode.
            torch.cuda._sleep(self.cfg.tma_sleep_cycles)
            self.prefill_dst.copy_(self.prefill_src)
            self._decode_body(self.graph_q, self.graph_k, self.graph_v, self.graph_out)
            self.inputs.out.copy_(self.graph_out)
        torch.cuda.synchronize()

    def _decode_body(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor
    ) -> None:
        # Simple per-token dot product.
        for t in range(self.seq_len):
            q_t = q[:, t, :]
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            dot = (q_t * k_t).sum(dim=-1, keepdim=True)
            out[:, t, :] = v_t * dot

    def _prefill_shaped(self, *, async_only: bool = False) -> list[torch.cuda.Event] | None:
        """Launch pseudo-TMA copies on multiple streams with a max_in_flight cap."""
        events = []
        for idx in range(self.prefill_chunks):
            stream = self.prefill_streams[idx % len(self.prefill_streams)]
            with torch.cuda.stream(stream):
                torch.cuda._sleep(self.cfg.tma_sleep_cycles)
                self.prefill_dst[idx].add_(self.prefill_src[idx])
            evt = torch.cuda.Event(enable_timing=False, blocking=False)
            evt.record(stream)
            events.append(evt)
            if len(events) > self.cfg.max_in_flight:
                events.pop(0).synchronize()

        if async_only:
            return events

        # Drain remaining work.
        for evt in events:
            evt.synchronize()
        return None

    def _decode_graph(self) -> None:
        assert self.inputs is not None
        # Refresh graph inputs to show a realistic copy-before-replay pattern.
        with torch.cuda.stream(self.decode_stream):
            self.graph_q.copy_(self.inputs.q)
            self.graph_k.copy_(self.inputs.k)
            self.graph_v.copy_(self.inputs.v)
            self.graph_out.zero_()
            self.decode_graph.replay()
            # Mirror back to inputs.out so validation stays consistent.
            self.inputs.out.copy_(self.graph_out)

    def benchmark_fn(self) -> None:
        if self.inputs is None:
            raise RuntimeError("Inputs not initialized")

        use_full = (
            self.graph_mode == GraphMode.FULL
            or (self.graph_mode == GraphMode.FULL_AND_PIECEWISE and self.seq_len <= self.max_capture_seq)
        )
        if use_full and self.full_graph is not None:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with self._nvtx_range("full_graph_high_pri"):
                with torch.cuda.stream(self.decode_stream):
                    start.record(self.decode_stream)
                    self.full_graph.replay()
                    end.record(self.decode_stream)
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
            self._history.setdefault("ttft_ms", []).append(total_ms)
            self._history.setdefault("decode_ms", []).append(total_ms)
            self._history.setdefault("per_token_ms", []).append(total_ms / max(1, self.seq_len))
            self._history.setdefault("graph_path", []).append("full_graph")
            return

        with self._nvtx_range("prefill_shaped_low_pri"):
            start_prefill = torch.cuda.Event(enable_timing=True)
            end_prefill = torch.cuda.Event(enable_timing=True)
            start_prefill.record()
            pref_events = self._prefill_shaped(async_only=True)
        with self._nvtx_range(
            "decode_graph_high_pri" if self.graph_mode != GraphMode.FULL_AND_PIECEWISE else "graph_fallback_piecewise"
        ):
            start_decode = torch.cuda.Event(enable_timing=True)
            end_decode = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self.decode_stream):
                start_decode.record(self.decode_stream)
                self._decode_graph()
                end_decode.record(self.decode_stream)
        if pref_events:
            for evt in pref_events:
                evt.synchronize()
        end_prefill.record()
        self._synchronize()
        torch.cuda.synchronize()
        ttft_ms = start_prefill.elapsed_time(end_prefill)
        decode_ms = start_decode.elapsed_time(end_decode)
        self._history.setdefault("ttft_ms", []).append(ttft_ms)
        self._history.setdefault("decode_ms", []).append(decode_ms)
        self._history.setdefault("per_token_ms", []).append(decode_ms / max(1, self.seq_len))
        self._history.setdefault("graph_path", []).append("piecewise_graph")

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None
        self.full_graph = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=8, warmup=2)

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedTmaPrefillDecodeBenchmark()
