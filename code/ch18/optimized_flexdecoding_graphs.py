"""FlexDecoding decode path wrapped in CUDA Graphs for lower launch overhead."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch18.baseline_flexdecoding import FlexDecodingHarness  # noqa: E402


class OptimizedFlexDecodingGraphsBenchmark(FlexDecodingHarness):
    """Capture a single-token decode in a CUDA Graph and replay per token."""

    def __init__(self) -> None:
        super().__init__(
            use_flex_attention=False,
            require_flex=False,
            decode_tokens=512,
            compile_enabled=False,
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_stream: torch.cuda.Stream | None = None
        self.static_decode_in: torch.Tensor | None = None
        self.static_decode_out: torch.Tensor | None = None
        self.base_position: int = 0

    def _run_warmup(self) -> None:
        """Compile and warm kernels before capture."""
        if self.model is None or self.prefill_tokens is None or self.decode_token is None:
            raise RuntimeError("Model/tokens not initialized")
        with torch.inference_mode():
            self.model.prefill(self.prefill_tokens)
            _ = self.model.decode(self.decode_token, self.base_position)
        torch.cuda.synchronize(self.device)

    def setup(self) -> None:
        self._initialize_and_capture()

    def _initialize_and_capture(self) -> None:
        super().setup()
        if self.model is None or self.prefill_tokens is None or self.decode_token is None:
            raise RuntimeError("Model/tokens not initialized")

        self.base_position = self.prefill_tokens.size(1)
        self.static_decode_in = torch.zeros_like(self.decode_token)
        self.static_decode_out = torch.empty_like(self.decode_token)
        self.capture_stream = torch.cuda.Stream(device=self.device)

        # Compile/warm outside capture to avoid lazy compile during graph capture.
        self._run_warmup()

        self.graph = torch.cuda.CUDAGraph()
        assert self.capture_stream is not None
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            if self.model is None:
                raise RuntimeError("Model not initialized for capture")
            q = self.model.q_proj(self.static_decode_in).view(
                self.static_decode_in.size(0),
                1,
                self.model.cfg.heads,
                self.model.head_dim,
            )
            out = self.model.decode_attention(q)
            self.static_decode_out.copy_(out)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if (
            self.model is None
            or self.prefill_tokens is None
            or self.decode_token is None
            or self.graph is None
            or self.capture_stream is None
            or self.static_decode_in is None
            or self.static_decode_out is None
        ):
            raise RuntimeError("Graph path not initialized")

        prefill_times: List[float] = []
        decode_times: List[float] = []

        self.model.clear_cache(batch=self.prefill_tokens.size(0))

        with torch.no_grad():
            with self._nvtx_range("flex_prefill"):
                start = time.perf_counter()
                _ = self.model.prefill(self.prefill_tokens)
                torch.cuda.synchronize(self.device)
                prefill_times.append((time.perf_counter() - start) * 1000.0)

            with self._nvtx_range("flex_decode_graph"):
                self.static_decode_in.copy_(self.decode_token)
                heads = self.model.cfg.heads
                head_dim = self.model.head_dim
                for pos in range(self.decode_tokens):
                    start = time.perf_counter()
                    k = self.model.k_proj(self.decode_token).view(1, 1, heads, head_dim)
                    v = self.model.v_proj(self.decode_token).view(1, 1, heads, head_dim)
                    self.model._update_cache(k, v, self.base_position + pos)
                    self.model._set_offset(self.base_position + pos)
                    with torch.cuda.stream(self.capture_stream):
                        self.graph.replay()
                    torch.cuda.synchronize(self.device)
                    decode_times.append((time.perf_counter() - start) * 1000.0)

        # Store last output for verification (graph replay writes into static_decode_out)
        self._last_output = self.static_decode_out
        self._history["prefill_ms"].extend(prefill_times)
        self._history["decode_ms"].extend(decode_times)
        return {"prefill_ms": prefill_times, "decode_ms": decode_times}

    def teardown(self) -> None:
        if self.model is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().teardown()
        self.graph = None
        self.capture_stream = None
        self.static_decode_in = None
        self.static_decode_out = None
        self.base_position = 0

def get_benchmark():
    return OptimizedFlexDecodingGraphsBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
