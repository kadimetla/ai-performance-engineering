"""Optimized decode bucketing demo: buckets + prewarm shrink CUDA graph churn."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch18.baseline_cudagraph_bucketing import (  # noqa: E402
    BaselineCUDAGraphBucketing,
)
from ch18.cudagraph_bucketing_common import (  # noqa: E402
    DEFAULT_CAPTURE_BATCH_SIZES,
    BucketBands,
    GraphTreeSimulator,
    capture_bins_from_vllm_config,
    default_bucket_bands,
    demo_traffic,
    load_vllm_config,
    pad_batch_to_capture,
    pad_fn_from_vllm_config,
)
from ch18.cudagraph_bucketing_metrics import (  # noqa: E402
    export_stats_to_prometheus,
)
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402


class OptimizedCUDAGraphBucketing(BaselineCUDAGraphBucketing):
    """
    Buckets batch/seq shapes, rounds to CUDA-graph capture sizes, and pre-warms
    the hot buckets so the first live requests replay instead of capturing.
    """

    def __init__(
        self,
        traffic: Iterable[Tuple[int, int]] | None = None,
        bucket_bands: BucketBands | None = None,
        prewarm_shapes: Iterable[Tuple[int, int]] | None = None,
        vllm_model: str = "gpt-oss-20b",
        use_vllm_bins: bool = True,
        region: str = "local",
        model_label: str = "gpt-oss-20b",
    ) -> None:
        super().__init__(traffic=traffic)
        self.vllm_model = vllm_model
        self.use_vllm_bins = use_vllm_bins
        self._vllm_config = load_vllm_config(vllm_model) if use_vllm_bins else None
        self.bucket_bands = bucket_bands if bucket_bands is not None else default_bucket_bands()
        self.prewarm_shapes: List[Tuple[int, int]] = list(prewarm_shapes) if prewarm_shapes else self._default_prewarm()
        self.region = region
        self.model_label = model_label

    def _default_prewarm(self) -> List[Tuple[int, int]]:
        # Prime the most common padded buckets from the demo traffic so the first live hits replay.
        freq: dict[Tuple[int, int], int] = {}
        capture_bins = capture_bins_from_vllm_config(self._vllm_config) if self._vllm_config else DEFAULT_CAPTURE_BATCH_SIZES
        pad_fn = pad_fn_from_vllm_config(self._vllm_config) if self._vllm_config else None
        for raw_batch, raw_seqlen in demo_traffic():
            b_bucket, s_bucket = self.bucket_bands.bucket(raw_batch, raw_seqlen)
            padded_batch = pad_batch_to_capture(b_bucket, capture_bins, pad_fn)
            if padded_batch is None:
                continue
            freq[(padded_batch, s_bucket)] = freq.get((padded_batch, s_bucket), 0) + 1
        return [shape for shape, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:4]]

    def build_simulator(self) -> GraphTreeSimulator:
        capture_bins = capture_bins_from_vllm_config(self._vllm_config) if self._vllm_config else DEFAULT_CAPTURE_BATCH_SIZES
        pad_fn = pad_fn_from_vllm_config(self._vllm_config) if self._vllm_config else None
        sim = GraphTreeSimulator(
            bucket_bands=self.bucket_bands,
            capture_batch_sizes=capture_bins,
            name="optimized_cudagraphs",
            pad_fn=pad_fn,
        )
        if self.prewarm_shapes:
            sim.prewarm(self.prewarm_shapes)
        return sim

    def run_compile_smoke(self) -> dict[str, int]:
        """
        Wrap a toy decode step in torch.compile(dynamic=True) and execute
        padded bucket shapes to confirm low recompile counts.
        """
        capture_bins = capture_bins_from_vllm_config(self._vllm_config) if self._vllm_config else DEFAULT_CAPTURE_BATCH_SIZES
        pad_fn = pad_fn_from_vllm_config(self._vllm_config) if self._vllm_config else None
        shapes: List[Tuple[int, int]] = []

        for raw_batch, raw_seqlen in self.traffic:
            b_bucket, s_bucket = self.bucket_bands.bucket(raw_batch, raw_seqlen)
            padded_batch = pad_batch_to_capture(b_bucket, capture_bins, pad_fn)
            if padded_batch is None:
                continue
            shapes.append((padded_batch, s_bucket))

        compile_counter = {"compiles": 0}

        def counting_backend(gm, example_inputs, **kwargs):
            compile_counter["compiles"] += 1
            inductor = getattr(torch, "_inductor", None)
            if inductor is not None and hasattr(inductor, "compile"):
                return inductor.compile(gm, example_inputs)  # type: ignore[call-arg]
            return gm.forward

        def decode_step(x: torch.Tensor) -> torch.Tensor:
            # Light fake decode math to keep compile fast.
            return torch.relu(x).sum(dim=-1)

        compiled = torch.compile(
            decode_step,
            dynamic=True,
            mode="reduce-overhead",
            backend=counting_backend,
        )

        for batch, seqlen in shapes:
            x = torch.randn(batch, seqlen, device="cpu")
            _ = compiled(x)

        return compile_counter


def _build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimized CUDA graph bucketing simulator", add_help=add_help)
    parser.add_argument("--vllm-model", type=str, default="gpt-oss-20b", help="Model name for capture bins.")
    parser.add_argument(
        "--no-vllm-bins",
        action="store_true",
        help="Force fallback capture bins instead of reading vLLM config",
    )
    parser.add_argument("--region", type=str, default="local", help="Region label for metrics/export.")
    parser.add_argument(
        "--model-label",
        type=str,
        default="gpt-oss-20b",
        help="Model label for metrics/export.",
    )
    parser.add_argument(
        "--prom-port",
        type=int,
        default=None,
        help="If set, starts a Prometheus HTTP exporter on this port and publishes graph stats",
    )
    parser.add_argument(
        "--skip-compile-smoke",
        action="store_true",
        help="Skip the torch.compile dynamic-shape smoke test",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    traffic = demo_traffic()
    optimized = OptimizedCUDAGraphBucketing(
        traffic=traffic,
        vllm_model=args.vllm_model,
        use_vllm_bins=not args.no_vllm_bins,
        region=args.region,
        model_label=args.model_label,
    )
    sim = optimized.run()
    print(sim.format_summary())

    if not args.skip_compile_smoke:
        compile_stats = optimized.run_compile_smoke()
        print(f"[compile] torch.compile(dynamic=True) recompiles: {compile_stats['compiles']}")

    export_stats_to_prometheus(
        sim.stats,
        region=args.region,
        model=args.model_label,
        start_port=args.prom_port,
    )


class OptimizedCUDAGraphBucketingBenchmark(BaseBenchmark):
    """Benchmark wrapper so the optimized simulator can run via benchmark_cli."""

    def __init__(self) -> None:
        super().__init__()
        self.vllm_model = "gpt-oss-20b"
        self.use_vllm_bins = True
        self.region = "local"
        self.model_label = "gpt-oss-20b"
        self.skip_compile_smoke = False
        self._last_sim: Optional[GraphTreeSimulator] = None
        self._compile_stats: Optional[dict] = None

    def _resolve_device(self) -> torch.device:
        return torch.device("cpu")

    def apply_target_overrides(self, argv: Iterable[str]) -> None:
        parser = _build_parser(add_help=False)
        try:
            args, _ = parser.parse_known_args(list(argv))
            self.vllm_model = args.vllm_model
            self.use_vllm_bins = not args.no_vllm_bins
            self.region = args.region
            self.model_label = args.model_label
            self.skip_compile_smoke = bool(args.skip_compile_smoke)
        except SystemExit:
            pass

    def benchmark_fn(self) -> None:
        optimized = OptimizedCUDAGraphBucketing(
            traffic=demo_traffic(),
            vllm_model=self.vllm_model,
            use_vllm_bins=self.use_vllm_bins,
            region=self.region,
            model_label=self.model_label,
        )
        sim = optimized.run()
        self._last_sim = sim
        if not self.skip_compile_smoke:
            self._compile_stats = optimized.run_compile_smoke()

    def get_custom_metrics(self) -> Optional[dict]:
        if self._last_sim is None:
            return None
        summary = self._last_sim.stats.summary()
        flat = {f"cudagraph.{k}": v for k, v in summary.items() if k != "hot_keys"}
        if self._compile_stats is not None:
            flat["cudagraph.compile_recompiles"] = float(self._compile_stats.get("compiles", 0))
        return flat

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=0, enable_profiling=False)


def get_benchmark() -> BaseBenchmark:
    return OptimizedCUDAGraphBucketingBenchmark()


if __name__ == "__main__":
    main()
