"""Optimized double buffering benchmark with pipelined shared-memory tiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch8.double_buffering_benchmark_base import DoubleBufferingBenchmarkBase


class OptimizedDoubleBufferingBenchmark(DoubleBufferingBenchmarkBase):
    nvtx_label = "optimized_double_buffering"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.input is not None
        assert self.output is not None
        self.extension.double_buffer_optimized(self.input, self.output)


def get_benchmark() -> DoubleBufferingBenchmarkBase:
    return OptimizedDoubleBufferingBenchmark()


def _apply_profile_overrides(args: argparse.Namespace, benchmark: DoubleBufferingBenchmarkBase) -> None:
    """Install optional overrides used only when profiling tools require smaller grids."""
    if args.elements and args.elements > 0:
        elements_override = args.elements
    elif args.profile_lite:
        lite_elements = args.lite_elements
        if lite_elements is None:
            lite_elements = benchmark.block * benchmark.tile * 1024
        elements_override = lite_elements
    else:
        elements_override = None

    skip_validation = args.profile_lite or args.skip_validation
    benchmark.configure_profile_overrides(
        elements=elements_override,
        skip_validation=skip_validation,
    )


def _run_with_torch_profiler(
    benchmark: DoubleBufferingBenchmarkBase,
    warmup_iters: int,
    record_iters: int,
    trace_path: Optional[str],
) -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    benchmark.setup()
    try:
        torch.cuda.synchronize()
        for _ in range(max(warmup_iters, 0)):
            benchmark._invoke_kernel()
        torch.cuda.synchronize()
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
            for _ in range(max(record_iters, 1)):
                with torch.autograd.profiler.record_function("double_buffer_optimized"):
                    benchmark._invoke_kernel()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
        if trace_path:
            prof.export_chrome_trace(trace_path)
            print(f"Exported PyTorch profiler trace to {trace_path}")
    finally:
        benchmark.teardown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the optimized double-buffering benchmark or launch profiling helpers. "
            "The profiling-only flags (--profile-lite/--lite-elements/--skip-validation) "
            "exist so Nsight Compute can finish on GB10 hardware; they should not be used "
            "for benchmark/regression numbers."
        ),
    )
    parser.add_argument(
        "--profile-lite",
        action="store_true",
        help="Shrink the tensor size (default 1/8th) so Nsight Compute can complete its passes (implies --skip-validation).",
    )
    parser.add_argument(
        "--lite-elements",
        type=int,
        default=None,
        help="Explicit element count for --profile-lite runs. Defaults to block*tile*1024.",
    )
    parser.add_argument(
        "--elements",
        type=int,
        default=None,
        help="Override tensor length directly (takes precedence over --profile-lite).",
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Capture a PyTorch profiler trace instead of running the benchmark harness.",
    )
    parser.add_argument(
        "--profiler-warmup",
        type=int,
        default=3,
        help="Warm-up iterations before recording PyTorch profiler events.",
    )
    parser.add_argument(
        "--profiler-iters",
        type=int,
        default=10,
        help="Recorded iterations for PyTorch profiler fallback.",
    )
    parser.add_argument(
        "--profiler-trace",
        type=str,
        default=None,
        help="Optional path to export a Chrome trace from the PyTorch profiler.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the CPU reference check (handy for very small profiling inputs).",
    )
    args = parser.parse_args()

    from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode

    benchmark = OptimizedDoubleBufferingBenchmark()
    _apply_profile_overrides(args, benchmark)

    if args.torch_profiler:
        _run_with_torch_profiler(
            benchmark=benchmark,
            warmup_iters=args.profiler_warmup,
            record_iters=args.profiler_iters,
            trace_path=args.profiler_trace,
        )
        return

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=30, warmup=5),
    )
    result = harness.benchmark(benchmark)
    print("=" * 70)
    print("Optimized Double Buffering")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
