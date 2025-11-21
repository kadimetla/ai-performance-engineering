"""Baseline async input pipeline benchmark (blocking copies, no pinning)."""

from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.async_input_pipeline.pipeline import AsyncInputPipelineBenchmark, PipelineConfig


def get_benchmark() -> AsyncInputPipelineBenchmark:
    cfg = PipelineConfig(
        batch_size=128,
        feature_shape=(3, 224, 224),
        dataset_size=512,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        non_blocking=False,
        use_copy_stream=False,
    )
    return AsyncInputPipelineBenchmark(cfg, label="baseline_async_input_pipeline")


if __name__ == "__main__":
    benchmark = get_benchmark()
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"\nbaseline_async_input_pipeline: {mean:.3f} ms/iter")
