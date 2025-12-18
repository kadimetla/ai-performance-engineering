"""Optimized async input pipeline benchmark (pinned + non-blocking + copy stream)."""

from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.common.async_input_pipeline import AsyncInputPipelineBenchmark, PipelineConfig


def get_benchmark() -> AsyncInputPipelineBenchmark:
    cfg = PipelineConfig(
        batch_size=128,
        feature_shape=(3, 224, 224),
        dataset_size=512,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=True,
        non_blocking=True,
        use_copy_stream=True,
    )
    return AsyncInputPipelineBenchmark(cfg, label="optimized_async_input_pipeline")


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
