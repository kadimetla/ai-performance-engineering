"""Optimized NVSHMEM IBGDA microbenchmark (single-GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch04.nvshmem_ibgda_microbench_multigpu import NvshmemIbgdaMicrobench
from core.harness.benchmark_harness import BenchmarkConfig

_DEFAULT_KWARGS = dict(
    mode="put",
    bytes_per_message=1048576,
    ctas=16,
    threads=256,
    iters=200,
)


class OptimizedNvshmemIbgdaMicrobenchSingle(NvshmemIbgdaMicrobench):
    multi_gpu_required = False

    def __init__(self) -> None:
        super().__init__(enable_ibgda=True, world_size=1, ibgda_batch=16, **_DEFAULT_KWARGS)

    def get_config(self) -> BenchmarkConfig:
        config = super().get_config()
        config.multi_gpu_required = False
        config.single_gpu = True
        return config


def get_benchmark() -> NvshmemIbgdaMicrobench:
    return OptimizedNvshmemIbgdaMicrobenchSingle()
