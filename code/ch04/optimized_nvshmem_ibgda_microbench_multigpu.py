"""Optimized NVSHMEM IBGDA microbenchmark (GPU-initiated path)."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch04.nvshmem_ibgda_microbench_multigpu import NvshmemIbgdaMicrobench

_DEFAULT_KWARGS = dict(
    mode="p",
    bytes_per_message=262144,
    ctas=512,
    threads=256,
    iters=1200,
)


class OptimizedNvshmemIbgdaMicrobench(NvshmemIbgdaMicrobench):
    def __init__(self) -> None:
        super().__init__(enable_ibgda=True, **_DEFAULT_KWARGS)


def get_benchmark() -> NvshmemIbgdaMicrobench:
    return OptimizedNvshmemIbgdaMicrobench()
