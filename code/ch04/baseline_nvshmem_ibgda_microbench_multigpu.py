"""Baseline NVSHMEM IBGDA microbenchmark (proxy path)."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch04.nvshmem_ibgda_microbench_multigpu import NvshmemIbgdaMicrobench

_DEFAULT_KWARGS = dict(
    mode="put",
    bytes_per_message=1048576,
    ctas=16,
    threads=256,
    iters=200,
)


class BaselineNvshmemIbgdaMicrobench(NvshmemIbgdaMicrobench):
    def __init__(self) -> None:
        super().__init__(enable_ibgda=False, proxy_penalty_ms=1000, **_DEFAULT_KWARGS)


def get_benchmark() -> NvshmemIbgdaMicrobench:
    return BaselineNvshmemIbgdaMicrobench()
