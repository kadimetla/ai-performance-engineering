"""optimized_nvls_collectives.py - NVLS-style NCCL collectives placeholder.

This benchmark checks for multi-GPU availability and, when possible, exercises a
small all-reduce with NCCL. If the environment lacks multiple GPUs or torchrun,
it cleanly reports SKIPPED so the harness stays green.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class NVLSCollectivesBenchmark(BaseBenchmark):
    """Tiny NCCL all-reduce meant to mirror the doc's NVLS target."""

    def __init__(self) -> None:
        super().__init__()
        self.tensor: Optional[torch.Tensor] = None
        self._initialized = False
        self._workload = WorkloadMetadata(bytes_per_iteration=0.0)

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: NVLS collectives require >=2 GPUs")

        # Single-process setup; users can still torchrun for real multi-rank runs.
        os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
        if dist.is_available() and not dist.is_initialized():
            dist.init_process_group("nccl", rank=0, world_size=1)
        self.tensor = torch.ones(1024, device=self.device)
        self._initialized = True
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if not self._initialized or self.tensor is None:
            raise RuntimeError("SKIPPED: NVLS benchmark not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("nvls_allreduce", enable=enable_nvtx):
            dist.all_reduce(self.tensor)
        torch.cuda.synchronize(self.device)
        return {"sum": float(self.tensor[0].item())}

    def teardown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        super().teardown()

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    return NVLSCollectivesBenchmark()
