"""scheduling_vllm_sglang.py - Continuous batching + speculative decode toy."""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class SchedulingBenchmark(BaseBenchmark):
    """Toy scheduler that batches requests and accepts speculative drafts."""

    def __init__(self) -> None:
        super().__init__()
        self.queue: Deque[int] = deque()
        self._workload = WorkloadMetadata(requests_per_iteration=1.0, tokens_per_iteration=64.0)
        self._history: Dict[str, float] = {}

    def setup(self) -> None:
        self.queue.clear()

    def _generate_requests(self, n: int = 8) -> None:
        for _ in range(n):
            self.queue.append(random.randint(4, 32))

    def _serve_batch(self, batch_tokens: int) -> int:
        # Simulate speculative accept ratio.
        accepted = int(batch_tokens * 0.8)
        return accepted

    def benchmark_fn(self) -> Optional[dict]:
        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("scheduling_vllm_sglang", enable=enable_nvtx):
            if not self.queue:
                self._generate_requests()
            batch_tokens = 0
            served = 0
            while self.queue and batch_tokens < 64:
                tokens = self.queue.popleft()
                batch_tokens += tokens
                served += self._serve_batch(tokens)
        self._history["served_tokens"] = served
        return {"served_tokens": served, "batched_tokens": batch_tokens}

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    return SchedulingBenchmark()
