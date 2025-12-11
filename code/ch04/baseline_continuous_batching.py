"""Baseline continuous batching shim for ch04 multi-GPU wrappers.

Reuses the chapter 15 baseline implementation so the ch04 harness can discover
the benchmark and gracefully skip on single-GPU hosts.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch15.baseline_continuous_batching import BaselineContinuousBatchingBenchmark as _BaselineContinuousBatchingBenchmark


class BaselineContinuousBatchingBenchmark(_BaselineContinuousBatchingBenchmark):
    """Direct alias to the chapter 15 baseline; skips when GPUs < 2 in the wrapper."""

    def __init__(self) -> None:
        super().__init__()
        init_tokens = torch.zeros((1, self.hidden_dim), dtype=torch.float32)
        self._verify_inputs = {"tokens": init_tokens}
        # Seed verification payload so audit can query without setup.
        self._set_verification_payload(
            inputs=self._verify_inputs,
            output=torch.zeros_like(init_tokens),
            batch_size=1,
            parameter_count=0,
        )

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: requires >=2 GPUs")
        super().setup()
        if self.batches:
            tokens = torch.cat(self.batches, dim=0).detach().clone()
            self._verify_inputs = {"tokens": tokens}
        else:
            self._verify_inputs = {"tokens": torch.zeros((1, self.hidden_dim), device=self.device)}

    def benchmark_fn(self) -> None:
        super().benchmark_fn()
        self._set_verification_payload(
            inputs={k: v for k, v in self._verify_inputs.items()},
            output=self.output,
            batch_size=self.num_samples if hasattr(self, "num_samples") else self.batch_size,
            parameter_count=sum(p.numel() for p in self.model.parameters()) if getattr(self, "model", None) else 0,
            output_tolerance=super().get_output_tolerance(),
        )



def get_benchmark() -> BaselineContinuousBatchingBenchmark:
    return BaselineContinuousBatchingBenchmark()
