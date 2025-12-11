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

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: requires >=2 GPUs")
        super().setup()
        if self.batches:
            self._verify_inputs = {"tokens": self.batches[0][:1].detach().clone()}
        else:
            self._verify_inputs = {"tokens": torch.zeros((1, self.hidden_dim), device=self.device)}

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()

    def get_verify_inputs(self) -> dict:
        return {k: v.detach().clone() for k, v in getattr(self, "_verify_inputs", {}).items()}

    def get_input_signature(self) -> dict:
        batch_shape = tuple(self._verify_inputs["tokens"].shape) if hasattr(self, "_verify_inputs") else (1, self.hidden_dim)
        return {
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "hidden_dim": self.hidden_dim,
            "shapes": {"tokens": batch_shape, "output": tuple(self.output.shape) if self.output is not None else (0,)},
            "dtypes": {"tokens": "float32"},
        }



def get_benchmark() -> BaselineContinuousBatchingBenchmark:
    return BaselineContinuousBatchingBenchmark()
