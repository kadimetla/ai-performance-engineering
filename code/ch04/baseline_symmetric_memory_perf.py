#!/usr/bin/env python3
"""Baseline symmetric-memory perf microbench (single GPU).

Measures copy latency/bandwidth using per-iteration buffer allocation.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.metrics import compute_memory_transfer_metrics
from ch04.verification_payload_mixin import VerificationPayloadMixin


class BaselineSymmetricMemoryPerfBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline device copy benchmark with per-iteration allocation."""

    def __init__(self, size_mb: float = 1.0):
        super().__init__()
        self.size_mb = size_mb
        self.numel = int((size_mb * 1024 * 1024) / 4)  # float32
        self.tensor: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._last_avg_ms = 0.0
        self._bytes_transferred = 0.0
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._verify_input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: requires CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.tensor = torch.randn(self.numel, device=self.device, dtype=torch.float32)
        self._verify_input = self.tensor[: 256 * 256].view(256, 256).detach()
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[Dict[str, float]]:
        if self.tensor is None:
            raise RuntimeError("Tensor not initialized")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = torch.empty_like(self.tensor)
        output.copy_(self.tensor, non_blocking=False)
        end.record()
        torch.cuda.synchronize(self.device)

        elapsed_ms = start.elapsed_time(end)
        bytes_moved = float(self.tensor.numel() * self.tensor.element_size())

        self._last_avg_ms = elapsed_ms
        self._bytes_transferred = bytes_moved
        self.output = output

        return {
            "copy.elapsed_ms": elapsed_ms,
            "copy.size_mb": self.size_mb,
        }

    def capture_verification_payload(self) -> None:
        if self.output is None:
            if self._verify_input is None:
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                self._verify_input = torch.randn(256, 256, device=self.device, dtype=torch.float32)
            output = self._verify_input.detach().clone()
            probe = self._verify_input
        else:
            probe = self.output[: 256 * 256].view(256, 256).detach()
            output = probe.clone()

        self._set_verification_payload(
            inputs={"tensor": probe},
            output=output,
            batch_size=int(probe.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-5, 1e-5),
            signature_overrides={"world_size": 1},
        )

    def teardown(self) -> None:
        self.tensor = None
        self.output = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred,
            elapsed_ms=self._last_avg_ms,
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "No output captured"
        if self._last_avg_ms <= 0:
            return "No timing recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineSymmetricMemoryPerfBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
