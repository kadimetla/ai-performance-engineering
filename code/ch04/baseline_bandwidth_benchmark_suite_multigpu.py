"""Benchmark wrapper for bandwidth suite; skips on <2 GPUs."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from ch04.verification_payload_mixin import VerificationPayloadMixin

import os
import time
from typing import Optional


def measure_peer_bandwidth(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    iterations: int = 50,
    stream: Optional[torch.cuda.Stream] = None,
) -> float:
    """Measure GPU-to-GPU bandwidth by copying a tensor between two devices."""
    if torch.cuda.device_count() < 2:
        raise RuntimeError("SKIPPED: bandwidth benchmark suite requires >=2 GPUs")
    if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
        raise TypeError("src and dst must be torch.Tensor")
    if src.numel() != dst.numel() or src.dtype != dst.dtype:
        raise ValueError("src and dst must have matching shape/dtype")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    bytes_per_iter = src.numel() * src.element_size()
    torch.cuda.synchronize(src.device)
    torch.cuda.synchronize(dst.device)
    start = time.perf_counter()
    if stream is not None:
        with torch.cuda.stream(stream):
            for _ in range(iterations):
                dst.copy_(src, non_blocking=True)
        stream.synchronize()
    else:
        for _ in range(iterations):
            dst.copy_(src, non_blocking=False)
    torch.cuda.synchronize(src.device)
    torch.cuda.synchronize(dst.device)
    elapsed = (time.perf_counter() - start) / iterations
    gb_per_iter = bytes_per_iter / 1e9
    return gb_per_iter / elapsed


class BandwidthSuiteMultiGPU(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.last_bandwidth_gbps: Optional[float] = None
        self.size_mb = 256
        self.inner_iterations = 50
        self.src: Optional[torch.Tensor] = None
        self.dst: Optional[torch.Tensor] = None
        self.stream: Optional[torch.cuda.Stream] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: bandwidth benchmark suite requires >=2 GPUs")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        bytes_per_iter = int(self.size_mb * 1024 * 1024)
        numel = bytes_per_iter // 4  # float32
        self.src = torch.randn(numel, device="cuda:0", dtype=torch.float32)
        self.dst = torch.empty_like(self.src, device="cuda:1")
        self.stream = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter * self.inner_iterations),
        )

    def benchmark_fn(self) -> None:
        if self.src is None or self.dst is None:
            raise RuntimeError("Benchmark not initialized")
        self.last_bandwidth_gbps = measure_peer_bandwidth(
            self.src,
            self.dst,
            iterations=self.inner_iterations,
            stream=self.stream,
        )

    def capture_verification_payload(self) -> None:
        if self.src is None or self.dst is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        probe = self.src[: 256 * 256].view(256, 256)
        output = self.dst[: 256 * 256].view(256, 256)
        self._set_verification_payload(
            inputs={"src": probe},
            output=output,
            batch_size=int(probe.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=30,
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return measured P2P bandwidth."""
        return {"p2p_bandwidth_gbps": float(self.last_bandwidth_gbps or 0.0)}
    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.0, 0.0)


def get_benchmark() -> BaseBenchmark:
    return BandwidthSuiteMultiGPU()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
