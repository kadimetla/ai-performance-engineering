#!/usr/bin/env python3
"""optimized_prefill_decode_disagg.py - NVLink peer-copy + pipelined disaggregation.

Same semantic workload as `baseline_prefill_decode_disagg.py`:
- Prefill on `cuda:0`
- Decode on `cuda:1`
- Same models, same inputs, same decode recurrence

Optimizations:
- KV handoff uses direct GPU peer copy (NVLink/NVSwitch when available)
- Prefill for request i+1 overlaps decode for request i (device-level pipelining)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.gpu_requirements import require_peer_access
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch15.verification_payload_mixin import VerificationPayloadMixin


def _enable_peer_access() -> None:
    # Fail fast if peer access is not supported. This benchmark is specifically about NVLink pooling.
    require_peer_access(0, 1)


class OptimizedPrefillDecodeDisaggBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized prefill/decode disaggregation with peer KV handoff + pipelining."""

    def __init__(
        self,
        *,
        batch_size: int = 8,
        prefill_length: int = 1024,
        decode_length: int = 64,
        hidden_size: int = 2048,
    ) -> None:
        super().__init__()
        self.batch_size = int(batch_size)
        self.prefill_length = int(prefill_length)
        self.decode_length = int(decode_length)
        self.hidden_size = int(hidden_size)

        tokens = self.batch_size * (self.prefill_length + self.decode_length)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

        self.prefill_device: Optional[torch.device] = None
        self.decode_device: Optional[torch.device] = None
        self.prefill_model: Optional[nn.Module] = None
        self.decode_model: Optional[nn.Module] = None
        self.prefill_inputs: Optional[torch.Tensor] = None
        self._verify_probe: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for prefill/decode disaggregation")
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: prefill/decode disaggregation requires >=2 GPUs")

        _enable_peer_access()

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.prefill_device = torch.device("cuda:0")
        self.decode_device = torch.device("cuda:1")

        self.prefill_model = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(
            self.prefill_device, dtype=torch.bfloat16
        ).eval()
        self.decode_model = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(
            self.decode_device, dtype=torch.bfloat16
        ).eval()

        self.prefill_inputs = torch.randn(
            self.batch_size,
            self.prefill_length,
            self.hidden_size,
            device=self.prefill_device,
            dtype=torch.bfloat16,
        )
        self._verify_probe = self.prefill_inputs[:1, :1, :256].detach()
        torch.cuda.synchronize(self.prefill_device)
        torch.cuda.synchronize(self.decode_device)

    def benchmark_fn(self) -> None:
        if (
            self.prefill_device is None
            or self.decode_device is None
            or self.prefill_model is None
            or self.decode_model is None
            or self.prefill_inputs is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        outputs: list[torch.Tensor] = [torch.empty(self.hidden_size, device=self.decode_device, dtype=torch.bfloat16) for _ in range(self.batch_size)]

        with self._nvtx_range("optimized_prefill_decode_disagg"):
            with torch.no_grad():
                for idx in range(self.batch_size):
                    # Prefill on GPU0 (async wrt GPU1 work).
                    prefill_out = self.prefill_model(self.prefill_inputs[idx : idx + 1])

                    # Direct peer handoff to GPU1 (no CPU staging).
                    kv_decode = prefill_out.to(self.decode_device, non_blocking=True)

                    # Decode on GPU1. This work overlaps the next request's prefill on GPU0.
                    token_state = kv_decode[:, -1:, :]
                    for _ in range(self.decode_length):
                        token_state = self.decode_model(token_state)
                    outputs[idx] = token_state.squeeze(0).squeeze(0)

        torch.cuda.synchronize(self.prefill_device)
        torch.cuda.synchronize(self.decode_device)
        self.output = torch.stack(outputs, dim=0)

    def capture_verification_payload(self) -> None:
        if (
            self.output is None
            or self._verify_probe is None
            or self.prefill_model is None
            or self.decode_model is None
        ):
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")

        output_slice = self.output[:2, :256].detach().cpu().float().clone()
        param_count = sum(p.numel() for p in self.prefill_model.parameters()) + sum(
            p.numel() for p in self.decode_model.parameters()
        )
        self._set_verification_payload(
            inputs={"probe": self._verify_probe.detach().cpu()},
            output=output_slice,
            batch_size=int(self.batch_size),
            parameter_count=int(param_count),
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.prefill_model = None
        self.decode_model = None
        self.prefill_inputs = None
        self._verify_probe = None
        self.output = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5, multi_gpu_required=True)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    return OptimizedPrefillDecodeDisaggBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
