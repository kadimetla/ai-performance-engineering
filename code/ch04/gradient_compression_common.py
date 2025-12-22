"""Shared gradient compression benchmark logic (single-process multi-GPU NCCL)."""

from __future__ import annotations

from typing import List, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch04.verification_payload_mixin import VerificationPayloadMixin


def attach_benchmark_metadata(bench: BaseBenchmark, module_file: str) -> BaseBenchmark:
    """Ensure subprocess runner calls get_benchmark() for parametrized benchmarks."""
    bench._module_file_override = module_file
    bench._factory_name_override = "get_benchmark"
    return bench


class GradientCompressionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Gradient all-reduce benchmark with optional compression."""

    def __init__(
        self,
        *,
        compression: str,
        equivalence_group: str,
        output_tolerance: tuple[float, float],
        tensor_size_mb: int = 128,
    ) -> None:
        super().__init__()
        self.signature_equivalence_group = equivalence_group
        self.signature_equivalence_ignore_fields = ("precision_flags",)
        self.compression = compression  # "none", "fp16", "int8"
        self.output_tolerance = output_tolerance
        self.tensor_size_mb = tensor_size_mb
        self.world_size = 0
        self.devices: List[torch.device] = []
        self.inputs: List[torch.Tensor] = []
        self.output: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        tokens = float(tensor_size_mb * 1024 * 1024)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.world_size = torch.cuda.device_count()
        if self.world_size < 2:
            raise RuntimeError("SKIPPED: requires >=2 GPUs")
        self.devices = [torch.device(f"cuda:{idx}") for idx in range(self.world_size)]
        numel = (self.tensor_size_mb * 1024 * 1024) // 4  # FP32 bytes
        self.inputs = [
            torch.randn(numel, device=device, dtype=torch.float32) for device in self.devices
        ]
        self._verify_input = self.inputs[0]
        self._synchronize_all()

    def benchmark_fn(self) -> None:
        if not self.inputs:
            raise RuntimeError("Inputs not initialized")
        with self._nvtx_range(f"gradient_compression_{self.compression}"):
            if self.compression == "none":
                outputs = [torch.empty_like(t) for t in self.inputs]
                torch.cuda.nccl.all_reduce(self.inputs, outputs=outputs)
                self.output = outputs[0]
            elif self.compression == "fp16":
                compressed = [t.to(torch.float16) for t in self.inputs]
                outputs = [torch.empty_like(t) for t in compressed]
                torch.cuda.nccl.all_reduce(compressed, outputs=outputs)
                self.output = outputs[0].float()
            elif self.compression == "int8":
                self.output = self._int8_all_reduce()
            else:
                raise ValueError(f"Unknown compression mode: {self.compression}")
        self._synchronize_all()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def _int8_all_reduce(self) -> torch.Tensor:
        max_vals = [t.abs().max() for t in self.inputs]
        max_tensors = [m.clone() for m in max_vals]
        torch.cuda.nccl.all_reduce(max_tensors, op=torch.cuda.nccl.ReduceOp.MAX)
        scales = [m * (self.world_size / 127.0) for m in max_tensors]
        quantized = [
            torch.clamp((t / scale).round(), -127, 127).to(torch.int8)
            for t, scale in zip(self.inputs, scales)
        ]
        outputs = [torch.empty_like(t) for t in quantized]
        torch.cuda.nccl.all_reduce(quantized, outputs=outputs)
        return outputs[0].float() * scales[0]

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        precision_flags = {
            "fp16": self.compression == "fp16",
            "bf16": False,
            "fp8": False,
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.output.detach().clone(),
            batch_size=int(self._verify_input.shape[0]),
            parameter_count=0,
            precision_flags=precision_flags,
            output_tolerance=self.output_tolerance,
            signature_overrides={
                "world_size": self.world_size,
                "ranks": list(range(self.world_size)),
                "collective_type": "all_reduce",
            },
        )

    def teardown(self) -> None:
        self.inputs = []
        self.output = None
        self._verify_input = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if not self.inputs:
            return "Inputs not initialized"
        return None

    def _synchronize_all(self) -> None:
        for device in self.devices or [self.device]:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
