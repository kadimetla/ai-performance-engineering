"""Alias to reuse the optimized MoE inference benchmark implemented in ch18."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch18.run_vllm_decoder import VLLMMoEInferenceBenchmark  # noqa: E402
from ch15.verification_payload_mixin import VerificationPayloadMixin
import torch


class Ch15VLLMMoEInferenceBenchmark(VerificationPayloadMixin, VLLMMoEInferenceBenchmark):
    """Chapter-local wrapper that supplies strict verification metadata."""

    def benchmark_fn(self) -> None:
        super().benchmark_fn()
        prompt = getattr(self, "prompts", None)
        if prompt is None:
            prompt = torch.zeros(1, 1, device=self.device, dtype=torch.int64)
        param_count = sum(p.numel() for p in self.model.parameters()) if getattr(self, "model", None) is not None else 0
        precision_flags = {
            "fp16": self.config.dtype_obj == torch.float16,
            "bf16": self.config.dtype_obj == torch.bfloat16,
            "fp8": False,
            "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
        }
        if getattr(self, "output", None) is not None:
            self._set_verification_payload(
                inputs={"prompt": prompt},
                output=self.output.detach().clone(),
                batch_size=int(prompt.shape[0]),
                parameter_count=param_count,
                precision_flags=precision_flags,
                output_tolerance=(1e-3, 1e-3),
            )

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        return super().get_output_tolerance()


def get_benchmark():
    return Ch15VLLMMoEInferenceBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
