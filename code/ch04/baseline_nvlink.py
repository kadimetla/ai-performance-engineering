"""baseline_nvlink.py - Multi-GPU baseline without NVLink in distributed training context."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from typing import Optional

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from ch04.verification_payload_mixin import VerificationPayloadMixin


class BaselineNVLinkBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: PCIe-based communication (no NVLink).
    
    NVLink: This baseline does not use NVLink for high-speed GPU-to-GPU communication.
    Uses PCIe-based communication which is slower.
    """
    
    def __init__(self):
        super().__init__()
        self.data_gpu0 = None
        self.data_gpu1 = None
        self.output: Optional[torch.Tensor] = None
        self.N = 10_000_000
        # Memory transfer benchmark - jitter check not applicable
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
            bytes_per_iteration=float(self.N * 4),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: requires >=2 GPUs")
        torch.manual_seed(42)
        # Baseline: PCIe-based communication (no NVLink)
        # NVLink provides high-speed GPU-to-GPU communication
        # This baseline uses PCIe (slower)
        
        # Multi-GPU: use PCIe (not NVLink)
        self.data_gpu0 = torch.randn(self.N, device=torch.device("cuda:0"), dtype=torch.float32)
        self.data_gpu1 = torch.randn(self.N, device=torch.device("cuda:1"), dtype=torch.float32)
        torch.cuda.synchronize(self.device)
        probe = torch.randn(1024, device=self.device)
        output = torch.zeros(1, device=self.device)
        self._set_verification_payload(
            inputs={"probe": probe},
            output=output,
            batch_size=probe.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: PCIe-based communication (no NVLink)."""
        with self._nvtx_range("baseline_nvlink"):
            # Multi-GPU: PCIe-based transfer (no NVLink)
            # Transfer through PCIe bus (slower than NVLink)
            self.data_gpu1.copy_(self.data_gpu0, non_blocking=False)
            torch.cuda.synchronize()
            self.output = self.data_gpu1.sum().unsqueeze(0)

            # Baseline: No NVLink benefits
            # PCIe-based communication (slower)
            payload = getattr(self, "_verification_payload", None)
            if payload is not None and self.output is not None:
                payload.output = self.output.detach().clone()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data_gpu0 = None
        self.data_gpu1 = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data_gpu0 is None:
            return "Data not initialized"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return super().get_input_signature()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()
    
    def get_output_tolerance(self) -> tuple:
        """Return custom tolerance for memory transfer benchmark."""
        return (1e-3, 1e-3)



def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineNVLinkBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
