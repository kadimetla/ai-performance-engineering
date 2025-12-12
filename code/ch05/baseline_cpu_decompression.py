"""baseline_cpu_decompression.py - CPU-bound decompression baseline.

Expands a small buffer compressed with zlib entirely on the CPU. Serves as a
baseline for the GPU-oriented nvCOMP-style path.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class CPUDecompressionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.compressed: Optional[bytes] = None
        self.output: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(bytes_per_iteration=0.0)

    def setup(self) -> None:
        payload = torch.randn(1024 * 1024, dtype=torch.float32).numpy().tobytes()
        self.compressed = zlib.compress(payload, level=6)

    def benchmark_fn(self) -> Optional[dict]:
        if self.compressed is None:
            raise RuntimeError("SKIPPED: no compressed payload available")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        start = self._record_start()
        with nvtx_range("cpu_decompress", enable=enable_nvtx):
            decompressed = zlib.decompress(self.compressed)
        latency_ms = self._record_stop(start)
        # Convert decompressed bytes to tensor for verification
        import numpy as np
        self.output = torch.from_numpy(np.frombuffer(decompressed, dtype=np.float32).copy())
        compressed_tensor = torch.tensor(list(self.compressed), dtype=torch.uint8)
        self._payload_compressed_tensor = compressed_tensor
        return {"latency_ms": latency_ms, "compressed_bytes": len(self.compressed)}

    def capture_verification_payload(self) -> None:
        compressed_tensor = self._payload_compressed_tensor
        self._set_verification_payload(
            inputs={"compressed": compressed_tensor},
            output=self.output.detach().clone(),
            batch_size=self.output.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_storage_io_metrics
        return compute_storage_io_metrics(
            bytes_read=getattr(self, '_bytes_read', 0.0),
            bytes_written=getattr(self, '_bytes_written', 0.0),
            read_time_ms=getattr(self, '_read_time_ms', 1.0),
            write_time_ms=getattr(self, '_write_time_ms', 1.0),
        )


def get_benchmark() -> BaseBenchmark:
    return CPUDecompressionBenchmark()
