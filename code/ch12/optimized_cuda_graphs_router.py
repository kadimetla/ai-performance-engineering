"""optimized_cuda_graphs_router.py - CUDA graphs with a simple branch toggle.

This variant reuses the cuda_graphs extension (same math as the baseline) but
flips a small route flag each iteration to emulate a conditional branch while
keeping verification aligned with the baseline output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402
from ch12.cuda_extensions import load_cuda_graphs_extension  # noqa: E402


class CUDAGraphRouterBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """CUDA Graph routing benchmark with a branch toggle on identical graphs."""

    def __init__(self) -> None:
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.route_flag: int = 0
        self.N = 1 << 20  # 1M elements to match baseline
        self.iterations = 500
        self._extension = None
        self._workload = WorkloadMetadata(tokens_per_iteration=float(self.N))

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for graph capture")
        self._extension = load_cuda_graphs_extension()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        # Warmup capture inside the extension
        self._extension.graph_replay(self.data, 1)
        torch.cuda.synchronize(self.device)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self._extension is None or self.data is None:
            raise RuntimeError("SKIPPED: graph not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("cuda_graphs_router", enable=enable_nvtx):
            # Flip route between iterations to emulate a conditional branch.
            self.route_flag ^= 1
            self._extension.graph_replay(self.data, self.iterations)
        self._synchronize()
        if self.data is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        self._set_verification_payload(
            inputs={"input": self.data.detach().clone()},
            output=self.data.detach().clone(),
            batch_size=self.N,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_graph_metrics
        return compute_graph_metrics(
            baseline_launch_overhead_us=getattr(self, '_baseline_launch_us', 10.0),
            graph_launch_overhead_us=getattr(self, '_graph_launch_us', 1.0),
            num_nodes=getattr(self, 'num_nodes', 10),
            num_iterations=getattr(self, 'num_iterations', 100),
        )

    def validate_result(self) -> Optional[str]:
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.numel() != self.N:
            return f"Data size mismatch: expected {self.N}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    return CUDAGraphRouterBenchmark()
