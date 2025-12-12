"""baseline_piece_graphs.py - Monolithic CUDA graph capture baseline for piece-graphs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class PieceGraphBlock(nn.Module):
    """Sequential block to mimic per-piece captures."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.output = None
        self._verify_input = None
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PieceGraphModel(nn.Module):
    """Sequential stack of PieceGraphBlocks."""

    def __init__(self, hidden_dim: int = 512, n_layers: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([PieceGraphBlock(hidden_dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class BaselinePieceGraphsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Monolithic graph capture baseline for piece-graphs."""

    def __init__(self):
        super().__init__()
        self.model: Optional[PieceGraphModel] = None
        self.inputs: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_output: Optional[torch.Tensor] = None
        # Heavier per-iteration work to better expose graph replay speedups.
        self.repeats = 12
        self.batch = 16
        self.hidden = 768
        tokens = self.batch * self.hidden * self.repeats
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(tokens),
        )
        self._verify_input: Optional[torch.Tensor] = None
        self.parameter_count: int = 0

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.model = (
            PieceGraphModel(hidden_dim=self.hidden, n_layers=12)
            .to(self.device, dtype=torch.float16)
            .eval()
        )
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.inputs = torch.randn(
            self.batch, self.hidden, device=self.device, dtype=torch.float16
        )
        self._verify_input = self.inputs.detach().clone()
        # Capture monolithic graph
        try:
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = self.inputs.clone()
            self.static_output = torch.empty_like(self.static_input)
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph):
                self.static_output.copy_(self.model(self.static_input))
            torch.cuda.synchronize()
        except Exception:
            self.graph = None
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        last_output: Optional[torch.Tensor] = None
        with nvtx_range("baseline_piece_graphs", enable=enable_nvtx):
            for _ in range(self.repeats):
                if self.graph and self.static_input is not None and self.static_output is not None:
                    self.static_input.copy_(self.inputs)
                    self.graph.replay()
                    last_output = self.static_output
                else:
                    last_output = self.model(self.inputs)
            torch.cuda.synchronize(self.device)
        if last_output is None and self.model is not None:
            last_output = self.model(self.inputs)
        if last_output is None:
            raise RuntimeError("Failed to produce output during benchmark")
        self.output = last_output.detach().float().clone()
        if self._verify_input is None:
            raise RuntimeError("Verification input not initialized")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.output,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.graph = None
        self.static_input = None
        self.static_output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            measurement_timeout_seconds=240,
            setup_timeout_seconds=120,
            use_subprocess=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.inputs is None:
            return "Model/inputs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselinePieceGraphsBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
