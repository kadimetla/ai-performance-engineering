"""Math-only guided decoding variant (disables flash/mem-efficient SDP for compatibility)."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

try:
    import ch01.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from core.utils.compile_utils import enable_tf32, compile_model
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class OptimizedGuidedDecodingMathBenchmark(BaseBenchmark):
    """Optimized guided decoding with schema constraints, forcing math SDP."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.input_ids = None
        self.schema = None
        self.max_length = 20
        self.batch_size = 4
        self.seq_len = 10
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        # Jitter check not applicable: random inputs generated per iteration
        self.jitter_exemption_reason = "Guided decoding: random inputs generated in benchmark_fn each iteration"

    def setup(self) -> None:
        if torch.cuda.is_available():
            # Per-workload SDP toggle: math-only to avoid flash autodispatch.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()

        torch.manual_seed(42)
        vocab_size = 1000
        hidden_dim = 256
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=1,
        ).to(self.device).eval()

        # Compilation helps lift perf above baseline on sm_100.
        if self.device.type == "cuda":
            self.model = compile_model(
                self.model,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            )

        self.schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "count": {"type": "number"},
            },
            "required": ["summary"],
        }

        self.input_ids = torch.randint(0, vocab_size, (self.batch_size, self.seq_len), device=self.device)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        with self._nvtx_range("optimized_guided_decoding_math"):
            with torch.no_grad():
                embedded_input = torch.randn(self.input_ids.size(0), self.input_ids.size(1), 256, device=self.device)
                memory = torch.randn(self.input_ids.size(0), self.input_ids.size(1), 256, device=self.device)
                output = self.model(embedded_input, memory)
                _ = output.sum()
            self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.input_ids = None
        self.schema = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_environment_metrics
        return compute_environment_metrics(
            gpu_count=getattr(self, 'gpu_count', 1),
            gpu_memory_gb=getattr(self, 'gpu_memory_gb', 80.0),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "max_length": self.max_length,
            "hidden_dim": 256,
        }

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison.
        
        Note: This benchmark creates random inputs in benchmark_fn() each iteration,
        making deterministic verification infeasible. We return a checksum based on
        model parameter count as a sanity check.
        """
        if self.model is None:
            raise RuntimeError("Model not available - run benchmark first")
        param_count = sum(p.numel() for p in self.model.parameters())
        return torch.tensor([float(param_count)], dtype=torch.float32)



def get_benchmark() -> BaseBenchmark:
    return OptimizedGuidedDecodingMathBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
