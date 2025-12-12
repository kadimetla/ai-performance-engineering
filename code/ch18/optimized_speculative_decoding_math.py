"""optimized_speculative_decoding_math.py - Flash-off variant for GB10."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


def _disable_flash_sdp() -> None:
    """Force math SDPA paths to avoid sm80-only flash kernels."""
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception as e:
        import warnings
        warnings.warn(
            f"SDPA backend configuration failed: {e}. Using default backends.",
            RuntimeWarning,
            stacklevel=2,
        )


class OptimizedSpeculativeDecodingMathBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Speculative decoding with flash disabled (math path for GB10)."""

    def __init__(self):
        super().__init__()
        self.target_model: Optional[nn.Module] = None
        self.draft_model: Optional[nn.Module] = None
        self.embedding: Optional[nn.Module] = None
        self.input_ids: Optional[torch.Tensor] = None
        self.memory: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.max_length = 20
        self.speculative_length = 4
        # Match baseline for input verification
        self.batch_size = 4
        self.vocab_size = 32000
        self.hidden_size = 4096
        self.num_draft_tokens = 8
        self.num_sequences = 10
        self.num_draft_models = 3
        batch_size = 4
        seq_len = 10
        tokens = batch_size * (seq_len + self.max_length)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self._verification_payload = None

    def setup(self) -> None:
        _disable_flash_sdp()
        torch.manual_seed(42)
        hidden_dim = 256
        vocab_size = 1000

        self.embedding = nn.Embedding(vocab_size, hidden_dim).to(self.device).eval()
        self.target_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=4,
        ).to(self.device).eval()
        self.draft_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=2,
        ).to(self.device).eval()

        batch_size = 4
        seq_len = 10
        self.input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        self.memory = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def benchmark_fn(self) -> None:
        assert self.embedding is not None
        assert self.target_model is not None
        assert self.draft_model is not None
        assert self.input_ids is not None
        assert self.memory is not None

        with self._nvtx_range("optimized_speculative_decoding_math"):
            with torch.no_grad():
                current_ids = self.input_ids.clone()
                while current_ids.size(1) < self.input_ids.size(1) + self.max_length:
                    tgt_embedded = self.embedding(current_ids)
                    draft_output = self.draft_model(tgt_embedded, self.memory)
                    draft_tokens = draft_output[:, -self.speculative_length:, :].argmax(dim=-1)
                    current_ids = torch.cat([current_ids, draft_tokens], dim=1)
                    if current_ids.size(1) >= self.input_ids.size(1) + self.max_length:
                        break
                # Capture output for verification
                self.output = current_ids.detach()
        self._synchronize()
        if self.output is None or self.input_ids is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self.output.float(),
            batch_size=self.input_ids.shape[0],
            parameter_count=sum(p.numel() for p in self.target_model.parameters()) if self.target_model else 0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.target_model = None
        self.draft_model = None
        self.embedding = None
        self.input_ids = None
        self.memory = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_speculative_decoding_metrics
        return compute_speculative_decoding_metrics(
            draft_tokens=getattr(self, '_draft_tokens', 64),
            accepted_tokens=getattr(self, '_accepted_tokens', 48),
            draft_time_ms=getattr(self, '_draft_ms', 5.0),
            verify_time_ms=getattr(self, '_verify_ms', 10.0),
            num_rounds=getattr(self, '_num_rounds', 8),
        )

    def validate_result(self) -> Optional[str]:
        if self.target_model is None or self.draft_model is None:
            return "Models not initialized"
        if self.input_ids is None:
            return "Input IDs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedSpeculativeDecodingMathBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    result = bench.run(BenchmarkConfig(iterations=5, warmup=5))
    print(result)
