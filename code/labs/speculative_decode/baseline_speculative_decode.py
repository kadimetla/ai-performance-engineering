"""Baseline speculative decoding: Standard autoregressive generation (no speculation).

This baseline generates tokens one at a time without draft model assistance.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding benchmark."""
    batch_size: int = 1
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 4
    prompt_length: int = 64
    decode_length: int = 128
    # Draft model settings (not used in baseline)
    draft_length: int = 4
    use_speculation: bool = False


class SimpleLM(nn.Module):
    """Simplified language model for benchmarking."""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.0,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor, past_hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass returning logits for last token."""
        x = self.embedding(input_ids)
        if past_hidden is not None:
            x = x + past_hidden
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x[:, -1:, :])  # Only compute logits for last position
        return logits, x[:, -1:, :]


class BaselineSpeculativeDecodeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Standard autoregressive decoding (no speculation).
    
    Generates tokens one at a time using greedy decoding.
    """
    
    def __init__(self, config: Optional[SpeculativeConfig] = None):
        super().__init__()
        self.config = config or SpeculativeConfig()
        self.model: Optional[SimpleLM] = None
        self.prompt_ids: Optional[torch.Tensor] = None
        self.tokens_generated: int = 0
        self.output: Optional[torch.Tensor] = None
        self.parameter_count = 0
        
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(self.config.batch_size * self.config.decode_length),
        )
    
    def setup(self) -> None:
        """Initialize model and prompt."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Create model
        self.model = SimpleLM(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
        ).to(self.device, dtype=torch.bfloat16)
        self.model.eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        # Create prompt
        self.prompt_ids = torch.randint(
            0, self.config.vocab_size,
            (self.config.batch_size, self.config.prompt_length),
            device=self.device,
        )
        
        # Warmup
        self._warmup()
        torch.cuda.synchronize()
    
    def _warmup(self) -> None:
        """Warmup model with a few iterations."""
        with torch.no_grad():
            for _ in range(3):
                self._generate(max_tokens=8)
    
    @torch.no_grad()
    def _generate(self, max_tokens: int) -> torch.Tensor:
        """Standard autoregressive generation (baseline)."""
        input_ids = self.prompt_ids.clone()
        hidden = None
        
        for _ in range(max_tokens):
            # Single forward pass for one token
            logits, hidden = self.model(input_ids[:, -1:], hidden)
            
            # Greedy selection
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            # Append to sequence (simplified - in practice we'd use KV cache)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def benchmark_fn(self) -> None:
        """Run baseline autoregressive decoding."""
        output_ids = self._generate(max_tokens=self.config.decode_length)
        self.tokens_generated = output_ids.shape[1] - self.prompt_ids.shape[1]
        self.output = output_ids.detach()
        self._synchronize()
        if self.prompt_ids is None or self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"prompt_ids": self.prompt_ids.detach()},
            output=self.output,
            batch_size=self.config.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={"bf16": True, "fp16": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.0, 0.0),
        )
    
    def teardown(self) -> None:
        """Clean up."""
        self.model = None
        self.prompt_ids = None
        torch.cuda.empty_cache()
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def validate_result(self) -> Optional[str]:
        if self.tokens_generated != self.config.decode_length:
            return f"Expected {self.config.decode_length} tokens, got {self.tokens_generated}"
        return None
    
    def get_custom_metrics(self) -> Optional[dict]:
        return {
            "speculative_decode.mode": "baseline",
            "speculative_decode.speculation_enabled": 0.0,
            "speculative_decode.tokens_generated": float(self.tokens_generated),
        }

def get_benchmark() -> BaseBenchmark:
    return BaselineSpeculativeDecodeBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
