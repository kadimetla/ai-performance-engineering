"""baseline_speculative_decoding.py - Baseline decoding without speculative execution in FlexAttention/KV cache context.

Demonstrates standard autoregressive decoding without speculative decoding optimization.
Speculative decoding: This baseline does not use speculative decoding.
Generates tokens one at a time, sequential and slow.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class BaselineSpeculativeDecodingBenchmark(BaseBenchmark):
    """Baseline: Standard autoregressive decoding (no speculative execution).
    
    Speculative decoding: This baseline does not use speculative decoding.
    Generates tokens one at a time, sequential and slow.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.input_ids = None
        self.memory = None
        self.max_length = 20
        batch_size = 4
        seq_len = 10
        tokens = batch_size * (seq_len + self.max_length)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model and input."""
        torch.manual_seed(42)
        # Baseline: Standard decoding - generate tokens one at a time
        # Speculative decoding predicts multiple tokens in parallel
        # This baseline does not use speculative decoding
        
        hidden_dim = 256
        vocab_size = 1000
        
        # TransformerDecoder requires embedding layer for input_ids
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=2
        )
        self.model = self.model.to(self.device).eval()
        self.embedding = self.embedding.to(self.device).eval()
        
        # Baseline: Standard decoding - sequential token generation
        batch_size = 4
        seq_len = 10
        self.input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        # Create dummy memory tensor for TransformerDecoder (encoder output)
        self.memory = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard autoregressive decoding."""
        with self._nvtx_range("baseline_speculative_decoding"):
            with torch.no_grad():
                # Baseline: Standard autoregressive decoding
                # Generate tokens one at a time (sequential)
                # No speculative decoding - cannot predict multiple tokens in parallel
                
                current_ids = self.input_ids.clone()
                for _ in range(self.max_length):
                    # Generate next token (sequential - no speculative decoding)
                    # TransformerDecoder requires embedded inputs and memory arguments
                    tgt_embedded = self.embedding(current_ids)
                    output = self.model(tgt_embedded, self.memory)
                    next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                    current_ids = torch.cat([current_ids, next_token], dim=1)
        self._synchronize()
                
                # Baseline: No speculative decoding
                # Sequential token generation (slow)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.embedding = None
        self.input_ids = None
        self.memory = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input_ids is None:
            return "Input IDs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineSpeculativeDecodingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineSpeculativeDecodingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Speculative Decoding")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
