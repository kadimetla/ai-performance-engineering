#!/usr/bin/env python3
"""Baseline: Sequential autoregressive decoding (NO speculative decoding).

This baseline generates tokens one at a time using ONLY the target model.
Compare with optimized version that uses speculative decoding for speedup.

The key insight: Sequential decoding requires N target model forward passes
for N tokens. Speculative decoding can achieve the same with fewer passes.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import sys
from pathlib import Path
import time

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkConfig, BenchmarkMode
from core.utils.logger import get_logger

logger = get_logger(__name__)
from core.benchmark.verification_mixin import VerificationPayloadMixin


class BaselineSpeculativeDecoding:
    """Baseline: Sequential autoregressive decoding with target model only.
    
    This does NOT use speculative decoding - it generates one token at a time
    using only the large target model. This is the standard approach that
    speculative decoding aims to improve upon.
    """
    
    def __init__(
        self,
        batch_size: int = 4,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_draft_tokens: int = 8,  # Used to match token count with optimized
        num_sequences: int = 10,
    ):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_draft_tokens = num_draft_tokens
        self.num_sequences = num_sequences
        # Total tokens to generate = num_sequences * num_draft_tokens
        # This ensures fair comparison with speculative decoding
        self.tokens_to_generate = num_sequences * num_draft_tokens
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_output = None  # For verification
        
        logger.info(f"Baseline Sequential Autoregressive Decoding")
        logger.info(f"  Tokens to generate: {self.tokens_to_generate}")
    
    def _create_target_model(self):
        """Create the target language model (large, accurate)."""
        class SimpleLM(nn.Module):
            def __init__(self, vocab_size, hidden_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.linear1 = nn.Linear(hidden_size, hidden_size)
                self.linear2 = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                x = torch.relu(self.linear1(x))
                logits = self.linear2(x)
                return logits
        
        return SimpleLM(self.vocab_size, self.hidden_size)
    
    def setup(self):
        """Initialize target model only (no draft model in baseline)."""
        # Target model (large) - this is what we use for sequential decode
        self.target_model = self._create_target_model().to(self.device).eval()
        
        # Initial input
        self.input_ids = torch.randint(
            0, self.vocab_size,
            (self.batch_size, 1),
            device=self.device
        )
        
        logger.info("Target model initialized (sequential baseline)")
    
    def _generate_one_token(self, current_ids: torch.Tensor) -> torch.Tensor:
        """Generate a single token using target model (sequential)."""
        with torch.no_grad():
            logits = self.target_model(current_ids[:, -1:])
            next_token = torch.argmax(logits, dim=-1)
        return next_token
    
    def run(self) -> Dict[str, float]:
        """Execute baseline sequential autoregressive decoding.
        
        This is the SLOW approach: Generate N tokens one at a time,
        each requiring a full target model forward pass.
        
        Cost: N * target_forward_time
        """
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        current_ids = self.input_ids.clone()
        
        # Generate tokens_to_generate tokens sequentially
        # Each token requires ONE target model forward pass
        for _ in range(self.tokens_to_generate):
            next_token = self._generate_one_token(current_ids)
            current_ids = torch.cat([current_ids, next_token], dim=1)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Capture output for verification
        self.last_output = current_ids.detach()
        
        # Calculate metrics
        tokens_generated = current_ids.shape[1] - self.input_ids.shape[1]
        tokens_per_sec = tokens_generated * self.batch_size / elapsed
        
        logger.info(f"Sequential decode: {tokens_generated} tokens")
        logger.info(f"Tokens/sec: {tokens_per_sec:.2f}")
        logger.info(f"Target forward passes: {tokens_generated} (one per token)")
        
        return {
            "latency_ms": elapsed * 1000,
            "tokens_per_sec": tokens_per_sec,
            "tokens_generated": tokens_generated,
            "target_forward_passes": tokens_generated,
        }

    def get_input_signature(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_draft_tokens": self.num_draft_tokens,
            "num_sequences": self.num_sequences,
        }
    
    def cleanup(self):
        """Clean up resources."""
        del self.target_model, self.input_ids
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 4,
    vocab_size: int = 32000,
    hidden_size: int = 4096,
    num_draft_tokens: int = 8,
    num_sequences: int = 10,
    profile: str = "none",
    return_benchmark: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Run baseline speculative decoding benchmark."""
    
    benchmark = BaselineSpeculativeDecoding(
        batch_size=batch_size,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_draft_tokens=num_draft_tokens,
        num_sequences=num_sequences,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(
        iterations=3,
        warmup=5,
        profile_mode=profile,
    )
    
    harness = BenchmarkHarness(mode=BenchmarkMode.INFERENCE, config=config)
    
    result = harness.benchmark(
        benchmark.run,
        name="baseline_speculative_decoding"
    )
    
    metrics = benchmark.run()
    
    if return_benchmark:
        # Return benchmark object for output extraction (don't cleanup yet)
        return benchmark, {
            "mean_time_ms": result.timing.mean_ms,
            **metrics,
        }
    
    benchmark.cleanup()
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        **metrics,
    }


class BaselineSpeculativeDecodingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness wrapper to expose the baseline decode loop."""

    def __init__(self):
        super().__init__()
        self._metrics: Dict[str, Any] = {}
        self._inner_benchmark = None
        self.output = None
        self._verification_payload = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def benchmark_fn(self) -> None:
        self._inner_benchmark, self._metrics = run_benchmark(return_benchmark=True)
        self.output = self._inner_benchmark.last_output
        self._synchronize()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output")
        # Use the generated token ids as verification output.

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input_ids": self._inner_benchmark.input_ids},
            output=self.output.float(),
            batch_size=self._inner_benchmark.input_ids.shape[0],
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        if self._inner_benchmark is not None:
            self._inner_benchmark.cleanup()
            self._inner_benchmark = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        # Single iteration; run_benchmark internally uses its own harness timing.
        return BenchmarkConfig(iterations=1, warmup=0)

    def get_custom_metrics(self) -> Dict[str, Any]:
        return self._metrics


def get_benchmark() -> BaseBenchmark:
    return BaselineSpeculativeDecodingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
