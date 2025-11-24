#!/usr/bin/env python3
"""Baseline: Speculative decoding without optimization.

Demonstrates basic speculative decoding with single draft model.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import sys
from pathlib import Path
import time

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.python.benchmark_harness import BenchmarkHarness, BenchmarkConfig, BenchmarkMode
from common.python.logger import get_logger

logger = get_logger(__name__)


class BaselineSpeculativeDecoding:
    """Baseline speculative decoding with single draft."""
    
    def __init__(
        self,
        batch_size: int = 4,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_draft_tokens: int = 4,
        num_sequences: int = 10,
    ):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_draft_tokens = num_draft_tokens
        self.num_sequences = num_sequences
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Baseline Speculative Decoding")
        logger.info(f"  Draft tokens: {num_draft_tokens}")
    
    def _create_simple_model(self, is_draft: bool = False):
        """Create simplified language model."""
        class SimpleLM(nn.Module):
            def __init__(self, vocab_size, hidden_size, is_draft):
                super().__init__()
                # Draft model is smaller
                h = hidden_size // 4 if is_draft else hidden_size
                
                self.embedding = nn.Embedding(vocab_size, h)
                self.linear1 = nn.Linear(h, h)
                self.linear2 = nn.Linear(h, vocab_size)
            
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                x = torch.relu(self.linear1(x))
                logits = self.linear2(x)
                return logits
        
        return SimpleLM(self.vocab_size, self.hidden_size, is_draft)
    
    def setup(self):
        """Initialize models."""
        # Target model (large, accurate)
        self.target_model = self._create_simple_model(is_draft=False).to(self.device).eval()
        
        # Draft model (small, fast)
        self.draft_model = self._create_simple_model(is_draft=True).to(self.device).eval()
        
        # Initial input
        self.input_ids = torch.randint(
            0, self.vocab_size,
            (self.batch_size, 1),
            device=self.device
        )
        
        logger.info("Models initialized")
    
    def _draft_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate draft tokens using small model."""
        draft_tokens = []
        current_ids = input_ids
        
        for _ in range(self.num_draft_tokens):
            with torch.no_grad():
                logits = self.draft_model(current_ids[:, -1:])
                next_token = torch.argmax(logits, dim=-1)
                draft_tokens.append(next_token)
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        return torch.cat(draft_tokens, dim=1)  # [batch, num_draft_tokens]
    
    def _verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Verify draft tokens with target model."""
        # Concatenate input + draft
        combined = torch.cat([input_ids, draft_tokens], dim=1)
        
        # Target model forward
        with torch.no_grad():
            logits = self.target_model(combined)
        
        # Check each draft token
        accepted = 0
        for i in range(self.num_draft_tokens):
            target_token = torch.argmax(logits[:, -(self.num_draft_tokens - i + 1), :], dim=-1)
            if torch.all(target_token == draft_tokens[:, i]):
                accepted += 1
            else:
                break
        
        # Return accepted tokens + next token from target
        if accepted < self.num_draft_tokens:
            final_tokens = draft_tokens[:, :accepted]
            next_token = torch.argmax(logits[:, -(self.num_draft_tokens - accepted), :], dim=-1)
            final_tokens = torch.cat([final_tokens, next_token.unsqueeze(1)], dim=1)
        else:
            final_tokens = draft_tokens
        
        return final_tokens, accepted
    
    def run(self) -> Dict[str, float]:
        """Execute baseline speculative decoding."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        current_ids = self.input_ids
        total_accepted = 0
        total_drafted = 0
        
        for _ in range(self.num_sequences):
            # Draft phase
            draft_tokens = self._draft_tokens(current_ids)
            total_drafted += self.num_draft_tokens
            
            # Verify phase
            accepted_tokens, num_accepted = self._verify_tokens(current_ids, draft_tokens)
            total_accepted += num_accepted
            
            # Update sequence
            current_ids = torch.cat([current_ids, accepted_tokens], dim=1)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Calculate metrics
        acceptance_rate = (total_accepted / total_drafted) * 100
        tokens_generated = current_ids.shape[1] - self.input_ids.shape[1]
        tokens_per_sec = tokens_generated * self.batch_size / elapsed
        
        logger.info(f"Acceptance rate: {acceptance_rate:.1f}%")
        logger.info(f"Tokens/sec: {tokens_per_sec:.2f}")
        
        return {
            "latency_ms": elapsed * 1000,
            "tokens_per_sec": tokens_per_sec,
            "acceptance_rate": acceptance_rate,
            "tokens_generated": tokens_generated,
        }
    
    def cleanup(self):
        """Clean up resources."""
        del self.target_model, self.draft_model, self.input_ids
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 4,
    vocab_size: int = 32000,
    hidden_size: int = 4096,
    num_draft_tokens: int = 4,
    num_sequences: int = 10,
    profile: str = "none",
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
        warmup=1,
        profile_mode=profile,
    )
    
    harness = BenchmarkHarness(mode=BenchmarkMode.INFERENCE, config=config)
    
    result = harness.benchmark(
        benchmark.run,
        name="baseline_speculative_decoding"
    )
    
    metrics = benchmark.run()
    benchmark.cleanup()
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        **metrics,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Speculative Decoding")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--profile", type=str, default="none")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_draft_tokens=args.num_draft_tokens,
        num_sequences=args.num_sequences,
        profile=args.profile,
    )
    
    print(f"\n{'='*60}")
    print(f"Baseline Speculative Decoding Results")
    print(f"{'='*60}")
    print(f"Tokens/sec: {result['tokens_per_sec']:.2f}")
    print(f"Acceptance rate: {result['acceptance_rate']:.1f}%")
    print(f"Tokens generated: {result['tokens_generated']}")
    print(f"Mean time: {result['mean_time_ms']:.2f} ms")
    print(f"{'='*60}\n")
