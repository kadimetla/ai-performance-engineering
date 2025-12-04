#!/usr/bin/env python3
"""Baseline: vLLM v1 integration without optimization.

Demonstrates basic vLLM v1 usage without advanced features like:
- Bucketed CUDA graphs
- Optimized KV cache management
- Prefix caching
"""

import torch
import time
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkConfig, BenchmarkMode
from core.utils.logger import get_logger

logger = get_logger(__name__)

# Check for vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, using simulation mode")


class BaselineVLLMV1Integration:
    """Baseline vLLM v1 without optimizations."""
    
    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        max_tokens: int = 128,
        batch_size: int = 8,
        use_vllm: bool = True,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        
        if not self.use_vllm:
            logger.info("Running in simulation mode (vLLM not available)")
    
    def setup(self):
        """Initialize vLLM model."""
        if self.use_vllm:
            import gc
            import torch
            
            # Force cleanup before vLLM initialization to avoid resource conflicts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            try:
                # Baseline: No CUDA graphs, no prefix caching
                self.llm = LLM(
                    model=self.model_name,
                    enforce_eager=True,  # Disable CUDA graphs (baseline)
                    enable_prefix_caching=False,  # Disable prefix caching
                    gpu_memory_utilization=0.7,  # Lower to avoid OOM during engine init
                    dtype="bfloat16",
                    tensor_parallel_size=1,  # Single GPU to avoid coordination issues
                    max_model_len=512,  # Limit context length to reduce memory
                )
                
                logger.info(f"Loaded model: {self.model_name}")
                logger.info("Baseline config: eager execution, no prefix caching")
            except RuntimeError as e:
                error_msg = str(e)
                if "Engine core initialization failed" in error_msg:
                    raise RuntimeError(
                        f"SKIPPED: vLLM engine initialization failed - likely due to GPU resource "
                        f"contention from previous benchmark. Original error: {error_msg}"
                    ) from e
                raise
        
        # Create prompts
        self.prompts = [
            f"Once upon a time in a land far away, there was a {i}" 
            for i in range(self.batch_size)
        ]
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=0.8,
            top_p=0.95,
        )
    
    def run(self) -> Dict[str, float]:
        """Execute baseline vLLM inference."""
        if not self.use_vllm:
            # Simulation mode - use actual model inference without vLLM optimizations
            logger.warning("vLLM not available - benchmark requires vLLM installation")
            raise RuntimeError("vLLM required for this benchmark. Install with: pip install vllm")
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Generate
        outputs = self.llm.generate(self.prompts, self.sampling_params)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        elapsed = end - start
        
        # Calculate metrics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / elapsed
        mean_latency_ms = (elapsed / len(self.prompts)) * 1000
        
        logger.info(f"Throughput: {throughput:.2f} tokens/sec")
        logger.info(f"Mean latency: {mean_latency_ms:.2f} ms")
        
        return {
            "mean_latency_ms": mean_latency_ms,
            "throughput_tokens_per_sec": throughput,
            "total_tokens": total_tokens,
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.use_vllm and hasattr(self, 'llm'):
            del self.llm
        torch.cuda.empty_cache()


def run_benchmark(
    model_name: str = "facebook/opt-125m",
    max_tokens: int = 128,
    batch_size: int = 8,
    use_vllm: bool = True,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline vLLM v1 benchmark."""
    
    benchmark = BaselineVLLMV1Integration(
        model_name=model_name,
        max_tokens=max_tokens,
        batch_size=batch_size,
        use_vllm=use_vllm,
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
        name="baseline_vllm_v1_integration"
    )
    
    metrics = benchmark.run()
    benchmark.cleanup()
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        **metrics,
        "model": model_name,
        "optimizations": "none",
    }


class BaselineVLLMV1IntegrationBenchmark(BaseBenchmark):
    """Harness wrapper for the baseline vLLM integration."""

    def __init__(self):
        super().__init__()
        self._metrics: Dict[str, Any] = {}

    def benchmark_fn(self) -> None:
        self._metrics = run_benchmark()
        self._synchronize()

    def get_config(self) -> BenchmarkConfig:
        # Run once; inner harness handles iterations/warmup.
        return BenchmarkConfig(iterations=1, warmup=0)

    def get_custom_metrics(self) -> Dict[str, Any]:
        return self._metrics


def get_benchmark() -> BaseBenchmark:
    return BaselineVLLMV1IntegrationBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
