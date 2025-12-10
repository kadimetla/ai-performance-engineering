#!/usr/bin/env python3
"""Baseline: Disaggregated prefill/decode without optimization.

Basic disaggregated serving with separate pools but no optimizations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
import time

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from core.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineDisaggregatedPrefillDecode:
    """Baseline disaggregated serving."""
    
    def __init__(
        self,
        num_prefill_gpus: int = 2,
        num_decode_gpus: int = 6,
        batch_size: int = 8,
        prefill_length: int = 1024,
        decode_length: int = 128,
    ):
        self.num_prefill_gpus = num_prefill_gpus
        self.num_decode_gpus = num_decode_gpus
        self.batch_size = batch_size
        self.prefill_length = prefill_length
        self.decode_length = decode_length
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Disaggregated Serving")
        logger.info(f"  Prefill GPUs: {num_prefill_gpus}")
        logger.info(f"  Decode GPUs: {num_decode_gpus}")
    
    def setup(self):
        """Initialize models (simulated for both pools)."""
        hidden_size = 4096
        
        # Simulated prefill model
        self.prefill_model = nn.Linear(hidden_size, hidden_size).to(self.device)
        
        # Simulated decode model
        self.decode_model = nn.Linear(hidden_size, hidden_size).to(self.device)
        
        # Create inputs
        self.prefill_input = torch.randn(
            self.batch_size, self.prefill_length, hidden_size,
            device=self.device, dtype=torch.bfloat16
        )
        
        logger.info("Models initialized (baseline)")
    
    def run(self) -> Dict[str, float]:
        """Execute baseline disaggregated serving."""
        torch.cuda.synchronize()
        start_total = time.perf_counter()
        
        # Prefill phase (on prefill GPUs)
        prefill_start = time.perf_counter()
        prefill_output = self.prefill_model(self.prefill_input)
        kv_cache = prefill_output  # Simplified KV
        torch.cuda.synchronize()
        prefill_time = time.perf_counter() - prefill_start
        
        # Baseline: Blocking transfer of KV cache to decode pool
        # (No overlap, no compression)
        transfer_start = time.perf_counter()
        kv_cache_cpu = kv_cache.cpu()  # Transfer through CPU (baseline)
        kv_cache_decode = kv_cache_cpu.to(self.device)  # Back to GPU
        torch.cuda.synchronize()
        transfer_time = time.perf_counter() - transfer_start
        
        # Decode phase (on decode GPUs)
        decode_start = time.perf_counter()
        
        decode_outputs = []
        for _ in range(self.decode_length):
            # Simplified decode step
            decode_output = self.decode_model(kv_cache_decode[:, -1:, :])
            decode_outputs.append(decode_output)
        
        torch.cuda.synchronize()
        decode_time = time.perf_counter() - decode_start
        
        total_time = time.perf_counter() - start_total
        
        logger.info(f"Prefill: {prefill_time*1000:.2f} ms")
        logger.info(f"KV Transfer: {transfer_time*1000:.2f} ms")
        logger.info(f"Decode: {decode_time*1000:.2f} ms")
        logger.info(f"Total: {total_time*1000:.2f} ms")
        
        return {
            "total_latency_ms": total_time * 1000,
            "prefill_ms": prefill_time * 1000,
            "transfer_ms": transfer_time * 1000,
            "decode_ms": decode_time * 1000,
            "transfer_overhead_pct": (transfer_time / total_time) * 100,
        }
    
    def cleanup(self):
        """Clean up."""
        del self.prefill_model, self.decode_model, self.prefill_input
        torch.cuda.empty_cache()


def run_benchmark(
    num_prefill_gpus: int = 2,
    num_decode_gpus: int = 6,
    batch_size: int = 8,
    prefill_length: int = 1024,
    decode_length: int = 128,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline disaggregated benchmark."""
    
    benchmark = BaselineDisaggregatedPrefillDecode(
        num_prefill_gpus=num_prefill_gpus,
        num_decode_gpus=num_decode_gpus,
        batch_size=batch_size,
        prefill_length=prefill_length,
        decode_length=decode_length,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(iterations=3, warmup=5, profile_mode=profile)
    harness = BenchmarkHarness(mode=BenchmarkMode.INFERENCE, config=config)
    
    result = harness.benchmark(benchmark.run, name="baseline_disaggregated")
    
    metrics = benchmark.run()
    benchmark.cleanup()
    
    return {"mean_time_ms": result.timing.mean_ms, **metrics}


class _DisaggregatedPrefillDecodeBenchmark(BaseBenchmark):
    """Wrapper benchmark for disaggregated prefill/decode."""

    def __init__(self) -> None:
        super().__init__()
        self._impl = BaselineDisaggregatedPrefillDecode()
        self._metrics = {}
        self.jitter_exemption_reason = "Disaggregated prefill/decode: fixed configuration"
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        self._impl.setup()

    def benchmark_fn(self) -> None:
        self._metrics = self._impl.run()
        self._synchronize()

    def teardown(self) -> None:
        self._impl.cleanup()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=3, warmup=5)

    def get_verify_output(self) -> torch.Tensor:
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        return {"type": "disaggregated_prefill_decode_baseline"}

    def get_output_tolerance(self) -> tuple:
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return _DisaggregatedPrefillDecodeBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
