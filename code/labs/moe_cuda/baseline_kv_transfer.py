"""labs.moe_cuda/baseline_kv_transfer.py - Sequential KV cache transfers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class BaselineKVTransferBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Sequential KV transfers (no overlap between compute and NVLink copies)."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024  # Must match optimized variants
        self.chunk_size = 256
        # Baseline for both overlap and graphs variants.
        self.num_chunks = 32
        self.dtype = torch.float16
        self.input_chunks: Optional[torch.Tensor] = None
        self.weight: Optional[torch.Tensor] = None
        self.workspace: Optional[torch.Tensor] = None
        self.kv_dest: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        tokens = self.num_chunks * self.chunk_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_chunks),
            tokens_per_iteration=float(tokens),
        )
        self._history: Dict[str, List[float]] = {"latency_ms": []}

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("labs.moe_cuda KV transfer requires CUDA")

        import gc
        
        # Clean up any leftover CUDA graph state from previous benchmarks
        # to prevent "Offset increment outside graph capture" errors
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        try:
            if hasattr(torch.cuda, 'graph_pool_trim'):
                torch.cuda.graph_pool_trim()
        except Exception:
            pass
        
        # CRITICAL: Reset CUDA random number generator state
        # CUDA graphs capture the RNG offset, which causes "Offset increment 
        # outside graph capture" errors when using torch.randn
        try:
            device_idx = torch.cuda.current_device()
            gen = torch.cuda.default_generators[device_idx]
            # set_offset(0) properly resets the graph capture state
            gen.set_offset(0)
            gen.manual_seed(42)
        except Exception:
            pass
        
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        
        try:
            torch._inductor.cudagraph_trees.reset_cudagraph_trees()
        except Exception:
            pass

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Create tensors using CPU randn + to(device) to avoid CUDA RNG graph capture issues
        self.input_chunks = torch.randn(
            self.num_chunks,
            self.chunk_size,
            self.hidden_size,
            dtype=self.dtype,
        ).to(self.device)
        self.weight = torch.randn(self.hidden_size, self.hidden_size, dtype=self.dtype).to(self.device)
        self.workspace = torch.zeros_like(self.input_chunks)
        self.kv_dest = torch.zeros_like(self.input_chunks)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if any(t is None for t in (self.input_chunks, self.weight, self.workspace, self.kv_dest)):
            raise RuntimeError("Buffers not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_cuda_kv_baseline", enable=enable_nvtx):
            latencies: List[float] = []
            for i in range(self.num_chunks):
                start = self._record_start()
                chunk = self.input_chunks[i]
                out = torch.matmul(chunk, self.weight)
                self.workspace[i].copy_(out)
                self.kv_dest[i].copy_(self.workspace[i])
                torch.cuda.synchronize(self.device)
                latencies.append(self._record_stop(start))
            self._history["latency_ms"].extend(latencies)
        # Verification: capture first chunk output (common across optimized variants)
        self.output = self.kv_dest[0, :1, : min(8, self.hidden_size)].detach().float().clone()
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")
        meta = torch.tensor([self.hidden_size], dtype=torch.int64, device="cpu")
        self._set_verification_payload(
            inputs={"meta": meta},
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={},
            output_tolerance=(0.1, 1.0),
        )
        return {"kv_transfer_ms": latencies}

    def teardown(self) -> None:
        self.input_chunks = None
        self.weight = None
        self.workspace = None
        self.kv_dest = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)  # Min warmup for CUDA

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self._history["latency_ms"]:
            return None
        return {"kv_transfer.mean_ms": float(sum(self._history["latency_ms"]) / len(self._history["latency_ms"]))}

    def validate_result(self) -> Optional[str]:
        if any(t is None for t in (self.input_chunks, self.weight, self.workspace, self.kv_dest)):
            return "Buffers not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    return BaselineKVTransferBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
