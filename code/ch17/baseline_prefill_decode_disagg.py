"""Baseline monolithic prefill+decode benchmark (native to Chapter 17)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class SimpleLLM(nn.Module):
    """Simplified LLM used for monolithic prefill+decode."""

    def __init__(self, hidden_dim: int = 1024, num_layers: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def prefill(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """Prefill over the full prompt (compute-bound path)."""
        x = torch.randn(
            prompt_tokens.size(0),
            prompt_tokens.size(1),
            self.hidden_dim,
            device=prompt_tokens.device,
            dtype=torch.bfloat16,
        )
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x[:, -1:, :]

    def decode(self, kv_cache: torch.Tensor, num_tokens: int = 16) -> torch.Tensor:
        """Decode a small number of tokens (memory-bound path)."""
        outputs = []
        x = kv_cache
        for _ in range(num_tokens):
            for layer in self.layers:
                x = torch.relu(layer(x))
            outputs.append(x)
        return torch.cat(outputs, dim=1)


class BaselinePrefillDecodeMonolithicBenchmark(BaseBenchmark):
    """Monolithic prefill+decode baseline (no disaggregation)."""

    def __init__(self):
        super().__init__()
        self.model: Optional[SimpleLLM] = None
        self.prompt: Optional[torch.Tensor] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self._history: Dict[str, List[float]] = {"ttft": [], "tpot": []}
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=256 + 16,
        )

    def setup(self) -> None:
        self.model = SimpleLLM(hidden_dim=1024, num_layers=12).to(self.device).to(torch.bfloat16).eval()
        self.prompt = torch.randint(0, 10000, (1, 256), device=self.device)
        with torch.no_grad():
            self.kv_cache = self.model.prefill(self.prompt)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if self.model is None or self.prompt is None:
            raise RuntimeError("Model or prompt not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())

        with nvtx_range("inference_monolithic", enable=enable_nvtx):
            with torch.no_grad():
                request_start = self._record_start()

                torch.cuda.synchronize(self.device)
                prefill_start = self._record_start()
                kv_cache = self.model.prefill(self.prompt)
                torch.cuda.synchronize(self.device)
                ttft_ms = self._record_stop(request_start)

                num_tokens = 16
                tpot_times_ms: List[float] = []
                token_output = kv_cache
                for _ in range(num_tokens):
                    token_start = self._record_start()
                    token_output = self.model.decode(token_output[:, -1:, :], num_tokens=1)
                    torch.cuda.synchronize(self.device)
                    tpot_times_ms.append(self._record_stop(token_start))

                self._history["ttft"].append(ttft_ms)
                self._history["tpot"].extend(tpot_times_ms)
                return {
                    "ttft_times_ms": [ttft_ms],
                    "tpot_times_ms": tpot_times_ms,
                }

    def teardown(self) -> None:
        self.model = None
        self.prompt = None
        self.kv_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self._history["ttft"]:
            return None
        return {
            "monolithic.ttft_ms": float(sum(self._history["ttft"]) / len(self._history["ttft"])),
            "monolithic.tpot_mean_ms": float(sum(self._history["tpot"]) / len(self._history["tpot"])),
        }

    def validate_result(self) -> Optional[str]:
        if not self._history["ttft"]:
            return "No TTFT samples recorded"
        if not self._history["tpot"]:
            return "No TPOT samples recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery (Chapter 17 baseline)."""
    return BaselinePrefillDecodeMonolithicBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(f"Prefill/decode baseline mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
