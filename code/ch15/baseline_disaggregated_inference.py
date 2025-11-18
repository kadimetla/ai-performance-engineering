"""Baseline CLI hook for the disaggregated inference walkthrough."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402
from ch15.baseline_moe_inference import BaselineMoeInferenceBenchmark  # noqa: E402


class _DisaggregatedInferenceBenchmark(BaselineMoeInferenceBenchmark):
    """Shared harness that simulates prefill/decode split execution."""

    def __init__(self, *, speculative_window: int, decode_parallelism: int):
        super().__init__()
        self.speculative_window = max(1, speculative_window)
        self.decode_parallelism = max(1, decode_parallelism)
        self._disagg_history: Dict[str, List[float]] = {
            "prefill_ms": [],
            "decode_ms": [],
        }

    def setup(self) -> None:
        super().setup()
        if self.model is not None:
            self.model.to(device=self.device, dtype=self.config.dtype_obj)

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if self.model is None or self.prompts is None or self.kv_cache is None:
            raise RuntimeError("Model or prompts not initialized")

        cfg = self.config
        ttft_samples: List[float] = []
        decode_samples: List[float] = []

        with torch.no_grad():
            with self._nvtx_range("disagg_prefill"):
                start = time.perf_counter()
                hidden, logits = self.model.prefill(self.prompts, kv_cache=self.kv_cache, cache_start=0)
                torch.cuda.synchronize(self.device)
                ttft_samples.append((time.perf_counter() - start) * 1000.0)

            seeds = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            context_position = cfg.context_window
            step = 0

            while step < cfg.decode_tokens:
                tokens_now = min(self.speculative_window, cfg.decode_tokens - step)
                start = time.perf_counter()

                for bucket in range(tokens_now):
                    position = context_position + step + bucket
                    _hidden, decode_logits = self.model.decode(
                        seeds,
                        kv_cache=self.kv_cache,
                        position=position,
                    )
                    torch.cuda.synchronize(self.device)
                    seeds = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)

                decode_samples.append((time.perf_counter() - start) * 1000.0)
                step += tokens_now

        total_ms = sum(ttft_samples) + sum(decode_samples)
        throughput = cfg.tokens_per_iteration / max(total_ms / 1000.0, 1e-6)
        nvlink_gbps = 0.0
        if ttft_samples:
            bytes_moved = cfg.batch_size * cfg.context_window * cfg.hidden_size * self._dtype_bytes
            nvlink_gbps = (bytes_moved * 8.0 / 1e9) / (ttft_samples[0] / 1000.0)

        self._history["ttft"].extend(ttft_samples)
        self._history["tpot"].extend(decode_samples)
        self._history["throughput"].append(throughput)
        self._history["nvlink"].append(nvlink_gbps)

        self._disagg_history["prefill_ms"].extend(ttft_samples)
        self._disagg_history["decode_ms"].extend(decode_samples)
        return {"prefill_ms": ttft_samples, "decode_ms": decode_samples}

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        parent_metrics = super().get_custom_metrics()
        if not self._disagg_history["prefill_ms"]:
            return parent_metrics
        extra = {
            "disagg.prefill_ms": float(statistics.mean(self._disagg_history["prefill_ms"])),
            "disagg.decode_step_ms": float(statistics.mean(self._disagg_history["decode_ms"])),
            "disagg.speculative_window": float(self.speculative_window),
            "disagg.decode_parallelism": float(self.decode_parallelism),
        }
        if parent_metrics is None:
            return extra
        parent_metrics.update(extra)
        return parent_metrics

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=4, warmup=1)


class BaselineDisaggregatedInferenceBenchmark(_DisaggregatedInferenceBenchmark):
    """Sequential prefill/decode simulation (no overlap)."""

    def __init__(self) -> None:
        super().__init__(speculative_window=1, decode_parallelism=1)


def get_benchmark():
    return BaselineDisaggregatedInferenceBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(result)
