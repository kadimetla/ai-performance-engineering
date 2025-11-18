"""Optimized MoE inference benchmark inspired by vLLM + NVIDIA Dynamo + SGLang."""

from __future__ import annotations

import argparse
import math
import os
import contextlib
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from common.python.gpu_memory_logger import (  # noqa: E402
    GpuMemoryLogger,
    resolve_gpu_log_interval,
    resolve_gpu_log_path,
)
from common.python.gpu_telemetry import query_gpu_telemetry  # noqa: E402
from common.python.moe_inference import (  # noqa: E402
    MoeInferenceConfig,
    SimpleMoEGPT,
    dtype_bytes,
    env_override_float,
    env_override_int,
)
from ch17.dynamic_routing import (  # noqa: E402
    DisaggregatedRouter,
    Priority,
    Request,
    WorkerMetrics,
)


class PagedKVCache:
    """Lightweight paged KV cache for benchmarking."""

    def __init__(
        self,
        *,
        batch_size: int,
        max_tokens: int,
        hidden: int,
        dtype: torch.dtype,
        device: torch.device,
        page_size: int,
    ) -> None:
        self.buffer = torch.zeros(batch_size, max_tokens, hidden, dtype=dtype, device=device)
        self.page_size = max(1, page_size)
        self.max_tokens = max_tokens
        self.tokens_written = 0
        self.page_faults = 0

    def reset(self) -> None:
        self.buffer.zero_()
        self.tokens_written = 0
        self.page_faults = 0

    def mark_prefill(self, tokens: int) -> None:
        self._update_usage(position=0, length=tokens)

    def write(self, position: int, values: torch.Tensor) -> None:
        length = values.size(1)
        self.buffer[:, position:position + length].copy_(values)
        self._update_usage(position=position, length=length)

    def _update_usage(self, position: int, length: int) -> None:
        prev_tokens = self.tokens_written
        self.tokens_written = max(self.tokens_written, position + length)
        prev_pages = math.ceil(prev_tokens / self.page_size)
        new_pages = math.ceil(self.tokens_written / self.page_size)
        if new_pages > prev_pages:
            self.page_faults += (new_pages - prev_pages)

    @property
    def occupancy_ratio(self) -> float:
        if self.max_tokens <= 0:
            return 0.0
        return min(1.0, self.tokens_written / self.max_tokens)

    @property
    def memory_gb(self) -> float:
        return self.buffer.element_size() * self.buffer.nelement() / (1024 ** 3)


class SpeculativeDecoder:
    """SGLang-style speculative decode helper using draft and target models."""

    def __init__(self, target_model: SimpleMoEGPT, draft_model: SimpleMoEGPT, chunk_size: int = 4):
        self.target_model = target_model
        self.draft_model = draft_model
        self.chunk_size = max(1, chunk_size)
        self.accepted_tokens = 0
        self.total_tokens = 0

    def reset(self) -> None:
        self.accepted_tokens = 0
        self.total_tokens = 0

    def decode(
        self,
        seed_tokens: torch.Tensor,
        total_tokens: int,
        paged_cache: PagedKVCache,
        base_position: int,
    ) -> Tuple[torch.Tensor, List[float]]:
        tokens = seed_tokens
        emitted = 0
        per_token_times: List[float] = []

        with torch.no_grad():
            while emitted < total_tokens:
                chunk = min(self.chunk_size, total_tokens - emitted)
                for _ in range(chunk):
                    start = time.perf_counter()
                    draft_hidden, draft_logits = self.draft_model.decode(tokens)
                    candidate = torch.argmax(draft_logits[:, -1, :], dim=-1, keepdim=True)

                    target_hidden, target_logits = self.target_model.decode(
                        tokens,
                        kv_cache=paged_cache.buffer,
                        position=base_position + emitted,
                    )
                    paged_cache.write(base_position + emitted, target_hidden)

                    target_next = torch.argmax(target_logits[:, -1, :], dim=-1, keepdim=True)
                    matches = candidate.eq(target_next)
                    self.accepted_tokens += matches.sum().item()
                    self.total_tokens += matches.numel()
                    tokens = torch.where(matches, candidate, target_next)

                    torch.cuda.synchronize()
                    per_token_times.append((time.perf_counter() - start) * 1000.0)
                    emitted += 1

                    if not matches.all():
                        break
        return tokens, per_token_times

    def acceptance_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_tokens


class VLLMMoEInferenceBenchmark(BaseBenchmark):
    """Optimized MoE inference benchmark with paged KV cache + speculative decode."""

    def __init__(self) -> None:
        super().__init__()
        self.config = self._build_config()
        self.model: Optional[SimpleMoEGPT] = None
        self.draft_model: Optional[SimpleMoEGPT] = None
        self.prompts: Optional[torch.Tensor] = None
        self.router = DisaggregatedRouter(config_path=os.getenv("DYNAMO_ROUTER_CONFIG"))
        self.paged_cache: Optional[PagedKVCache] = None
        self.spec_decoder: Optional[SpeculativeDecoder] = None
        self.prefill_workers = env_override_int("OPT_MOE_PREFILL_WORKERS", 2)
        self.decode_workers = env_override_int("OPT_MOE_DECODE_WORKERS", 2)
        self._dtype_bytes = dtype_bytes(self.config.dtype_obj)
        total_tokens = self.config.context_window + self.config.decode_tokens
        self._workload_metadata = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(self.config.tokens_per_iteration),
        )
        self._history: Dict[str, List[float]] = {
            "ttft": [],
            "tpot": [],
            "throughput": [],
            "spec_accept": [],
            "nvlink": [],
            "nvlink_measured": [],
            "prefill_share": [],
            "paged_hit": [],
            "page_faults": [],
            "memory_gb": [],
        }
        self._iteration = 0
        self._mem_logger: Optional[GpuMemoryLogger] = None
        self._mem_log_path: Optional[Path] = None
        self._nvlink_warned: bool = False
        self._nvlink_status: str = "unknown"

    def _build_config(self) -> MoeInferenceConfig:
        return MoeInferenceConfig(
            vocab_size=env_override_int("OPT_MOE_VOCAB", 16384),
            hidden_size=env_override_int("OPT_MOE_HIDDEN", 1024),
            ffn_size=env_override_int("OPT_MOE_FFN", 4096),
            num_layers=env_override_int("OPT_MOE_LAYERS", 8),
            num_moe_layers=env_override_int("OPT_MOE_MOE_LAYERS", 4),
            num_experts=env_override_int("OPT_MOE_EXPERTS", 32),
            top_k=2,
            moe_layer_frequency=max(1, env_override_int("OPT_MOE_MOE_FREQ", 2)),
            batch_size=env_override_int("OPT_MOE_BATCH", 1),
            context_window=env_override_int("OPT_MOE_CONTEXT", 512),
            decode_tokens=env_override_int("OPT_MOE_DECODE", 32),
            router_noise=env_override_float("OPT_MOE_ROUTER_NOISE", 0.05),
            dtype=torch.bfloat16,
        )

    # --------------------------------------------------------------------- setup
    def setup(self) -> None:
        torch.manual_seed(21)
        cfg = self.config
        self.model = SimpleMoEGPT(cfg, device=self.device).eval()

        draft_cfg = MoeInferenceConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=max(512, cfg.hidden_size // 2),
            ffn_size=max(1024, cfg.ffn_size // 2),
            num_layers=max(2, cfg.num_layers // 2),
            num_moe_layers=max(1, cfg.num_moe_layers // 2),
            num_experts=max(4, cfg.num_experts // 2),
            top_k=1,
            moe_layer_frequency=cfg.moe_layer_frequency,
            batch_size=cfg.batch_size,
            context_window=cfg.context_window,
            decode_tokens=cfg.decode_tokens,
            router_noise=cfg.router_noise,
            dtype=cfg.dtype_obj,
        )
        self.draft_model = SimpleMoEGPT(draft_cfg, device=self.device).eval()

        self.prompts = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch_size, cfg.context_window),
            device=self.device,
        )
        self.paged_cache = PagedKVCache(
            batch_size=cfg.batch_size,
            max_tokens=cfg.context_window + cfg.decode_tokens,
            hidden=cfg.hidden_size,
            dtype=cfg.dtype_obj,
            device=self.device,
            page_size=env_override_int("OPT_MOE_PAGE_SIZE", 512),
        )
        self.spec_decoder = SpeculativeDecoder(
            target_model=self.model,
            draft_model=self.draft_model,
            chunk_size=env_override_int("OPT_MOE_SPEC_CHUNK", 4),
        )
        self._refresh_router_metrics()
        torch.cuda.synchronize(self.device)
        if torch.cuda.is_available() and hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats(self.device)
            log_path = resolve_gpu_log_path(None)
            logger = GpuMemoryLogger(
                device=self.device,
                interval=resolve_gpu_log_interval(1.0),
                log_path=log_path,
            )
            if logger.start():
                self._mem_logger = logger
                self._mem_log_path = log_path

    def _refresh_router_metrics(self) -> None:
        timestamp = time.time()
        for idx in range(self.prefill_workers):
            metrics = WorkerMetrics(
                queue_length=max(1, self.config.batch_size // max(1, self.prefill_workers)),
                gpu_utilization=45.0 + idx * 3.0,
                memory_usage=48.0 + idx * 2.5,
                kv_cache_usage=50.0 + idx * 4.0,
                active_requests=max(1, self.config.batch_size // max(1, self.prefill_workers)),
                last_updated=timestamp,
            )
            self.router.update_worker_metrics("prefill", f"prefill-{idx}", metrics)

        for idx in range(self.decode_workers):
            metrics = WorkerMetrics(
                queue_length=max(1, self.config.batch_size // max(1, self.decode_workers)),
                gpu_utilization=55.0 + idx * 4.0,
                memory_usage=58.0 + idx * 3.0,
                kv_cache_usage=62.0 + idx * 3.5,
                active_requests=max(1, self.config.batch_size // max(1, self.decode_workers)),
                last_updated=timestamp,
            )
            self.router.update_worker_metrics("decode", f"decode-{idx}", metrics)

    # --------------------------------------------------------------- benchmark_fn
    def benchmark_fn(self) -> Dict[str, List[float]]:
        if any(obj is None for obj in (self.model, self.prompts, self.paged_cache, self.spec_decoder)):
            raise RuntimeError("Benchmark not initialized")

        cfg = self.config
        paged_cache = self.paged_cache  # type: ignore[assignment]
        spec = self.spec_decoder  # type: ignore[assignment]
        paged_cache.reset()
        spec.reset()

        if torch.cuda.is_available() and hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats(self.device)
        logical_index = self.device.index if self.device.index is not None else None
        telemetry_before = query_gpu_telemetry(logical_index)

        router_assignments = {"prefill": 0, "decode": 0}
        prompt_stub = [0] * cfg.context_window
        prefix_cache = torch.randint(0, max(1, cfg.context_window // 4), (cfg.batch_size,))
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            for idx in range(cfg.batch_size):
                req = Request(
                    id=f"req-{self._iteration}-{idx}",
                    prompt_tokens=prompt_stub,
                    priority=Priority.STANDARD,
                    timestamp=time.time(),
                    prefix_cached_length=int(prefix_cache[idx].item()),
                    expected_output_length=cfg.decode_tokens,
                )
                stage, _ = self.router.route_request(req)
                if stage == "prefill":
                    router_assignments["prefill"] += 1
                else:
                    router_assignments["decode"] += 1

        ttft_times: List[float] = []
        tpot_times: List[float] = []

        with torch.no_grad():
            with self._nvtx_range("prefill_dualpipe"):
                prefill_start = self._record_start()
                hidden, logits = self.model.prefill(self.prompts, kv_cache=paged_cache.buffer, cache_start=0)
                torch.cuda.synchronize(self.device)
                ttft_ms = self._record_stop(prefill_start)
                ttft_times.append(ttft_ms)
                paged_cache.mark_prefill(cfg.context_window)

            next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            with self._nvtx_range("speculative_decode"):
                _, decode_times = spec.decode(
                    next_tokens,
                    cfg.decode_tokens,
                    paged_cache,
                    base_position=cfg.context_window,
                )
                tpot_times.extend(decode_times)

        telemetry_after = query_gpu_telemetry(logical_index)

        total_time_s = (sum(ttft_times) + sum(tpot_times)) / 1000.0
        throughput = cfg.tokens_per_iteration / max(total_time_s, 1e-6)
        prefill_bytes = cfg.batch_size * cfg.context_window * cfg.hidden_size * self._dtype_bytes
        nvlink_gbps = 0.0
        if ttft_times[0] > 0:
            nvlink_gbps = (prefill_bytes * 8.0 / 1e9) / (ttft_times[0] / 1000.0)
        measured_nvlink = self._compute_nvlink_delta(telemetry_before, telemetry_after, total_time_s)
        self._nvlink_status = telemetry_after.get("nvlink_status", "unknown")

        self._history["ttft"].extend(ttft_times)
        self._history["tpot"].extend(tpot_times)
        self._history["throughput"].append(throughput)
        self._history["spec_accept"].append(spec.acceptance_rate())
        self._history["nvlink"].append(nvlink_gbps)
        if measured_nvlink is not None:
            self._history["nvlink_measured"].append(measured_nvlink)
        else:
            if not self._nvlink_warned:
                self._nvlink_warned = True
        self._history["prefill_share"].append(router_assignments["prefill"] / max(1, cfg.batch_size))
        self._history["paged_hit"].append(paged_cache.occupancy_ratio)
        self._history["page_faults"].append(float(paged_cache.page_faults))
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated(self.device)  # type: ignore[arg-type]
            if peak_bytes:
                self._history["memory_gb"].append(peak_bytes / (1024 ** 3))

        self._iteration += 1
        self._refresh_router_metrics()
        return {
            "ttft_times_ms": ttft_times,
            "tpot_times_ms": tpot_times,
        }

    def _compute_nvlink_delta(
        self,
        telemetry_before: Dict[str, Optional[float]],
        telemetry_after: Dict[str, Optional[float]],
        elapsed_s: float,
    ) -> Optional[float]:
        if elapsed_s <= 0:
            return None
        tx_before = telemetry_before.get("nvlink_tx_bytes_total") if telemetry_before else None
        tx_after = telemetry_after.get("nvlink_tx_bytes_total") if telemetry_after else None
        rx_before = telemetry_before.get("nvlink_rx_bytes_total") if telemetry_before else None
        rx_after = telemetry_after.get("nvlink_rx_bytes_total") if telemetry_after else None
        if None in (tx_before, tx_after, rx_before, rx_after):
            return None
        delta_tx = max(0.0, tx_after - tx_before)
        delta_rx = max(0.0, rx_after - rx_before)
        total_delta = delta_tx + delta_rx
        if total_delta <= 0.0:
            return None
        return (total_delta * 8.0) / (elapsed_s * 1e9)

    # ------------------------------------------------------------------ lifecycle
    def teardown(self) -> None:
        self.model = None
        self.draft_model = None
        self.prompts = None
        self.paged_cache = None
        self.spec_decoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._mem_logger is not None:
            self._mem_logger.stop()
            self._mem_logger = None

    # ------------------------------------------------------------------- configs
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=6, warmup=1)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload_metadata

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self._history["throughput"]:
            return None
        metrics = {
            "optimized_moe.throughput_tok_s": float(statistics.mean(self._history["throughput"])),
            "optimized_moe.ttft_mean_ms": float(statistics.mean(self._history["ttft"])),
            "optimized_moe.tpot_mean_ms": float(statistics.mean(self._history["tpot"])),
            "optimized_moe.spec_accept_rate": float(statistics.mean(self._history["spec_accept"])),
            "optimized_moe.nvlink_reported_gbps": float(statistics.mean(self._history["nvlink"])),
            "optimized_moe.prefill_share": float(statistics.mean(self._history["prefill_share"])),
            "optimized_moe.paged_cache_occupancy": float(statistics.mean(self._history["paged_hit"])),
            "optimized_moe.page_faults": float(statistics.mean(self._history["page_faults"])),
        }
        if self._history["nvlink_measured"]:
            metrics["optimized_moe.nvlink_measured_gbps"] = float(
                statistics.mean(self._history["nvlink_measured"])
            )
        else:
            code = {
                "ok": 0.0,
                "nvlink_counters_missing": 1.0,
                "nvlink_disabled": 2.0,
                "nvml_unavailable": 3.0,
            }.get(self._nvlink_status, 4.0)
            metrics["optimized_moe.nvlink_status_code"] = code
        if self._history["memory_gb"]:
            metrics["optimized_moe.peak_memory_gb"] = float(statistics.mean(self._history["memory_gb"]))
        if self.paged_cache is not None:
            metrics["optimized_moe.kv_cache_gb"] = self.paged_cache.memory_gb
        return metrics

    def validate_result(self) -> Optional[str]:
        if not self._history["ttft"]:
            return "No TTFT samples captured"
        if not self._history["tpot"]:
            return "No decode tokens captured"
        return None


def get_benchmark() -> BaseBenchmark:
    return VLLMMoEInferenceBenchmark()


def _run_harness(iterations: Optional[int], warmup: Optional[int]) -> None:
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    config = benchmark.get_config()
    if iterations is not None:
        config.iterations = iterations
    if warmup is not None:
        config.warmup = warmup
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"Optimized MoE inference mean latency: {mean_ms:.3f} ms")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="vLLM-style MoE inference benchmark harness.")
    parser.add_argument("--iterations", type=int, help="Override benchmark iterations")
    parser.add_argument("--warmup", type=int, help="Override warmup iterations")
    args = parser.parse_args(argv)
    _run_harness(args.iterations, args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
