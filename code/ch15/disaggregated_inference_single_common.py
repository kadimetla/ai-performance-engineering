"""Shared single-GPU disaggregated inference benchmark logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from core.benchmark.verification import PrecisionFlags
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.optimization.moe_inference import (
    MoeInferenceConfig,
    SimpleMoEGPT,
    allocate_kv_cache,
)
from ch15.verification_payload_mixin import VerificationPayloadMixin


@dataclass(frozen=True)
class DisaggConfig:
    vocab_size: int = 16384
    hidden_size: int = 768
    ffn_size: int = 3072
    num_layers: int = 4
    num_moe_layers: int = 2
    num_experts: int = 16
    top_k: int = 2
    batch_size: int = 1
    requests_per_rank: int = 24
    context_window: int = 1024
    decode_tokens: int = 32
    dtype: torch.dtype = torch.bfloat16

    @property
    def tokens_per_request(self) -> int:
        return self.context_window + self.decode_tokens


def _build_moe_config(cfg: DisaggConfig) -> MoeInferenceConfig:
    return MoeInferenceConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        ffn_size=cfg.ffn_size,
        num_layers=cfg.num_layers,
        num_moe_layers=cfg.num_moe_layers,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        moe_layer_frequency=1,
        batch_size=cfg.batch_size,
        context_window=cfg.context_window,
        decode_tokens=cfg.decode_tokens,
        router_noise=0.0,
        dtype=cfg.dtype,
    )


class DisaggregatedInferenceSingleGPUBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Single-GPU analogue of disaggregated prefill/decode inference."""

    def __init__(self, *, use_host_staging: bool, label: str, cfg: Optional[DisaggConfig] = None) -> None:
        super().__init__()
        self.use_host_staging = bool(use_host_staging)
        self.label = label
        self.cfg = cfg or DisaggConfig()
        tokens = self.cfg.requests_per_rank * self.cfg.tokens_per_request
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.cfg.requests_per_rank),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.cfg.requests_per_rank),
            tokens_per_iteration=float(tokens),
        )
        self.prefill_model: Optional[SimpleMoEGPT] = None
        self.decode_model: Optional[SimpleMoEGPT] = None
        self.prompts: Optional[torch.Tensor] = None
        self.kv_caches: List[torch.Tensor] = []
        self._output: Optional[torch.Tensor] = None
        self._param_count = 0

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for disaggregated inference")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = _build_moe_config(self.cfg)
        self.prefill_model = SimpleMoEGPT(config, device=self.device).eval()
        self.decode_model = SimpleMoEGPT(config, device=self.device).eval()
        self.prompts = torch.randint(
            0,
            self.cfg.vocab_size,
            (self.cfg.requests_per_rank, self.cfg.batch_size, self.cfg.context_window),
            device=self.device,
            dtype=torch.long,
        )
        if not self.use_host_staging:
            self.kv_caches = [
                allocate_kv_cache(
                    self.cfg.batch_size,
                    self.cfg.tokens_per_request,
                    self.cfg.hidden_size,
                    self.cfg.dtype,
                    self.device,
                )
                for _ in range(self.cfg.requests_per_rank)
            ]
        self._param_count = sum(p.numel() for p in self.prefill_model.parameters()) + sum(
            p.numel() for p in self.decode_model.parameters()
        )
        torch.cuda.synchronize(self.device)

    def _allocate_kv_cache(self) -> torch.Tensor:
        return allocate_kv_cache(
            self.cfg.batch_size,
            self.cfg.tokens_per_request,
            self.cfg.hidden_size,
            self.cfg.dtype,
            self.device,
        )

    def benchmark_fn(self) -> None:
        if self.prefill_model is None or self.decode_model is None or self.prompts is None:
            raise RuntimeError("setup() must run before benchmark_fn()")

        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for idx in range(self.cfg.requests_per_rank):
                prompt = self.prompts[idx]
                hidden, logits = self.prefill_model.prefill(prompt)
                seed_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                if self.use_host_staging:
                    kv_cpu = hidden.cpu()
                    kv_cache = self._allocate_kv_cache()
                    kv_cache[:, : self.cfg.context_window] = kv_cpu.to(self.device)
                else:
                    kv_cache = self.kv_caches[idx]
                    kv_cache[:, : self.cfg.context_window] = hidden

                tokens = seed_tokens
                for step in range(self.cfg.decode_tokens):
                    _, decode_logits = self.decode_model.decode(
                        tokens,
                        kv_cache=kv_cache,
                        position=self.cfg.context_window + step,
                    )
                    tokens = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
                outputs.append(tokens.squeeze(0))

        torch.cuda.synchronize(self.device)
        self._output = torch.cat([out.detach().cpu() for out in outputs], dim=0)

    def capture_verification_payload(self) -> None:
        if self._output is None or self.prompts is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
        meta_dtype = torch.float32
        self._set_verification_payload(
            inputs={
                "prompt": self.prompts[0].detach().cpu(),
                "decode_tokens": torch.zeros((self.cfg.decode_tokens,), dtype=meta_dtype),
                "hidden_size": torch.zeros((self.cfg.hidden_size,), dtype=meta_dtype),
                "num_layers": torch.zeros((self.cfg.num_layers,), dtype=meta_dtype),
                "num_experts": torch.zeros((self.cfg.num_experts,), dtype=meta_dtype),
            },
            output=self._output,
            batch_size=int(self._output.shape[0]),
            parameter_count=int(self._param_count),
            precision_flags=PrecisionFlags(bf16=True, tf32=tf32_enabled),
            output_tolerance=(0.0, 0.0),
            signature_overrides={
                "world_size": 1,
                "pipeline_stages": 1,
                "pipeline_stage_boundaries": [(0, 0)],
                "per_rank_batch_size": self.cfg.requests_per_rank,
                "collective_type": "local_copy",
            },
        )

    def teardown(self) -> None:
        self.prefill_model = None
        self.decode_model = None
        self.prompts = None
        self.kv_caches = []
        self._output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=3, warmup=5, measurement_timeout_seconds=900)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def attach_benchmark_metadata(bench: BaseBenchmark, module_file: str) -> BaseBenchmark:
    """Ensure subprocess runner calls get_benchmark() for parameterized benchmarks."""
    bench._module_file_override = module_file
    bench._factory_name_override = "get_benchmark"
    return bench
