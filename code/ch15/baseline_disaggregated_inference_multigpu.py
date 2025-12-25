"""Baseline disaggregated inference benchmark (multi-GPU torchrun pipeline).

Chapter 15: Disaggregated Inference

This benchmark models a disaggregated prefill/decode pipeline across multiple GPUs.
Baseline behavior is serialized: prefill completes for all requests before decode starts.
The optimized pair overlaps prefill and decode via pipelined transfers.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch15.verification_payload_mixin import VerificationPayloadMixin  # noqa: E402
from core.benchmark.verification import PrecisionFlags  # noqa: E402
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.optimization.moe_inference import (  # noqa: E402
    MoeInferenceConfig,
    SimpleMoEGPT,
    allocate_kv_cache,
)

_DEFAULT_WORLD_SIZE = int(os.getenv("AISP_DISAGG_WORLD_SIZE", "4"))


@dataclass(frozen=True)
class DisaggConfig:
    vocab_size: int = 16384
    hidden_size: int = 1024
    ffn_size: int = 4096
    num_layers: int = 8
    num_moe_layers: int = 4
    num_experts: int = 16
    top_k: int = 2
    batch_size: int = 1
    requests_per_rank: int = 4
    context_window: int = 256
    decode_tokens: int = 256
    dtype: torch.dtype = torch.bfloat16

    @property
    def tokens_per_request(self) -> int:
        return self.context_window + self.decode_tokens


@dataclass
class _LocalPair:
    prefill_device: torch.device
    decode_device: torch.device
    prefill_model: SimpleMoEGPT
    decode_model: SimpleMoEGPT
    prompts: torch.Tensor


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


def _init_distributed() -> Tuple[int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for multi-GPU disaggregation")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def _run_prefill(
    cfg: DisaggConfig,
    model: SimpleMoEGPT,
    prompts: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    kv_chunks: List[torch.Tensor] = []
    seed_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for req_idx in range(cfg.requests_per_rank):
            request_prompt = prompts[req_idx : req_idx + 1]
            hidden, logits = model.prefill(request_prompt)
            seed_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            kv_chunks.append(hidden)
            seed_chunks.append(seed_tokens)
    return kv_chunks, seed_chunks


def _run_decode(
    cfg: DisaggConfig,
    model: SimpleMoEGPT,
    kv_chunks: List[torch.Tensor],
    seed_chunks: List[torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for kv_prompt, seed_tokens in zip(kv_chunks, seed_chunks):
            kv_cache = allocate_kv_cache(
                cfg.batch_size,
                cfg.tokens_per_request,
                cfg.hidden_size,
                cfg.dtype,
                device,
            )
            kv_cache[:, : cfg.context_window] = kv_prompt
            tokens = seed_tokens
            for step in range(cfg.decode_tokens):
                _, decode_logits = model.decode(
                    tokens,
                    kv_cache=kv_cache,
                    position=cfg.context_window + step,
                )
                tokens = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
            outputs.append(tokens)
    return outputs


def _run_torchrun_worker(
    cfg: DisaggConfig,
    *,
    overlap: bool,
    label: str,
    iters: int,
    warmup: int,
) -> None:
    rank, world_size, device = _init_distributed()
    expected_world_size = _DEFAULT_WORLD_SIZE
    if world_size != expected_world_size:
        raise RuntimeError(
            f"Expected world_size={expected_world_size} (set AISP_DISAGG_WORLD_SIZE), got {world_size}."
        )
    if world_size % 2 != 0:
        raise RuntimeError("world_size must be even (prefill ranks + decode ranks)")

    num_pairs = world_size // 2
    is_prefill = rank < num_pairs
    pair_id = rank if is_prefill else rank - num_pairs
    peer_rank = pair_id + num_pairs if is_prefill else pair_id

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    moe_cfg = _build_moe_config(cfg)
    model = SimpleMoEGPT(moe_cfg, device=device).eval()

    prompts: Optional[torch.Tensor] = None
    if is_prefill:
        prompts = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.requests_per_rank, cfg.context_window),
            device=device,
            dtype=torch.long,
        )

    def run_iteration() -> List[torch.Tensor]:
        if is_prefill:
            if overlap:
                handles = []
                with torch.no_grad():
                    for req_idx in range(cfg.requests_per_rank):
                        request_prompt = prompts[req_idx : req_idx + 1]
                        hidden, logits = model.prefill(request_prompt)
                        seed_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        handles.append(dist.isend(hidden, dst=peer_rank))
                        handles.append(dist.isend(seed_tokens, dst=peer_rank))
                for handle in handles:
                    handle.wait()
            else:
                kv_chunks, seed_chunks = _run_prefill(cfg, model, prompts)
                for kv_prompt, seed_tokens in zip(kv_chunks, seed_chunks):
                    dist.send(kv_prompt, dst=peer_rank)
                    dist.send(seed_tokens, dst=peer_rank)
            return []

        kv_chunks = []
        seed_chunks = []
        if overlap:
            for _ in range(cfg.requests_per_rank):
                kv_buf = torch.empty(
                    (cfg.batch_size, cfg.context_window, cfg.hidden_size),
                    device=device,
                    dtype=cfg.dtype,
                )
                seed_buf = torch.empty(
                    (cfg.batch_size, 1),
                    device=device,
                    dtype=torch.long,
                )
                kv_req = dist.irecv(kv_buf, src=peer_rank)
                seed_req = dist.irecv(seed_buf, src=peer_rank)
                kv_req.wait()
                seed_req.wait()
                decoded = _run_decode(cfg, model, [kv_buf], [seed_buf], device)
                kv_chunks.extend(decoded)
        else:
            for _ in range(cfg.requests_per_rank):
                kv_buf = torch.empty(
                    (cfg.batch_size, cfg.context_window, cfg.hidden_size),
                    device=device,
                    dtype=cfg.dtype,
                )
                seed_buf = torch.empty(
                    (cfg.batch_size, 1),
                    device=device,
                    dtype=torch.long,
                )
                dist.recv(kv_buf, src=peer_rank)
                dist.recv(seed_buf, src=peer_rank)
                kv_chunks.append(kv_buf)
                seed_chunks.append(seed_buf)
            kv_chunks = _run_decode(cfg, model, kv_chunks, seed_chunks, device)
        return kv_chunks

    dist.barrier()
    torch.cuda.synchronize(device)

    for _ in range(max(warmup, 0)):
        run_iteration()
    torch.cuda.synchronize(device)
    dist.barrier()

    start = time.perf_counter()
    for _ in range(max(iters, 1)):
        run_iteration()
    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = time.perf_counter() - start

    if rank == 0:
        total_requests = cfg.requests_per_rank * num_pairs * cfg.batch_size
        tokens_per_iter = total_requests * cfg.tokens_per_request
        tokens_per_s = tokens_per_iter * (max(iters, 1) / max(elapsed, 1e-9))
        time_per_iter_ms = (elapsed / max(iters, 1)) * 1000.0
        print(f"rank0 {label} tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 {label} time_per_iter_ms: {time_per_iter_ms:.3f}")

    dist.destroy_process_group()


class _DisaggregatedInferenceMultiGPUBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Shared multi-GPU disaggregated inference harness."""

    def __init__(self, *, overlap: bool, label: str) -> None:
        super().__init__()
        self.cfg = DisaggConfig()
        self.world_size = _DEFAULT_WORLD_SIZE
        if self.world_size % 2 != 0:
            raise RuntimeError("world_size must be even for disaggregated inference")
        self.num_pairs = self.world_size // 2
        self.overlap = bool(overlap)
        self.label = label
        self._pairs: List[_LocalPair] = []
        self._output: Optional[torch.Tensor] = None
        self._verify_prompt: Optional[torch.Tensor] = None
        self._param_count: int = 0

        total_requests = self.cfg.requests_per_rank * self.num_pairs * self.cfg.batch_size
        tokens_per_iter = total_requests * self.cfg.tokens_per_request
        self.register_workload_metadata(
            requests_per_iteration=float(total_requests),
            tokens_per_iteration=float(tokens_per_iter),
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for multi-GPU disaggregation")
        if torch.cuda.device_count() < self.world_size:
            raise RuntimeError(
                f"SKIPPED: requires >= {self.world_size} GPUs (found {torch.cuda.device_count()})"
            )

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        moe_cfg = _build_moe_config(self.cfg)
        self._pairs = []
        total_params = 0
        for pair_id in range(self.num_pairs):
            prefill_device = torch.device(f"cuda:{pair_id}")
            decode_device = torch.device(f"cuda:{pair_id + self.num_pairs}")
            prefill_model = SimpleMoEGPT(moe_cfg, device=prefill_device).eval()
            decode_model = SimpleMoEGPT(moe_cfg, device=decode_device).eval()
            decode_model.load_state_dict(prefill_model.state_dict())
            prompts = torch.randint(
                0,
                self.cfg.vocab_size,
                (self.cfg.requests_per_rank, self.cfg.context_window),
                device=prefill_device,
                dtype=torch.long,
            )
            total_params += sum(p.numel() for p in prefill_model.parameters())
            total_params += sum(p.numel() for p in decode_model.parameters())
            self._pairs.append(
                _LocalPair(
                    prefill_device=prefill_device,
                    decode_device=decode_device,
                    prefill_model=prefill_model,
                    decode_model=decode_model,
                    prompts=prompts,
                )
            )

        self._param_count = total_params
        if not self._pairs:
            raise RuntimeError("Failed to initialize prompts for verification")
        self._verify_prompt = self._pairs[0].prompts
        for pair in self._pairs:
            torch.cuda.synchronize(pair.prefill_device)
            torch.cuda.synchronize(pair.decode_device)

    def benchmark_fn(self) -> None:
        if not self._pairs:
            raise RuntimeError("setup() must run before benchmark_fn()")

        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for pair in self._pairs:
                kv_chunks, seed_chunks = _run_prefill(self.cfg, pair.prefill_model, pair.prompts)
                decoded = _run_decode(
                    self.cfg,
                    pair.decode_model,
                    [kv.to(pair.decode_device, non_blocking=self.overlap) for kv in kv_chunks],
                    [seed.to(pair.decode_device, non_blocking=self.overlap) for seed in seed_chunks],
                    pair.decode_device,
                )
                outputs.extend(decoded)

        self._output = torch.cat([out.detach().cpu() for out in outputs], dim=0)

    def capture_verification_payload(self) -> None:
        if self._output is None or self._verify_prompt is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
        meta_dtype = torch.float32
        self._set_verification_payload(
            inputs={
                "prompt": self._verify_prompt,
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
                "world_size": self.world_size,
                "pipeline_stages": 2,
                "per_rank_batch_size": self.cfg.requests_per_rank,
                "collective_type": "send_recv",
            },
        )

    def validate_result(self) -> Optional[str]:
        if self._output is None:
            return "No output captured"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=self.world_size,
            iterations=4,
            warmup=3,
            multi_gpu_required=True,
            measurement_timeout_seconds=900,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=[],
            env={
                "AISP_DISAGG_WORLD_SIZE": str(self.world_size),
                "NCCL_DEBUG": "WARN",
                "NCCL_P2P_LEVEL": "NVL",
                "NCCL_P2P_DISABLE": "0",
            },
            parse_rank0_only=True,
            multi_gpu_required=True,
            name=self.label,
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )



class BaselineDisaggregatedInferenceMultiGPUBenchmark(_DisaggregatedInferenceMultiGPUBenchmark):
    """Serialized prefill then decode across multi-GPU ranks."""

    def __init__(self) -> None:
        super().__init__(overlap=False, label="baseline_disaggregated_inference_multigpu")


def get_benchmark() -> BaseBenchmark:
    return BaselineDisaggregatedInferenceMultiGPUBenchmark()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _run_torchrun_worker(
        DisaggConfig(),
        overlap=False,
        label="baseline_disaggregated_inference_multigpu",
        iters=int(args.iters),
        warmup=int(args.warmup),
    )


if __name__ == "__main__":
    main()
