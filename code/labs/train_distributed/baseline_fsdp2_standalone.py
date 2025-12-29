#!/usr/bin/env python3
"""Baseline: FSDP2 with BF16 mixed precision.

Demonstrates standard FSDP2 training without FP8 optimization.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

# Add common to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_BATCH_SIZE = 4
_DEFAULT_MICRO_BATCH = 1
_DEFAULT_SEQ_LEN = 2048
_DEFAULT_HIDDEN = 4096
_DEFAULT_LAYERS = 8


class SimpleTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear2(torch.relu(self.linear1(self.norm(x))))


def _init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("FSDP2 baseline requires torchrun (RANK/WORLD_SIZE missing).")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def _resolve_world_size() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for FSDP2 benchmarks")
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("FSDP2 baseline requires >=2 GPUs.")
    return world_size


def _build_model(hidden_size: int, num_layers: int, device: torch.device) -> nn.ModuleList:
    return nn.ModuleList([
        SimpleTransformerLayer(hidden_size)
        for _ in range(num_layers)
    ]).to(device)


def _run_worker(
    *,
    iters: int,
    warmup: int,
    batch_size: int,
    micro_batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
) -> None:
    rank, world_size, local_rank = _init_distributed()
    if world_size < 2:
        raise RuntimeError("FSDP2 baseline requires >=2 GPUs.")

    if batch_size % micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(f"cuda:{local_rank}")

    model = _build_model(hidden_size, num_layers, device)
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    fsdp_model = FSDP(model, mixed_precision=mixed_precision, use_orig_params=True)

    input_tensor = torch.randn(
        micro_batch_size,
        seq_length,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)
    grad_accum_steps = batch_size // micro_batch_size

    def _step() -> float:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            output = input_tensor
            for layer in fsdp_model:
                output = layer(output)
            loss = output.mean() / grad_accum_steps
            loss.backward()
        optimizer.step()
        return float(loss.detach().item() * grad_accum_steps)

    for _ in range(max(warmup, 0)):
        _step()
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    loss_value = 0.0
    for _ in range(max(iters, 1)):
        loss_value = _step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    tokens_per_iter = batch_size * seq_length
    tokens_per_s = tokens_per_iter * (max(iters, 1) / max(elapsed, 1e-9))

    if rank == 0:
        print(f"rank0 tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 time_per_iter_ms: {(elapsed / max(iters,1)) * 1000.0:.3f}")
        print(f"rank0 loss: {loss_value:.6f}")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline FSDP2 BF16 training")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--micro-batch-size", type=int, default=_DEFAULT_MICRO_BATCH)
    parser.add_argument("--seq-length", type=int, default=_DEFAULT_SEQ_LEN)
    parser.add_argument("--hidden-size", type=int, default=_DEFAULT_HIDDEN)
    parser.add_argument("--num-layers", type=int, default=_DEFAULT_LAYERS)
    args = parser.parse_args()

    _run_worker(
        iters=args.iters,
        warmup=args.warmup,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )


def run_benchmark(**_kwargs) -> Dict[str, float]:
    """Legacy entrypoint; use torchrun via the harness for multi-GPU runs."""
    raise RuntimeError("baseline_fsdp2_standalone must be executed via torchrun.")


class BaselineFSDP2Benchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness entry that launches this module via torchrun."""

    def __init__(self) -> None:
        super().__init__()
        self.register_workload_metadata(requests_per_iteration=float(_DEFAULT_BATCH_SIZE))
        self._model: Optional[nn.ModuleList] = None
        self._input: Optional[torch.Tensor] = None
        self._output: Optional[torch.Tensor] = None
        self._param_count = 0
        self._world_size = _resolve_world_size()

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._model = _build_model(_DEFAULT_HIDDEN, _DEFAULT_LAYERS, self.device)
        self._model = self._model.to(torch.bfloat16)
        self._input = torch.randn(
            _DEFAULT_BATCH_SIZE,
            _DEFAULT_SEQ_LEN,
            _DEFAULT_HIDDEN,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._param_count = sum(p.numel() for p in self._model.parameters())

    def benchmark_fn(self) -> None:
        if self._model is None or self._input is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        x = self._input
        for layer in self._model:
            x = layer(x)
        self._output = x

    def capture_verification_payload(self) -> None:
        if self._output is None or self._input is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self._input},
            output=self._output,
            batch_size=_DEFAULT_BATCH_SIZE,
            parameter_count=int(self._param_count),
            precision_flags=PrecisionFlags(bf16=True, tf32=False),
            output_tolerance=(0.1, 1.0),
            signature_overrides={"world_size": self._world_size},
        )

    def _prepare_verification_payload(self) -> None:
        if hasattr(self, "_subprocess_verify_output"):
            return
        self.setup()
        try:
            self.benchmark_fn()
            self.capture_verification_payload()
            self._subprocess_verify_output = self.get_verify_output()
            self._subprocess_output_tolerance = self.get_output_tolerance()
            self._subprocess_input_signature = self.get_input_signature()
        finally:
            self.teardown()

    def teardown(self) -> None:
        self._model = None
        self._input = None
        self._output = None
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self._output is None:
            return "No output captured"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=self._world_size,
            iterations=3,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=900,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        self._prepare_verification_payload()
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=[],
            multi_gpu_required=True,
            name="baseline_fsdp2_standalone",
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineFSDP2Benchmark()


if __name__ == "__main__":
    main()
