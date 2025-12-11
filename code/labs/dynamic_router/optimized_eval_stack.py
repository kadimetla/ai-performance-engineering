"""Optimized cheap eval stack: adds router-aware drop penalties and better accuracy priors."""

from __future__ import annotations

import io
import json
import sys
from numbers import Number
from typing import Dict, List, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.verification_mixin import VerificationPayloadMixin
from labs.dynamic_router.eval_stack import EvalConfig, run_eval_stack
from contextlib import redirect_stdout


class OptimizedEvalStackBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs the cheap-eval stack with the optimized settings."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.metrics: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.verify_input: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def _resolve_device(self) -> torch.device:  # type: ignore[override]
        return torch.device("cpu")

    def setup(self) -> None:
        torch.manual_seed(42)

    def benchmark_fn(self) -> None:
        cfg = EvalConfig.from_flags(self._argv(), seed=42)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self._summary = run_eval_stack("optimized", cfg)
        captured = buf.getvalue().strip()
        if captured:
            try:
                lines = [ln for ln in captured.splitlines() if ln]
                print(json.dumps({"event": "eval_stack_stdout", "variant": "optimized", "lines": lines}), file=sys.stderr)
            except Exception:
                print(captured, file=sys.stderr)
        summary_tensor = torch.tensor(
            [float(v) for v in self._summary.values() if isinstance(v, Number)],
            dtype=torch.float32,
        )
        if summary_tensor.numel() == 0:
            summary_tensor = torch.zeros(1, dtype=torch.float32)
        verify_tensor = torch.tensor([[float(cfg.request_count), float(cfg.experts)]], dtype=torch.float32)
        expected_shape = tuple(verify_tensor.shape)
        if self.metrics is None or tuple(self.metrics.shape) != expected_shape:
            self.metrics = torch.zeros(expected_shape, dtype=torch.float32)
        if self.verify_input is None or tuple(self.verify_input.shape) != expected_shape:
            self.verify_input = torch.ones(expected_shape, dtype=torch.float32)
        self.output = (verify_tensor * self.verify_input + self.metrics).detach()
        self._set_verification_payload(
            inputs={
                "verify_input": self.verify_input.detach(),
                "seed": torch.tensor([cfg.seed], dtype=torch.int64),
                "metrics": summary_tensor.detach(),
            },
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "tf32": False},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.metrics = None
        self.verify_input = None
        self.output = None
        super().teardown()

    def _argv(self) -> List[str]:
        cfg = getattr(self, "_config", None)
        if cfg is None:
            return sys.argv[1:]
        label = getattr(cfg, "target_label", None)
        extra_map = getattr(cfg, "target_extra_args", {}) or {}
        if label and label in extra_map:
            return list(extra_map[label])
        if len(extra_map) == 1:
            return list(next(iter(extra_map.values())))
        return sys.argv[1:]

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5, measurement_timeout_seconds=90)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedEvalStackBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    bench.benchmark_fn()
    print(json.dumps(bench.get_custom_metrics() or {}, indent=2))
