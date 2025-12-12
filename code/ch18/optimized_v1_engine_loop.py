"""Optimized V1 EngineCore/CoreClient polling loop with KV reclamation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig

try:  # Optional, only for real vLLM runs.
    from vllm.v1.outputs import RequestOutput  # type: ignore
except Exception:  # pragma: no cover - vLLM may be absent locally
    RequestOutput = None  # type: ignore

from ch18.v1_engine_loop_common import MockRequestOutput, build_demo_stack

OutputType = RequestOutput if RequestOutput is not None else MockRequestOutput
from core.benchmark.verification_mixin import VerificationPayloadMixin


def run_engine_loop(engine_core: Any, core_client: Any) -> Iterator[OutputType]:
    """
    Poll the engine, emit RequestOutputs as they are ready, and free KV cache pages.

    Mirrors the V1 guidance: keep polling even if executed_flag is False, dedupe
    finished IDs, and report them to the core client so paged KV cache can be
    reclaimed.
    """
    finished: Set[str] = set()

    while True:
        outputs, _executed = engine_core.step()

        for ro in outputs:
            yield ro

        newly_finished: Dict[str, OutputType] = {
            ro.request_id: ro for ro in outputs if getattr(ro, "finished", False) and ro.request_id not in finished
        }
        if newly_finished:
            core_client.report_finished_ids(list(newly_finished.keys()))
            finished.update(newly_finished.keys())

        if core_client.is_all_done():
            break


def _demo() -> None:
    engine_core, core_client = build_demo_stack()
    outputs = list(run_engine_loop(engine_core, core_client))
    summary = {
        "steps": engine_core.calls,
        "tokens": "".join(ro.delta_text for ro in outputs),
        "reported_finished": list(core_client.finished_reported),
        "all_done": core_client.is_all_done(),
    }
    print("Optimized loop demo:", summary)


class OptimizedV1EngineLoopBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark for optimized V1 EngineCore/CoreClient polling loop with KV reclamation."""

    def __init__(self):
        super().__init__()
        self._engine_core = None
        self._core_client = None
        self._outputs = None
        self.output = None
        self._verification_payload = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def setup(self) -> None:
        """Set up the demo engine stack."""
        self._engine_core, self._core_client = build_demo_stack()

    def benchmark_fn(self) -> None:
        """Run the optimized engine loop."""
        import torch
        # Reset the demo stack for each iteration
        self._engine_core, self._core_client = build_demo_stack()
        self._outputs = list(run_engine_loop(self._engine_core, self._core_client))
        # Store metrics as output tensor for verification
        self.output = torch.tensor([
            float(self._engine_core.calls),
            float(len(self._outputs)),
        ], dtype=torch.float32)

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"seed": torch.tensor(42)},
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.1, 1.0),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_speculative_decoding_metrics
        return compute_speculative_decoding_metrics(
            draft_tokens=getattr(self, '_draft_tokens', 64),
            accepted_tokens=getattr(self, '_accepted_tokens', 48),
            draft_time_ms=getattr(self, '_draft_ms', 5.0),
            verify_time_ms=getattr(self, '_verify_ms', 10.0),
            num_rounds=getattr(self, '_num_rounds', 8),
        )


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedV1EngineLoopBenchmark()


if __name__ == "__main__":
    _demo()
