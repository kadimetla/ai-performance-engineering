"""Harness-level negative tests for anti-cheat protections."""

import sys
from pathlib import Path
from typing import Optional

import pytest
import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkConfig, BaseBenchmark

# Skip if CUDA is unavailable (harness requires CUDA devices)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required - NVIDIA GPU and tools must be available",
)


class _CheatingSetupBenchmark(BaseBenchmark):
    """Benchmark that pre-computes output in setup (should be caught)."""

    def __init__(self):
        super().__init__()
        self.output = None

    def setup(self) -> None:
        self.output = torch.randn(8, device=self.device)

    def benchmark_fn(self) -> None:
        # No-op timed body
        return None

    def get_verify_output(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("output missing")
        return self.output
    
    def get_verify_inputs(self):
        if self.output is None:
            raise RuntimeError("output missing")
        return {"x": self.output}

    def get_input_signature(self):
        return {"shapes": {"x": (1, 8)}, "dtypes": {"x": "float32"}, "batch_size": 1, "parameter_count": 0, "precision_flags": {}}

    def get_output_tolerance(self):
        return (1e-3, 1e-3)


class _GraphCheatBenchmark(BaseBenchmark):
    """Benchmark that will be flagged by forced graph cheat detector."""

    def __init__(self):
        super().__init__()
        self.x = None

    def setup(self) -> None:
        self.x = torch.randn(4, device=self.device)

    def benchmark_fn(self) -> torch.Tensor:
        return self.x * 2

    def get_verify_output(self) -> torch.Tensor:
        return self.x
    
    def get_verify_inputs(self):
        if self.x is None:
            raise RuntimeError("input missing")
        return {"x": self.x}

    def get_input_signature(self):
        return {"shapes": {"x": (4,)}, "dtypes": {"x": "float32"}, "batch_size": 1, "parameter_count": 0, "precision_flags": {}}

    def get_output_tolerance(self):
        return (1e-3, 1e-3)


class _StreamCheatBenchmark(BaseBenchmark):
    """Benchmark used to test stream audit failures."""

    def __init__(self):
        super().__init__()
        self.x = None

    def setup(self) -> None:
        self.x = torch.randn(4, device=self.device)

    def benchmark_fn(self) -> torch.Tensor:
        return self.x + 1

    def get_verify_output(self) -> torch.Tensor:
        return self.x
    
    def get_verify_inputs(self):
        if self.x is None:
            raise RuntimeError("input missing")
        return {"x": self.x}

    def get_input_signature(self):
        return {"shapes": {"x": (4,)}, "dtypes": {"x": "float32"}, "batch_size": 1, "parameter_count": 0, "precision_flags": {}}

    def get_output_tolerance(self):
        return (1e-3, 1e-3)


class _DummyAuditor:
    def check_issues(self):
        return False, ["auditor failure"]


class _DummyAuditContext:
    """Helper context to mimic audit_streams returning an auditor."""

    def __enter__(self):
        return _DummyAuditor()

    def __exit__(self, exc_type, exc, tb):
        return False


def _run_harness(benchmark: BaseBenchmark, config: BenchmarkConfig):
    """Helper to run harness in thread mode and return result."""
    harness = BenchmarkHarness()
    result = harness._benchmark_with_threading(benchmark, config)
    return result


def test_setup_precompute_failures():
    """Harness should block benchmarks that compute outputs during setup."""
    bench = _CheatingSetupBenchmark()
    config = BenchmarkConfig(iterations=1, warmup=0)
    result = _run_harness(bench, config)
    assert result.timeout_stage == "setup"
    assert any("pre-computation" in err.lower() for err in result.errors)


def test_graph_cheat_detection_forces_failure(monkeypatch):
    """Harness should raise when graph cheat detector flags work during capture."""
    bench = _GraphCheatBenchmark()
    config = BenchmarkConfig(iterations=1, warmup=0, enable_cuda_graph=True)

    # Force detector to report a cheat regardless of timings
    monkeypatch.setattr(
        "core.harness.benchmark_harness.GraphCaptureCheatDetector.check_for_cheat",
        lambda self, **kwargs: (True, "forced cheat"),
    )

    result = _run_harness(bench, config)
    assert any("graph capture cheat" in err.lower() for err in result.errors)


def test_stream_audit_failure(monkeypatch):
    """Harness should fail when stream audit or sync detects issues."""
    bench = _StreamCheatBenchmark()
    config = BenchmarkConfig(iterations=1, warmup=0)

    # Force stream sync check to fail
    monkeypatch.setattr(
        "core.harness.benchmark_harness.check_stream_sync_completeness",
        lambda pre, post: (False, "unsynced stream"),
    )

    monkeypatch.setattr(
        "core.harness.benchmark_harness.audit_streams",
        lambda device=None: _DummyAuditContext(),
    )

    result = _run_harness(bench, config)
    assert any("stream timing violation" in err.lower() for err in result.errors)


class _DummyAuditContext:
    """Helper context to mimic audit_streams returning an auditor."""

    def __enter__(self):
        return _DummyAuditor()

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyAuditor:
    def check_issues(self):
        return False, ["auditor failure"]
