"""Integration regression test for ch20 optimized_multiple_all_techniques."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.filterwarnings("ignore:Attempting to run cuBLAS.*:UserWarning"),
]

repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.env import apply_env_defaults

apply_env_defaults()

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkConfig, BenchmarkMode
from ch20.optimized_autotuning import OptimizedAutotuningBenchmark
from ch20.optimized_end_to_end_bandwidth import OptimizedEndToEndBandwidthBenchmark
from ch20.optimized_moe import OptimizedMoeBenchmark
from ch20.optimized_multiple_all_techniques import OptimizedAllTechniquesBenchmark

pytestmark.append(
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required - benchmark needs a NVIDIA GPU"
    )
)


class ShortBenchmarkConfigMixin:
    """Override harness config to keep tests fast and subprocess-friendly."""

    def get_config(self):
        return BenchmarkConfig(
            iterations=4,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,
            use_subprocess=True,
        )


class ShortMultipleAllTechniquesBenchmark(ShortBenchmarkConfigMixin, OptimizedAllTechniquesBenchmark):
    """Benchmark variant with fewer iterations to keep tests fast."""


class ShortAutotuningBenchmark(ShortBenchmarkConfigMixin, OptimizedAutotuningBenchmark):
    """Autotuning benchmark with shortened config."""


class ShortEndToEndBandwidthBenchmark(ShortBenchmarkConfigMixin, OptimizedEndToEndBandwidthBenchmark):
    """End-to-end bandwidth benchmark with shortened config."""


class ShortMoeBenchmark(ShortBenchmarkConfigMixin, OptimizedMoeBenchmark):
    """MoE benchmark variant for integration tests."""


SHORT_BENCHMARKS = [
    ShortMultipleAllTechniquesBenchmark,
    ShortAutotuningBenchmark,
    ShortEndToEndBandwidthBenchmark,
    ShortMoeBenchmark,
]


def _require_triton_cfg():
    """Return TorchInductor Triton config or skip if unavailable."""
    inductor = getattr(torch, "_inductor", None)
    if inductor is None or not hasattr(inductor, "config"):
        pytest.skip("TorchInductor configuration unavailable")
    triton_cfg = getattr(inductor.config, "triton", None)
    if triton_cfg is None:
        pytest.skip("TorchInductor Triton configuration unavailable")
    for attr in ("cudagraphs", "cudagraph_trees"):
        if not hasattr(triton_cfg, attr):
            pytest.skip(f"Triton configuration missing {attr}")
    return triton_cfg


@pytest.fixture(scope="session", autouse=True)
def _warm_cuda_primary_context():
    """
    Ensure CUDA primary context is created before benchmarks run.
    
    Without this, the first cuBLAS call emits a warning about initializing the
    primary context. A one-time torch.cuda.init() plus a tiny CUDA op prevents
    the noise.
    """
    if torch.cuda.is_available():
        torch.cuda.init()
        try:
            # Any trivial CUDA work forces the context to materialize
            torch.empty(1, device="cuda").add_(1)
            # Prime cuBLAS handle to avoid first-use warning inside tests
            torch.ones((1, 1), device="cuda").matmul(torch.ones((1, 1), device="cuda"))
            torch.cuda.synchronize()
        except Exception:
            # If anything goes wrong, keep tests running; the runtime will still lazy-init later
            pass


@pytest.fixture()
def triton_cfg_guard():
    cfg = _require_triton_cfg()
    original = {attr: getattr(cfg, attr) for attr in ("cudagraphs", "cudagraph_trees")}
    try:
        yield cfg, original
    finally:
        for attr, value in original.items():
            setattr(cfg, attr, value)


@pytest.mark.parametrize("benchmark_cls", SHORT_BENCHMARKS)
@pytest.mark.parametrize("enable_cudagraph_features", [True, False])
def test_ch20_compiled_benchmarks_run_with_subprocess(triton_cfg_guard, benchmark_cls, enable_cudagraph_features):
    """Ensure subprocess execution succeeds regardless of cudagraph defaults."""
    triton_cfg, original_values = triton_cfg_guard
    for attr in original_values:
        setattr(triton_cfg, attr, enable_cudagraph_features)

    benchmark = benchmark_cls()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(
            iterations=1,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,
            adaptive_iterations=False,
        ),
    )

    try:
        result = harness.benchmark(benchmark)
    except RuntimeError as exc:
        raise

    assert result.errors == [], f"Benchmark reported errors: {result.errors}"
    assert result.timing is not None
    assert result.timing.iterations == 1


def test_inductor_config_restored_after_success(triton_cfg_guard):
    """Teardown should restore Inductor config even when toggles were enabled."""
    triton_cfg, _ = triton_cfg_guard
    for attr in ("cudagraphs", "cudagraph_trees"):
        setattr(triton_cfg, attr, True)

    benchmark = ShortMultipleAllTechniquesBenchmark()
    benchmark.setup()
    benchmark.teardown()

    for attr in ("cudagraphs", "cudagraph_trees"):
        assert getattr(triton_cfg, attr) is True


def test_inductor_config_restored_after_failure(triton_cfg_guard):
    """If setup raises, cudagraph toggles must still be restored."""
    triton_cfg, original_values = triton_cfg_guard
    benchmark = ShortMultipleAllTechniquesBenchmark()
    original_compile = torch.compile

    def exploding_compile(*args, **kwargs):
        raise RuntimeError("intentional compile failure")

    torch.compile = exploding_compile  # type: ignore[assignment]
    try:
        try:
            benchmark.setup()
        except RuntimeError:
            pass
    finally:
        torch.compile = original_compile  # type: ignore[assignment]

    for attr, value in original_values.items():
        assert getattr(triton_cfg, attr) == value
