"""Harness-level enforcement tests for validate_environment().

These tests use a synthetic /proc + /sys snapshot via EnvironmentProbe and verify
that BenchmarkHarness FAILS for chapter/lab benchmarks when the environment is
invalid (i.e., validate_environment() returns errors).
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

import pytest

from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness
from core.harness.validity_checks import EnvironmentProbe


def _write_file(root: Path, relpath: str, content: str) -> None:
    path = root / relpath.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_base_env(root: Path, *, governor: str = "performance") -> None:
    _write_file(root, "/proc/swaps", "Filename\tType\tSize\tUsed\tPriority\n")
    _write_file(root, "/proc/sys/vm/swappiness", "0\n")
    _write_file(root, "/proc/cpuinfo", "processor\t: 0\n")
    _write_file(root, "/sys/devices/virtual/dmi/id/product_name", "BareMetal\n")
    _write_file(root, "/sys/devices/system/node/node0/cpulist", "0-3\n")
    _write_file(root, "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor", governor + "\n")


def _load_ch_fake_benchmark(module_dir: Path) -> object:
    """Load a benchmark class from a file path containing '/ch' so the harness enforces."""
    module_path = module_dir / "ch_fake_env_bench.py"
    module_path.write_text(
        textwrap.dedent(
            """
            import torch
            from core.harness.benchmark_harness import BaseBenchmark

            class EnvBench(BaseBenchmark):
                allow_cpu = True

                def __init__(self):
                    super().__init__()
                    self.x = None

                def setup(self) -> None:
                    self.x = torch.ones(1, device=self.device)

                def benchmark_fn(self) -> None:
                    self.x.add_(1)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("ch_fake_env_bench", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.EnvBench()


def _run_harness(env_root: Path, *, probe: EnvironmentProbe, allow_virtualization: Optional[bool] = None) -> list[str]:
    bench_dir = env_root / "bench_mod"
    bench_dir.mkdir(parents=True, exist_ok=True)
    bench = _load_ch_fake_benchmark(bench_dir)
    harness = BenchmarkHarness(environment_probe=probe)
    config_kwargs = dict(iterations=1, warmup=5, use_subprocess=False)
    if allow_virtualization is not None:
        config_kwargs["allow_virtualization"] = allow_virtualization
    config = BenchmarkConfig(**config_kwargs)
    result = harness._benchmark_with_threading(bench, config)
    return list(result.errors)


def test_environment_enforcement_cpu_governor_mismatch() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root, governor="powersave")
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}))
        assert any("ENVIRONMENT INVALID" in e and "CPU governor mismatch" in e for e in errors), errors


def test_environment_enforcement_cgroup_cpu_quota() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/proc/self/cgroup", "0::/test.slice\n")
        _write_file(env_root, "/sys/fs/cgroup/test.slice/cpu.max", "100000 100000\n")
        _write_file(env_root, "/sys/fs/cgroup/test.slice/memory.max", "max\n")
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}))
        assert any("ENVIRONMENT INVALID" in e and "cpu.max" in e for e in errors), errors


def test_environment_enforcement_cgroup_memory_limit() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/proc/self/cgroup", "0::/test.slice\n")
        _write_file(env_root, "/sys/fs/cgroup/test.slice/cpu.max", "max 100000\n")
        _write_file(env_root, "/sys/fs/cgroup/test.slice/memory.max", "1073741824\n")
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}))
        assert any("ENVIRONMENT INVALID" in e and "memory.max" in e for e in errors), errors


def test_environment_enforcement_numa_affinity_spans_nodes() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/sys/devices/system/node/node0/cpulist", "0-1\n")
        _write_file(env_root, "/sys/devices/system/node/node1/cpulist", "2-3\n")
        probe = EnvironmentProbe(root=env_root, env={}, cpu_affinity={0, 2})
        errors = _run_harness(env_root, probe=probe)
        assert any("ENVIRONMENT INVALID" in e and "NUMA" in e for e in errors), errors


def test_environment_enforcement_swap_enabled() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(
            env_root,
            "/proc/swaps",
            "Filename\tType\tSize\tUsed\tPriority\n/swapfile\tfile\t1024\t0\t-2\n",
        )
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}))
        assert any("ENVIRONMENT INVALID" in e and "Swap is enabled" in e for e in errors), errors


def test_environment_enforcement_virtualization_detected() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/proc/cpuinfo", "processor\t: 0\nflags\t: hypervisor\n")
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}), allow_virtualization=False)
        assert any("ENVIRONMENT INVALID" in e and "Virtualization detected" in e for e in errors), errors
