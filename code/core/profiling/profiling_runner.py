"""Unified profiling runner for nsys, ncu, and Proton.

Provides single interface for running profiler backends with proper timeout handling,
error recovery, and artifact management.
"""

from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING, cast

# Use TYPE_CHECKING to avoid circular import at type-checking time
if TYPE_CHECKING:
    from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
    from core.profiling.metrics_extractor import NcuMetrics, NsysMetrics
else:
    NsysMetrics = Any
    NcuMetrics = Any

# Lazy initialization to avoid circular import at runtime
BenchmarkType = Any
BenchmarkConfigType = Any
BENCHMARK_AVAILABLE = False
_benchmark_types_initialized = False

def _init_benchmark_types():
    """Lazy initialization of benchmark types to avoid circular import."""
    global BenchmarkType, BenchmarkConfigType, BENCHMARK_AVAILABLE, _benchmark_types_initialized
    if _benchmark_types_initialized:
        return
    
    try:
        import core.harness.benchmark_harness as _benchmark_harness
        BenchmarkType = _benchmark_harness.Benchmark
        BenchmarkConfigType = _benchmark_harness.BenchmarkConfig
        BENCHMARK_AVAILABLE = True
    except (ImportError, AttributeError):
        BENCHMARK_AVAILABLE = False
    
    _benchmark_types_initialized = True


class WrapperFactory(Protocol):
    def __call__(self, benchmark: BenchmarkType, benchmark_module: Any, benchmark_class: str, config: BenchmarkConfigType) -> Optional[Path]:
        ...

import torch

def _unavailable_wrapper(*args, **kwargs) -> Optional[Path]:
    return None

METRICS_EXTRACTOR_AVAILABLE = False
create_benchmark_wrapper: WrapperFactory = _unavailable_wrapper

try:
    from core.profiling.metrics_extractor import extract_nsys_metrics, extract_ncu_metrics, extract_proton_metrics
    from core.profiling import profiler_config as _profiler_config
    from core.profiling.profiler_wrapper import create_benchmark_wrapper as _create_wrapper
    create_benchmark_wrapper = cast(WrapperFactory, _create_wrapper)
    METRICS_EXTRACTOR_AVAILABLE = True
except ImportError:
    def extract_nsys_metrics(nsys_rep_path: Path, timeout: int = 60) -> NsysMetrics:
        raise ImportError("metrics_extractor not available")
    
    def extract_ncu_metrics(ncu_rep_path: Path, timeout: int = 60) -> NcuMetrics:
        raise ImportError("metrics_extractor not available")
    _profiler_config = None


def check_nsys_available() -> bool:
    """Check if nsys is available on the system."""
    if shutil.which("nsys") is None:
        return False
    try:
        result = subprocess.run(
            ["nsys", "--version"],
            capture_output=True,
            timeout=5,
            check=False,
            env=os.environ.copy()
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_ncu_available() -> bool:
    """Check if ncu is available on the system."""
    if shutil.which("ncu") is None:
        return False
    try:
        result = subprocess.run(
            ["ncu", "--version"],
            capture_output=True,
            timeout=5,
            check=False,
            env=os.environ.copy()
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_proton_available() -> bool:
    """Check if Proton profiler is available."""
    if shutil.which("proton") is not None:
        return True
    try:
        import importlib
        importlib.import_module("torch._inductor.tools.proton")
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _export_ncu_csv(ncu_rep: Path, metrics: Optional[List[str]] = None, timeout: int = 120) -> Optional[Path]:
    """Export an ncu-rep to CSV so deep_profiling_report can consume it."""
    if not ncu_rep.exists():
        return None
    csv_path = ncu_rep.with_suffix(".csv")
    cmd = [
        "ncu",
        "--csv",
        "--page",
        "details",
        "--import",
        str(ncu_rep),
    ]
    if metrics:
        cmd.extend(["--metrics", ",".join(metrics)])
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    try:
        csv_path.write_text(result.stdout)
    except Exception:
        return None
    return csv_path


def _run_deep_profiling_report(
    ncu_csv: Path,
    nsys_report: Optional[Path],
    output_base: Path,
    *,
    timeout: int = 90,
) -> Optional[Dict[str, str]]:
    """Invoke the deep profiling reporter (markdown + JSON)."""
    script = _repo_root() / "tools" / "analysis" / "deep_profiling_report.py"
    if not script.exists() or not ncu_csv.exists():
        return None
    md_path = output_base.with_suffix(".md")
    json_path = output_base.with_suffix(".json")
    cmd = [
        sys.executable,
        str(script),
        "--ncu-csv",
        str(ncu_csv),
        "--output-json",
        str(json_path),
        "--print-markdown",
    ]
    if nsys_report and nsys_report.exists():
        cmd.extend(["--nsys-report", str(nsys_report)])
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    try:
        md_path.write_text(result.stdout)
    except Exception:
        return None
    return {"deep_profile_md": str(md_path), "deep_profile_json": str(json_path)}


def run_nsys_profiling(
    benchmark: BenchmarkType,
    benchmark_module: Any,
    benchmark_class: str,
    output_dir: Path,
    config: BenchmarkConfigType,
    profiler_config: Optional[Any] = None,
    timeout_seconds: Optional[int] = None,
    metrics: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Run nsys profiling on a benchmark.
    
    Args:
        benchmark: Benchmark instance to profile
        benchmark_module: Module containing the benchmark class
        benchmark_class: Name of benchmark class
        output_dir: Directory for profiling outputs
        config: Benchmark configuration
        profiler_config: Optional profiler configuration (uses default if None)
        timeout_seconds: Optional timeout for profiling run
        metrics: Optional metrics set name (e.g., 'roofline', 'deep_dive')
    
    Returns:
        Dictionary with 'profiling_outputs' and 'metrics' keys, or None on failure
    """
    _init_benchmark_types()
    if not BENCHMARK_AVAILABLE or not METRICS_EXTRACTOR_AVAILABLE:
        return None
    
    if profiler_config is None and _profiler_config is not None:
        profiler_config = _profiler_config.build_profiler_config_from_benchmark(
            config,
            benchmark_module=benchmark_module,
            benchmark_class=benchmark_class,
        )
    if profiler_config is None:
        return None
    
    # Create wrapper script
    wrapper_script = create_benchmark_wrapper(
        benchmark, benchmark_module, benchmark_class, config
    )
    
    if not wrapper_script:
        return None
    
    try:
        # Create output path
        nsys_output = output_dir / f"nsys_{benchmark_class}"
        
        # Build nsys command using profiler_config
        nsys_command = profiler_config.get_nsys_command(
            str(nsys_output),
            str(wrapper_script),
            nvtx_includes=getattr(profiler_config, "nvtx_includes", None),
        )
        
        # Run nsys
        result = subprocess.run(
            nsys_command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=os.environ.copy()
        )
        
        if result.returncode != 0:
            return None
        
        # Extract metrics
        nsys_rep_path = Path(f"{nsys_output}.nsys-rep")
        if not nsys_rep_path.exists():
            return None
        
        metrics_obj = extract_nsys_metrics(nsys_rep_path)
        
        return {
            "profiling_outputs": {"nsys_rep": str(nsys_rep_path)},
            "metrics": metrics_obj
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def run_ncu_profiling(
    benchmark: BenchmarkType,
    benchmark_module: Any,
    benchmark_class: str,
    output_dir: Path,
    config: BenchmarkConfigType,
    profiler_config: Optional[Any] = None,
    timeout_seconds: Optional[int] = None,
    metrics: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Run ncu profiling on a benchmark.
    
    Args:
        benchmark: Benchmark instance to profile
        benchmark_module: Module containing the benchmark class
        benchmark_class: Name of benchmark class
        output_dir: Directory for profiling outputs
        config: Benchmark configuration
        profiler_config: Optional profiler configuration (uses default if None)
        timeout_seconds: Optional timeout for profiling run
        metrics: Optional metrics set name (e.g., 'roofline', 'deep_dive')
    
    Returns:
        Dictionary with 'profiling_outputs' and 'metrics' keys, or None on failure
    """
    _init_benchmark_types()
    if not BENCHMARK_AVAILABLE or not METRICS_EXTRACTOR_AVAILABLE:
        return None
    
    if profiler_config is None and _profiler_config is not None:
        profiler_config = _profiler_config.build_profiler_config_from_benchmark(
            config,
            benchmark_module=benchmark_module,
            benchmark_class=benchmark_class,
        )
    if profiler_config is None:
        return None
    
    # Create wrapper script
    wrapper_script = create_benchmark_wrapper(
        benchmark, benchmark_module, benchmark_class, config
    )
    
    if not wrapper_script:
        return None
    
    try:
        # Create output path
        ncu_output = output_dir / f"ncu_{benchmark_class}"
        
        # Build ncu command using profiler_config
        metrics_list = None
        if metrics:
            metrics_list = [metric.strip() for metric in metrics.split(",") if metric.strip()]
        ncu_command = profiler_config.get_ncu_command(
            str(ncu_output),
            str(wrapper_script),
            metrics=metrics_list,
            nvtx_includes=getattr(profiler_config, "nvtx_includes", None),
        )
        
        # Run ncu
        result = subprocess.run(
            ncu_command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=os.environ.copy()
        )
        
        if result.returncode != 0:
            return None
        
        # Extract metrics
        ncu_rep_path = Path(f"{ncu_output}.ncu-rep")
        if not ncu_rep_path.exists():
            return None
        
        metrics_obj = extract_ncu_metrics(ncu_rep_path)
        
        return {
            "profiling_outputs": {"ncu_rep": str(ncu_rep_path)},
            "metrics": metrics_obj
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def run_proton_profiling(
    benchmark: BenchmarkType,
    benchmark_module: Any,
    benchmark_class: str,
    output_dir: Path,
    config: BenchmarkConfigType,
    profiler_config: Optional[Any] = None,
    timeout_seconds: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Run Proton profiling on a benchmark."""
    _init_benchmark_types()
    if not BENCHMARK_AVAILABLE or not METRICS_EXTRACTOR_AVAILABLE:
        return None
    
    if profiler_config is None and _profiler_config is not None:
        profiler_config = _profiler_config.build_profiler_config_from_benchmark(
            config,
            benchmark_module=benchmark_module,
            benchmark_class=benchmark_class,
        )
    if profiler_config is None:
        return None
    
    wrapper_script = create_benchmark_wrapper(
        benchmark, benchmark_module, benchmark_class, config
    )
    if not wrapper_script:
        return None
    
    try:
        proton_output = output_dir / f"proton_{benchmark_class}"
        if hasattr(profiler_config, "get_proton_command"):
            proton_command = profiler_config.get_proton_command(
                str(proton_output),
                str(wrapper_script),
            )
        else:
            proton_command = [
                "proton",
                "profile",
                "--output",
                f"{proton_output}.json",
                "--python-script",
                str(wrapper_script),
            ]
        
        result = subprocess.run(
            proton_command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            return None
        
        report_path = proton_output if proton_output.suffix else proton_output.with_suffix(".json")
        if not report_path.exists():
            # Proton may honor explicit suffixes; try both
            alt = Path(f"{proton_output}.json")
            report_path = alt if alt.exists() else report_path
        if not report_path.exists():
            return None
        
        metrics_obj = extract_proton_metrics(report_path)
        return {
            "profiling_outputs": {"proton_report": str(report_path)},
            "metrics": metrics_obj,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def run_profiling_orchestration(
    benchmark: BenchmarkType,
    config: BenchmarkConfigType,
    timing_fn: Callable[[Callable[..., Any], BenchmarkConfigType], List[float]],
    output_dir: Optional[Path] = None,
    profiler_config: Optional[Any] = None
) -> Dict[str, Any]:
    """Unified profiling orchestration: runs timing benchmark and optional nsys/ncu profiling.
    
    This function centralizes all profiling orchestration logic, including:
    - Running the timing benchmark
    - Creating profiling output directories
    - Running nsys profiling if enabled
    - Running ncu profiling if enabled
    - Extracting and returning metrics
    
    Args:
        benchmark: Benchmark instance to profile
        config: Benchmark configuration (with enable_nsys/enable_ncu flags)
        timing_fn: Function to run timing benchmark (takes benchmark_fn, config, returns List[float])
        output_dir: Optional output directory (defaults to "profiling_results")
        profiler_config: Optional profiler configuration
    
    Returns:
        Dictionary with:
        - 'times_ms': List[float] - timing results
        - 'profiling_outputs': Dict[str, str] - paths to profiling artifacts
        - 'nsys_metrics': Dict or Pydantic model - nsys metrics if available
        - 'ncu_metrics': Dict or Pydantic model - ncu metrics if available
    """
    _init_benchmark_types()
    if not BENCHMARK_AVAILABLE or not METRICS_EXTRACTOR_AVAILABLE:
        # Fallback: just run timing
        times_ms = timing_fn(benchmark.benchmark_fn, config)
        return {
            "times_ms": times_ms,
            "profiling_outputs": {},
            "nsys_metrics": {},
            "ncu_metrics": {},
        }
    
    # Create profiling output directory
    if output_dir is None:
        if config.profiling_output_dir:
            prof_dir = Path(config.profiling_output_dir)
        else:
            prof_dir = Path("profiling_results")
    else:
        prof_dir = Path(output_dir)
    
    prof_dir.mkdir(parents=True, exist_ok=True)
    
    # Get benchmark module and class info
    benchmark_module = inspect.getmodule(benchmark)
    benchmark_class = benchmark.__class__.__name__
    
    # If module is None, try to get it from the class
    if benchmark_module is None:
        benchmark_module = inspect.getmodule(benchmark.__class__)
    
    prof_cfg = profiler_config
    if prof_cfg is None and _profiler_config is not None:
        prof_cfg = _profiler_config.build_profiler_config_from_benchmark(
            config,
            benchmark_module=benchmark_module,
            benchmark_class=benchmark_class,
        )
    
    # Run timing benchmark first
    times_ms = timing_fn(benchmark.benchmark_fn, config)
    
    # Initialize return values
    profiling_outputs = {}
    nsys_metrics = {}
    ncu_metrics = {}
    proton_metrics = {}
    
    # Run nsys profiling if enabled and available
    if config.enable_nsys and check_nsys_available():
        if benchmark_module is not None:
            # Use profiling_timeout_seconds if set, otherwise fall back to nsys_timeout_seconds
            profiling_timeout = getattr(config, 'get_effective_timeout', None)
            if profiling_timeout:
                nsys_timeout = profiling_timeout('profiling') or config.nsys_timeout_seconds
            else:
                nsys_timeout = getattr(config, 'profiling_timeout_seconds', None) or config.nsys_timeout_seconds
            nsys_result = run_nsys_profiling(
                benchmark, benchmark_module, benchmark_class, prof_dir, config,
                profiler_config=prof_cfg,
                timeout_seconds=nsys_timeout
            )
            if nsys_result:
                profiling_outputs.update(nsys_result.get("profiling_outputs", {}))
                metrics_obj = nsys_result.get("metrics", {})
                if isinstance(metrics_obj, dict):
                    nsys_metrics = metrics_obj
                else:
                    nsys_metrics = metrics_obj.to_dict() if hasattr(metrics_obj, 'to_dict') else {}
    
    # Run ncu profiling if enabled and available
    if config.enable_ncu and check_ncu_available():
        if benchmark_module is not None:
            # Use profiling_timeout_seconds if set, otherwise fall back to ncu_timeout_seconds
            profiling_timeout = getattr(config, 'get_effective_timeout', None)
            if profiling_timeout:
                ncu_timeout = profiling_timeout('profiling') or config.ncu_timeout_seconds
            else:
                ncu_timeout = getattr(config, 'profiling_timeout_seconds', None) or config.ncu_timeout_seconds
            ncu_result = run_ncu_profiling(
                benchmark, benchmark_module, benchmark_class, prof_dir, config,
                profiler_config=prof_cfg,
                timeout_seconds=ncu_timeout
            )
            if ncu_result:
                profiling_outputs.update(ncu_result.get("profiling_outputs", {}))
                metrics_obj = ncu_result.get("metrics", {})
                if isinstance(metrics_obj, dict):
                    ncu_metrics = metrics_obj
                else:
                    ncu_metrics = metrics_obj.to_dict() if hasattr(metrics_obj, 'to_dict') else {}
    
    # Run Proton profiling if enabled and available
    if getattr(config, "enable_proton", False) and check_proton_available():
        if benchmark_module is not None:
            profiling_timeout = getattr(config, 'get_effective_timeout', None)
            if profiling_timeout:
                proton_timeout = profiling_timeout('profiling') or getattr(config, 'proton_timeout_seconds', None)
            else:
                proton_timeout = getattr(config, 'proton_timeout_seconds', None) or getattr(config, 'profiling_timeout_seconds', None)
            proton_result = run_proton_profiling(
                benchmark, benchmark_module, benchmark_class, prof_dir, config,
                profiler_config=prof_cfg,
                timeout_seconds=proton_timeout
            )
            if proton_result:
                profiling_outputs.update(proton_result.get("profiling_outputs", {}))
                metrics_obj = proton_result.get("metrics", {})
                if isinstance(metrics_obj, dict):
                    proton_metrics = metrics_obj
                else:
                    proton_metrics = metrics_obj.to_dict() if hasattr(metrics_obj, 'to_dict') else {}

    return {
        "times_ms": times_ms,
        "profiling_outputs": profiling_outputs,
        "nsys_metrics": nsys_metrics,
        "ncu_metrics": ncu_metrics,
        "proton_metrics": proton_metrics,
    }
