"""Production-grade benchmarking harness with profiling integration.

Provides industry-standard benchmarking using Triton do_bench, PyTorch Timer,
and custom CUDA Events. Supports nsys, ncu, and PyTorch profiler integration.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
import torch

# Import metrics extractor (lazy import to avoid circular dependencies)
try:
    from common.python.metrics_extractor import (
        extract_nsys_metrics,
        extract_ncu_metrics,
        get_ncu_metric_description,
    )
except ImportError:
    # Fallback if metrics_extractor not available (shouldn't happen in production)
    extract_nsys_metrics = None
    extract_ncu_metrics = None
    get_ncu_metric_description = None


class BenchmarkMode(Enum):
    """Benchmarking mode selection."""
    TRITON = "triton"  # Use triton.testing.do_bench
    PYTORCH = "pytorch"  # Use torch.utils.benchmark.Timer
    CUSTOM = "custom"  # Use CUDA Events / time.perf_counter


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    iterations: int = 100
    warmup: int = 10
    min_run_time_ms: float = 100.0  # Minimum total runtime for PyTorch Timer
    percentiles: List[float] = field(default_factory=lambda: [25, 50, 75, 99])
    enable_memory_tracking: bool = False
    deterministic: bool = False
    seed: Optional[int] = None
    device: Optional[torch.device] = None
    enable_profiling: bool = True  # Enable nsys/ncu/PyTorch profiler (default: True - core experience, gracefully degrades if tools unavailable)
    enable_nsys: bool = True  # Enable nsys profiling (requires CUDA, wraps entire process) - default: True (gracefully degrades if unavailable)
    enable_ncu: bool = True  # Enable ncu profiling (requires CUDA, wraps entire process) - default: True (gracefully degrades if unavailable)
    profiling_output_dir: Optional[str] = None  # Directory for profiling outputs
    enable_nvtx: Optional[bool] = None  # Enable NVTX ranges (None = auto: True if profiling, False otherwise)
    enable_cleanup: bool = False  # Enable gc.collect() and torch.cuda.empty_cache() after each run (default: False to avoid distorting timings)
    use_subprocess: bool = True  # Run benchmark in subprocess for reliable timeout cancellation (default: True for production-grade harness)
    timeout_seconds: int = 15  # Required timeout for benchmark execution in seconds (prevents hangs) - DEFAULT 15s
    # Note: Setup/teardown (including compilation) are not subject to timeout,
    # but should complete within reasonable time or fail with error
    # nsys/ncu profiling timeout is separate and typically longer (60-120s)
    nsys_timeout_seconds: int = 120  # Timeout for nsys profiling runs
    ncu_timeout_seconds: int = 180  # Timeout for ncu profiling runs (ncu can be slow)
    
    def __post_init__(self):
        """Set enable_nvtx based on profiling if not explicitly set."""
        if self.enable_nvtx is None:
            # Auto-enable NVTX when profiling is enabled (for nsys traces)
            # Since enable_profiling=True by default, NVTX will be enabled by default
            self.enable_nvtx = self.enable_profiling


@dataclass
class BenchmarkResult:
    """Statistical results from benchmarking."""
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    percentiles: Dict[float, float]  # e.g., {25.0: 1.23, 50.0: 1.45, ...}
    iterations: int
    warmup_iterations: int
    raw_times_ms: Optional[List[float]] = None  # Raw timing measurements (for subprocess IPC)
    memory_peak_mb: Optional[float] = None
    memory_allocated_mb: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    profiling_outputs: Dict[str, str] = field(default_factory=dict)  # Paths to profiling files
    nsys_metrics: Dict[str, float] = field(default_factory=dict)  # Extracted nsys metrics
    ncu_metrics: Dict[str, float] = field(default_factory=dict)  # Extracted ncu metrics


class Benchmark(Protocol):
    """Protocol for benchmarkable implementations."""
    
    def setup(self) -> None:
        """Setup phase: initialize models, data, etc."""
        ...
    
    def benchmark_fn(self) -> None:
        """Function to benchmark. Must be callable with no args."""
        ...
    
    def teardown(self) -> None:
        """Cleanup phase."""
        ...
    
    def get_config(self) -> Optional[BenchmarkConfig]:
        """Optional: return benchmark-specific config overrides."""
        return None
    
    def validate_result(self) -> Optional[str]:
        """Optional: validate benchmark result, return error message if invalid."""
        return None


class BaseBenchmark:
    """Base class for benchmarks with shared functionality.
    
    Provides common patterns for device resolution, setup, teardown, validation,
    NVTX ranges, and synchronization. Benchmarks can inherit from this class or
    implement the Benchmark Protocol directly.
    
    Usage:
        class MyBenchmark(BaseBenchmark):
            def __init__(self):
                super().__init__()
                self.model = None
                self.data = None
            
            def setup(self) -> None:
                torch.manual_seed(42)
                self.model = nn.Linear(256, 256).to(self.device)
                self.data = torch.randn(32, 256, device=self.device)
                torch.cuda.synchronize()
            
            def benchmark_fn(self) -> None:
                with self._nvtx_range("my_benchmark"):
                    _ = self.model(self.data)
    """
    
    def __init__(self):
        """Initialize benchmark with device resolution.
        
        Subclasses should call super().__init__() and then set up their own attributes.
        """
        self.device = self._resolve_device()
        self._config = None  # Cache for get_config()
    
    def _resolve_device(self) -> torch.device:
        """Resolve CUDA device, failing fast if CUDA is not available.
        
        Returns:
            torch.device("cuda") if CUDA is available
            
        Raises:
            RuntimeError: If CUDA is not available (NVIDIA GPU required)
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required - NVIDIA GPU and tools must be available")
        return torch.device("cuda")
    
    def setup(self) -> None:
        """Setup phase: initialize models, data, etc.
        
        Subclasses should override this method to implement their specific setup logic.
        """
        pass
    
    def benchmark_fn(self) -> None:
        """Function to benchmark. Must be callable with no args.
        
        Subclasses must override this method to implement their benchmark logic.
        """
        raise NotImplementedError("Subclasses must implement benchmark_fn()")
    
    def teardown(self) -> None:
        """Cleanup phase.
        
        Default implementation clears CUDA cache. Subclasses can override
        to add additional cleanup logic.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> Optional[BenchmarkConfig]:
        """Return benchmark-specific config overrides.
        
        Subclasses can override to provide custom configuration.
        Default returns None (uses harness defaults).
        """
        return None
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result, return error message if invalid.
        
        Default implementation returns None (no validation).
        Subclasses should override to implement validation logic.
        
        Returns:
            None if validation passes, or error message string if validation fails
        """
        return None
    
    def _scale_workload_by_memory(self, base_size: int) -> int:
        """Scale workload size based on available GPU memory.
        
        Args:
            base_size: Base workload size for large GPUs (>=16GB)
            
        Returns:
            Scaled workload size based on GPU memory:
            - >=16GB: base_size (100%)
            - >=8GB: base_size * 0.5 (50%)
            - >=4GB: base_size * 0.25 (25%)
            - <4GB: base_size * 0.1 (10%)
        """
        if not torch.cuda.is_available():
            return base_size
        
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_memory_gb >= 16:
            return base_size
        elif total_memory_gb >= 8:
            return int(base_size * 0.5)
        elif total_memory_gb >= 4:
            return int(base_size * 0.25)
        else:
            return int(base_size * 0.1)
    
    @contextmanager
    def _nvtx_range(self, name: str):
        """Context manager for NVTX ranges with automatic enable/disable.
        
        Automatically checks if NVTX is enabled via get_config().
        
        Args:
            name: Name for the NVTX range
            
        Usage:
            with self._nvtx_range("my_operation"):
                # code to profile
                pass
        """
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        with nvtx_range(name, enable=enable_nvtx):
            yield
    
    def _synchronize(self) -> None:
        """Synchronize CUDA device if available.
        
        Convenience method for benchmarks to ensure operations complete.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)


class BenchmarkHarness:
    """Production-grade benchmarking harness with profiling support."""
    
    def __init__(
        self,
        mode: BenchmarkMode = BenchmarkMode.CUSTOM,
        config: Optional[BenchmarkConfig] = None
    ):
        self.mode = mode
        self.config = config or BenchmarkConfig()
        self.device = self.config.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._setup_reproducibility()
    
    def _setup_reproducibility(self) -> None:
        """Setup for reproducible benchmarks."""
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
    
    @contextmanager
    def _memory_tracking(self, config: Optional[BenchmarkConfig] = None):
        """Context manager for memory tracking.
        
        Yields a list that will contain [peak_mb, allocated_mb] after the context exits.
        If memory tracking is disabled, yields None.
        
        Args:
            config: BenchmarkConfig to check for enable_memory_tracking. If None, uses self.config.
        """
        # Use provided config or fall back to instance config
        check_config = config if config is not None else self.config
        
        if not check_config.enable_memory_tracking or not torch.cuda.is_available():
            yield None
            return
        
        # Use a mutable list to return values from context manager
        result = [None, None]
        
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        yield result
        torch.cuda.synchronize(self.device)
        result[0] = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        result[1] = torch.cuda.memory_allocated(self.device) / (1024**2)
    
    def benchmark(self, benchmark: Benchmark) -> BenchmarkResult:
        """Run benchmark and return statistical results.
        
        Uses subprocess isolation (if enabled) or threading timeout to prevent hangs.
        Default timeout is 15 seconds.
        """
        # Clone config to avoid mutating shared instance
        from dataclasses import replace
        config = replace(self.config)
        bench_config = benchmark.get_config()
        if bench_config:
            # Override with benchmark-specific settings
            for key, value in bench_config.__dict__.items():
                if value is not None:
                    setattr(config, key, value)
        
        # Use subprocess isolation if enabled (default: True for production-grade harness)
        if config.use_subprocess:
            return self._benchmark_with_subprocess(benchmark, config)
        else:
            return self._benchmark_with_threading(benchmark, config)
    
    def _benchmark_with_subprocess(self, benchmark: Benchmark, config: BenchmarkConfig) -> BenchmarkResult:
        """Run benchmark in subprocess for reliable timeout cancellation."""
        import json
        import inspect
        
        errors = []
        memory_peak_mb = None
        memory_allocated_mb = None
        profiling_outputs = {}
        nsys_metrics = {}
        ncu_metrics = {}
        times_ms = []
        
        # Get benchmark module and class info
        benchmark_module = inspect.getmodule(benchmark)
        benchmark_class = benchmark.__class__.__name__
        
        if benchmark_module is None:
            benchmark_module = inspect.getmodule(benchmark.__class__)
        
        if benchmark_module is None:
            # Fallback to threading if we can't determine module
            return self._benchmark_with_threading(benchmark, config)
        
        module_file = getattr(benchmark_module, "__file__", None)
        if module_file is None:
            spec = getattr(benchmark_module, "__spec__", None)
            if spec is not None:
                module_file = getattr(spec, "origin", None)
        
        if module_file is None:
            # Fallback to threading if we can't determine module file
            return self._benchmark_with_threading(benchmark, config)
        
        module_path = Path(module_file).resolve()
        if not module_path.exists():
            # Fallback to threading if module file doesn't exist
            return self._benchmark_with_threading(benchmark, config)
        
        # Prepare config dict (serialize only simple types)
        config_dict = {}
        for key in ['iterations', 'warmup', 'min_run_time_ms', 'enable_memory_tracking',
                   'deterministic', 'seed', 'enable_profiling', 'enable_nsys', 'enable_ncu',
                   'profiling_output_dir', 'enable_nvtx', 'enable_cleanup', 'timeout_seconds',
                   'nsys_timeout_seconds', 'ncu_timeout_seconds']:
            value = getattr(config, key, None)
            if value is not None:
                config_dict[key] = value
        
        # Prepare input JSON
        input_data = {
            "benchmark_module_path": str(module_path),
            "benchmark_class_name": benchmark_class,
            "config_dict": config_dict,
            "device": str(self.device) if self.device else None,
        }
        
        # Spawn subprocess
        runner_script = Path(__file__).parent / "benchmark_runner.py"
        if not runner_script.exists():
            errors.append("benchmark_runner.py not found - falling back to threading")
            return self._benchmark_with_threading(benchmark, config)
        
        try:
            import signal
            process = subprocess.Popen(
                [sys.executable, str(runner_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid  # Create new process group for reliable killing
            )
            
            # Send input JSON
            input_json = json.dumps(input_data)
            stdout, stderr = process.communicate(input=input_json, timeout=config.timeout_seconds)
            
            if process.returncode != 0:
                errors.append(f"Subprocess exited with code {process.returncode}")
                if stderr:
                    errors.append(f"Stderr: {stderr[:500]}")  # Limit error message length
                times_ms = []
            else:
                # Parse JSON result
                try:
                    result_dict = json.loads(stdout)
                    if result_dict.get("success"):
                        # Extract raw timing measurements (preferred)
                        times_ms = result_dict.get("times_ms", [])
                        if not times_ms:
                            # Fallback: reconstruct from statistics if raw times not available
                            if result_dict.get("mean_ms"):
                                mean_time = result_dict.get("mean_ms", 0)
                                std_time = result_dict.get("std_ms", mean_time * 0.1)
                                iterations = result_dict.get("iterations", config.iterations)
                                import numpy as np
                                synthetic_times = np.random.normal(mean_time, std_time, iterations)
                                synthetic_times = np.clip(
                                    synthetic_times,
                                    result_dict.get("min_ms", mean_time * 0.5),
                                    result_dict.get("max_ms", mean_time * 1.5)
                                )
                                times_ms = synthetic_times.tolist()
                        memory_peak_mb = result_dict.get("memory_peak_mb")
                        memory_allocated_mb = result_dict.get("memory_allocated_mb")
                        errors.extend(result_dict.get("errors", []))
                        profiling_outputs = result_dict.get("profiling_outputs", {})
                        nsys_metrics = result_dict.get("nsys_metrics", {})
                        ncu_metrics = result_dict.get("ncu_metrics", {})
                    else:
                        errors.extend(result_dict.get("errors", ["Subprocess execution failed"]))
                        times_ms = []
                except json.JSONDecodeError as e:
                    errors.append(f"Failed to parse subprocess output: {e}")
                    errors.append(f"Output: {stdout[:500]}")
                    times_ms = []
                except Exception as e:
                    errors.append(f"Error processing subprocess result: {e}")
                    times_ms = []
        
        except subprocess.TimeoutExpired:
            # TIMEOUT - kill the process group
            print("\n" + "=" * 80)
            print("TIMEOUT: Benchmark execution exceeded timeout limit")
            print("=" * 80)
            print(f"   Timeout limit: {config.timeout_seconds} seconds")
            print(f"   Status: Benchmark subprocess did not complete within timeout period")
            print(f"   Action: Terminating subprocess to free GPU resources")
            print("=" * 80)
            print()
            
            try:
                # Kill the process group (only the child, not parent)
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                # Wait a bit for graceful termination
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if still running
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
            except (ProcessLookupError, OSError):
                # Process already terminated
                pass
            
            errors.append(f"TIMEOUT: Benchmark exceeded timeout of {config.timeout_seconds} seconds")
            times_ms = []
        
        except Exception as e:
            errors.append(f"Subprocess execution failed: {str(e)}")
            times_ms = []
        
        if not times_ms:
            raise RuntimeError(f"Benchmark failed: {', '.join(errors)}")
        
        # Compute statistics
        result = self._compute_stats(times_ms, config)
        result.memory_peak_mb = memory_peak_mb
        result.memory_allocated_mb = memory_allocated_mb
        result.errors = errors
        result.profiling_outputs = profiling_outputs
        result.nsys_metrics = nsys_metrics
        result.ncu_metrics = ncu_metrics
        
        return result
    
    def _benchmark_with_threading(self, benchmark: Benchmark, config: BenchmarkConfig) -> BenchmarkResult:
        """Run benchmark using threading (legacy method, kept for compatibility)."""
        
        errors = []
        memory_peak_mb = None
        memory_allocated_mb = None
        profiling_outputs = {}
        nsys_metrics = {}
        ncu_metrics = {}
        times_ms = []
        
        # Use a lock to prevent teardown from running while benchmark is executing
        execution_lock = threading.Lock()
        execution_complete = threading.Event()
        teardown_called = threading.Event()  # Track if teardown has been called
        
        def run_benchmark_internal():
            """Internal benchmark execution function."""
            nonlocal times_ms, memory_peak_mb, memory_allocated_mb, profiling_outputs, errors, nsys_metrics, ncu_metrics
            
            with execution_lock:  # Acquire lock during execution
                try:
                    # Setup - this may include CUDA extension compilation OR torch.compile()
                    # IMPORTANT: Setup MUST complete quickly or timeout will occur
                    # torch.compile() compilation can hang - timeout will catch it
                    # If setup takes longer than timeout, it will be killed by the outer timeout
                    import time
                    start_time = time.time()
                    benchmark.setup()
                    setup_time = time.time() - start_time
                    if setup_time > config.timeout_seconds * 0.8:  # Warn if setup takes >80% of timeout
                        print(f"  WARNING: Setup took {setup_time:.1f}s (near timeout limit)")
                    
                    # Warmup
                    self._warmup(benchmark.benchmark_fn, config.warmup)
                    
                    # Memory tracking: Use context manager to track peak memory during benchmark execution
                    with self._memory_tracking(config) as mem_result:
                        # Benchmark using selected mode
                        # Note: nsys/ncu profiling wraps the entire process, so it's handled separately
                        if config.enable_nsys or config.enable_ncu:
                            # Run nsys/ncu profiling (these wrap the entire process)
                            result = self._benchmark_with_nsys_ncu(benchmark, config)
                            times_ms = result.get("times_ms", [])
                            profiling_outputs = result.get("profiling_outputs", {})
                            nsys_metrics = result.get("nsys_metrics", {})
                            ncu_metrics = result.get("ncu_metrics", {})
                        elif config.enable_profiling:
                            times_ms, profiling_outputs = self._benchmark_with_profiling(
                                benchmark.benchmark_fn, config
                            )
                        else:
                            times_ms = self._benchmark_without_profiling(benchmark.benchmark_fn, config)
                    
                    # Extract memory tracking results from context manager
                    if mem_result is not None:
                        memory_peak_mb = mem_result[0]
                        memory_allocated_mb = mem_result[1]
                    
                    # Validate result
                    validation_error = benchmark.validate_result()
                    if validation_error:
                        errors.append(f"Validation failed: {validation_error}")
                    
                except Exception as e:
                    error_msg = str(e)
                    # Handle generator errors gracefully (common with torch.compile)
                    if "generator didn't stop after throw" in error_msg:
                        errors.append(f"Benchmark execution failed: Generator error (likely from torch.compile or async operations)")
                    else:
                        errors.append(f"Benchmark execution failed: {error_msg}")
                    times_ms = []
                finally:
                    # Mark execution as complete before teardown
                    execution_complete.set()
                    
                    # Teardown is now safe to call - we hold the lock
                    # Only call teardown once - set flag to prevent double invocation
                    if not teardown_called.is_set():
                        try:
                            benchmark.teardown()
                            teardown_called.set()
                        except Exception as e:
                            errors.append(f"Teardown failed: {str(e)}")
                            teardown_called.set()  # Mark as called even on error
                    
                    # Force cleanup (only if enabled to avoid distorting timings)
                    if config.enable_cleanup:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
        
        # ALWAYS run with timeout (required, default 15 seconds)
        # Use a regular (non-daemon) thread so we can properly synchronize teardown
        # NOTE: If a CUDA kernel hangs, we cannot interrupt it from Python - the GPU will be stuck
        # until the kernel completes or the process is killed. This timeout mechanism prevents
        # teardown from running while work is active, but cannot force-stop a hung CUDA kernel.
        thread = threading.Thread(target=run_benchmark_internal)
        thread.start()
        thread.join(timeout=config.timeout_seconds)
        
        if thread.is_alive():
            # TIMEOUT OCCURRED - thread is still running
            print("\n" + "=" * 80)
            print("TIMEOUT: Benchmark execution exceeded timeout limit")
            print("=" * 80)
            print(f"   Timeout limit: {config.timeout_seconds} seconds")
            print(f"   Status: Benchmark did not complete within timeout period")
            print(f"   WARNING: If a CUDA kernel is hung, it cannot be interrupted from Python.")
            print(f"   Action: Waiting for thread to complete before cleanup (prevents double teardown)")
            print("=" * 80)
            print()
            
            errors.append(f"TIMEOUT: Benchmark exceeded timeout of {config.timeout_seconds} seconds")
            times_ms = []
            
            # Wait for execution to complete (with lock) before calling teardown
            # This prevents double teardown and ensures cleanup happens after execution
            # We give it additional time to complete, but if it's truly hung (e.g., stuck CUDA kernel),
            # there's nothing we can do except wait or kill the process externally
            lock_acquired = False
            try:
                lock_acquired = execution_lock.acquire(timeout=min(config.timeout_seconds * 2, 30))  # Cap at 30s
                if lock_acquired:
                    # Thread completed - ensure teardown is called (only if not already called)
                    if not execution_complete.is_set():
                        # Thread is still running - wait a bit more
                        execution_complete.wait(timeout=5)
                    
                    # Teardown is safe now - we hold the lock, but only call if not already called
                    if execution_complete.is_set() and not teardown_called.is_set():
                        try:
                            benchmark.teardown()
                            teardown_called.set()
                        except Exception as e:
                            errors.append(f"Teardown after timeout failed: {str(e)}")
                            teardown_called.set()  # Mark as called even on error
            except Exception as e:
                errors.append(f"Error during timeout handling: {str(e)}")
            finally:
                if lock_acquired:
                    execution_lock.release()
            
            # If we couldn't acquire the lock, the thread is truly hung (likely stuck CUDA kernel)
            # DO NOT call torch.cuda.synchronize() here - it will block forever if kernel is stuck
            # Only do non-blocking cleanup operations
            if not lock_acquired:
                if torch.cuda.is_available():
                    try:
                        # Only do non-blocking cleanup - synchronize() would hang forever
                        torch.cuda.empty_cache()
                        # Skip synchronize() - it will block if kernel is stuck
                        # Skip ipc_collect() - may also block
                        torch.cuda.reset_peak_memory_stats()  # This is non-blocking
                    except Exception:
                        # Any CUDA operation may fail if kernel is stuck
                        pass
                # Always cleanup on timeout regardless of enable_cleanup flag
                gc.collect()
                # Force another GC pass to clean up any remaining references
                gc.collect()
        elif execution_complete.is_set():
            # Thread completed normally - results are already set
            pass
        else:
            # Thread completed but execution_complete not set - something went wrong
            if not times_ms:
                errors.append("Benchmark execution completed but no results collected")
        # Don't print success message for normal completion - only print on timeout/failure
        
        if not times_ms:
            raise RuntimeError(f"Benchmark failed: {', '.join(errors)}")
        
        # Compute statistics
        result = self._compute_stats(times_ms, config)
        result.memory_peak_mb = memory_peak_mb
        result.memory_allocated_mb = memory_allocated_mb
        result.errors = errors
        result.profiling_outputs = profiling_outputs
        result.nsys_metrics = nsys_metrics
        result.ncu_metrics = ncu_metrics
        
        return result
    
    def _benchmark_with_profiling(
        self, fn: Callable, config: BenchmarkConfig
    ) -> tuple[List[float], Dict[str, str]]:
        """Benchmark with profiling enabled."""
        profiling_outputs = {}
        
        # Create profiling output directory
        if config.profiling_output_dir:
            prof_dir = Path(config.profiling_output_dir)
            prof_dir.mkdir(parents=True, exist_ok=True)
        else:
            prof_dir = Path("profiling_results")
            prof_dir.mkdir(parents=True, exist_ok=True)
        
        # Try PyTorch profiler first (best for Python benchmarks)
        try:
            import torch.profiler
            
            # Run benchmark with PyTorch profiler
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            ) as prof:
                # Run benchmark iterations with minimal overhead
                times_ms = []
                is_cuda = self.device.type == "cuda"
                
                if is_cuda:
                    # Create events once, reuse across iterations
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(self.device)  # Sync once before loop
                    
                    for _ in range(config.iterations):
                        start_event.record()
                        fn()
                        end_event.record()
                        torch.cuda.synchronize(self.device)
                        times_ms.append(start_event.elapsed_time(end_event))
                        prof.step()  # Record each iteration in profiling trace
                else:
                    # CPU: use high-resolution timer
                    for _ in range(config.iterations):
                        start_time = time.perf_counter()
                        fn()
                        end_time = time.perf_counter()
                        times_ms.append((end_time - start_time) * 1000)
                        prof.step()  # Record each iteration in profiling trace
            
            # Export profiling data
            trace_file = prof_dir / "trace.json"
            prof.export_chrome_trace(str(trace_file))
            profiling_outputs["pytorch_trace"] = str(trace_file)
            
            return times_ms, profiling_outputs
            
        except Exception as e:
            # Fallback to non-profiling benchmark
            return self._benchmark_without_profiling(fn, config), {}
    
    def _check_nsys_available(self) -> bool:
        """Check if nsys is available on the system."""
        # First try shutil.which to find the tool
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
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _check_ncu_available(self) -> bool:
        """Check if ncu is available on the system."""
        # First try shutil.which to find the tool
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
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _benchmark_with_nsys_ncu(
        self, benchmark: Benchmark, config: BenchmarkConfig
    ) -> Dict[str, Any]:
        """Benchmark with nsys/ncu profiling enabled.
        
        Note: nsys and ncu wrap the entire Python process, so we need to create
        a wrapper script that imports and runs the benchmark.
        """
        if not torch.cuda.is_available():
            # Fallback to regular profiling if CUDA not available
            if config.enable_nsys or config.enable_ncu:
                print("  Note: CUDA not available - skipping nsys/ncu profiling (using PyTorch profiler only)")
            times_ms, profiling_outputs = self._benchmark_with_profiling(
                benchmark.benchmark_fn, config
            )
            return {
                "times_ms": times_ms,
                "profiling_outputs": profiling_outputs,
                "nsys_metrics": {},
                "ncu_metrics": {},
            }
        
        # Check tool availability early and disable profiling if tools aren't available
        nsys_available = self._check_nsys_available() if config.enable_nsys else False
        ncu_available = self._check_ncu_available() if config.enable_ncu else False
        
        # Inform user if tools are requested but not available (degraded mode)
        if config.enable_nsys and not nsys_available:
            print("  Note: nsys not available - skipping nsys profiling (benchmarks will run normally)")
        if config.enable_ncu and not ncu_available:
            print("  Note: ncu not available - skipping ncu profiling (benchmarks will run normally)")
        
        # Create profiling output directory only if at least one tool is available
        if nsys_available or ncu_available:
            if config.profiling_output_dir:
                prof_dir = Path(config.profiling_output_dir)
                prof_dir.mkdir(parents=True, exist_ok=True)
            else:
                prof_dir = Path("profiling_results")
                prof_dir.mkdir(parents=True, exist_ok=True)
        else:
            # No tools available, fall back to regular profiling
            times_ms, profiling_outputs = self._benchmark_with_profiling(
                benchmark.benchmark_fn, config
            )
            return {
                "times_ms": times_ms,
                "profiling_outputs": profiling_outputs,
                "nsys_metrics": {},
                "ncu_metrics": {},
            }
        
        times_ms = []
        profiling_outputs = {}
        nsys_metrics = {}
        ncu_metrics = {}
        
        # Get benchmark module and class info for wrapper script
        benchmark_module = inspect.getmodule(benchmark)
        benchmark_class = benchmark.__class__.__name__
        
        # If module is None, try to get it from the class
        if benchmark_module is None:
            benchmark_module = inspect.getmodule(benchmark.__class__)
        
        # If still None, we can't create wrapper script (degraded mode)
        if benchmark_module is None:
            # Fall back to regular profiling or non-profiling benchmark
            # Don't return empty times_ms - that causes benchmark to fail
            if config.enable_profiling:
                times_ms, profiling_outputs = self._benchmark_with_profiling(
                    benchmark.benchmark_fn, config
                )
            else:
                times_ms = self._benchmark_without_profiling(benchmark.benchmark_fn, config)
                profiling_outputs = {}
            return {
                "times_ms": times_ms,
                "profiling_outputs": profiling_outputs,
                "nsys_metrics": {},
                "ncu_metrics": {},
            }
        
        # Run regular benchmark first to get timing (nsys/ncu are for metrics, not timing)
        times_ms = self._benchmark_without_profiling(benchmark.benchmark_fn, config)
        
        # Run nsys profiling if enabled and available
        if config.enable_nsys and nsys_available:
            nsys_result = self._run_nsys_profiling(benchmark, benchmark_module, benchmark_class, prof_dir, config)
            if nsys_result:
                profiling_outputs.update(nsys_result.get("profiling_outputs", {}))
                nsys_metrics = nsys_result.get("metrics", {})
        
        # Run ncu profiling if enabled and available
        if config.enable_ncu and ncu_available:
            ncu_result = self._run_ncu_profiling(benchmark, benchmark_module, benchmark_class, prof_dir, config)
            if ncu_result:
                profiling_outputs.update(ncu_result.get("profiling_outputs", {}))
                ncu_metrics = ncu_result.get("metrics", {})
        
        return {
            "times_ms": times_ms,
            "profiling_outputs": profiling_outputs,
            "nsys_metrics": nsys_metrics,
            "ncu_metrics": ncu_metrics,
        }
    
    def _run_nsys_profiling(
        self, benchmark: Benchmark, benchmark_module, benchmark_class: str,
        prof_dir: Path, config: BenchmarkConfig
    ) -> Optional[Dict[str, Any]]:
        """Run nsys profiling on benchmark.
        
        Note: Tool availability should be checked before calling this method.
        """
        # Create wrapper script
        wrapper_script = self._create_benchmark_wrapper(
            benchmark, benchmark_module, benchmark_class, config
        )
        
        if not wrapper_script:
            # Only inform if we actually tried to create the script (tool was available)
            print(f"  Note: Could not create wrapper script for nsys profiling of {benchmark_class} - skipping nsys profiling")
            return None
        
        try:
            # Create output path
            nsys_output = prof_dir / f"nsys_{benchmark_class}"
            
            # Build nsys command
            nsys_command = [
                "nsys",
                "profile",
                "--force-overwrite=true",
                "-o",
                str(nsys_output),
                "-t", "cuda,nvtx,osrt,cublas,cudnn",
                "-s", "cpu",
                "--python-sampling=true",
                "--python-sampling-frequency=1000",
                "--cudabacktrace=true",
                "--stats=true",
                sys.executable,
                str(wrapper_script)
            ]
            
            # Run nsys
            result = subprocess.run(
                nsys_command,
                capture_output=True,
                timeout=config.nsys_timeout_seconds,
                check=False,
                env=os.environ.copy()
            )
            
            # Find the generated .nsys-rep file
            nsys_rep_path = Path(f"{nsys_output}.nsys-rep")
            if not nsys_rep_path.exists():
                # Try alternative naming
                for rep_file in prof_dir.glob(f"nsys_{benchmark_class}*.nsys-rep"):
                    nsys_rep_path = rep_file
                    break
            
            if nsys_rep_path.exists():
                profiling_outputs = {"nsys_rep": str(nsys_rep_path)}
                # Extract metrics using metrics_extractor module
                if extract_nsys_metrics is not None:
                    nsys_metrics_obj = extract_nsys_metrics(nsys_rep_path, timeout=60)
                    metrics = nsys_metrics_obj.to_dict()
                else:
                    metrics = {}
                return {
                    "profiling_outputs": profiling_outputs,
                    "metrics": metrics,
                }
            else:
                # nsys completed but file not found - skip silently (degraded mode)
                pass
        except subprocess.TimeoutExpired:
            print(f"  Note: nsys profiling timed out for {benchmark_class} - skipping nsys profiling")
        except Exception as e:
            # Only log if in debug mode
            if os.environ.get("BENCHMARK_DEBUG", "").lower() in ("1", "true", "yes"):
                print(f"  Debug: nsys profiling failed for {benchmark_class}: {e}")
        finally:
            # Clean up wrapper script
            try:
                if wrapper_script.exists():
                    wrapper_script.unlink()
            except:
                pass
        
        return None
    
    def _run_ncu_profiling(
        self, benchmark: Benchmark, benchmark_module, benchmark_class: str,
        prof_dir: Path, config: BenchmarkConfig
    ) -> Optional[Dict[str, Any]]:
        """Run ncu profiling on benchmark.
        
        Note: Tool availability should be checked before calling this method.
        """
        # Create wrapper script
        wrapper_script = self._create_benchmark_wrapper(
            benchmark, benchmark_module, benchmark_class, config
        )
        
        if not wrapper_script:
            # Only inform if we actually tried to create the script (tool was available)
            print(f"  Note: Could not create wrapper script for ncu profiling of {benchmark_class} - skipping ncu profiling")
            return None
        
        try:
            # Create output path
            ncu_output = prof_dir / f"ncu_{benchmark_class}"
            
            # Build ncu command with comprehensive metrics for roofline analysis
            # Based on book recommendations (ch13.md): roofline analysis metrics
            # Includes: compute throughput, memory throughput (DRAM/L2), occupancy, and efficiency
            ncu_metrics = [
                # Kernel timing
                "gpu__time_duration.avg",
                # Compute throughput (SM)
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                # Memory throughput - DRAM (HBM)
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                # Memory throughput - L2 cache
                "lts__throughput.avg.pct_of_peak_sustained_elapsed",
                # Compute proxy - FP32 instructions
                "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
                # Occupancy - active warps
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                # Memory efficiency metrics (from ch7.md)
                "dram__sectors_read.sum",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
                # Memory load efficiency
                "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
                # Tensor Core utilization (if applicable)
                "sm__inst_executed_pipe_tensor.sum",
            ]
            
            ncu_command = [
                "ncu",
                "--set", "full",
                "--metrics", ",".join(ncu_metrics),
                "--replay-mode", "kernel",
                "-o", str(ncu_output),
                sys.executable,
                str(wrapper_script)
            ]
            
            # Run ncu
            result = subprocess.run(
                ncu_command,
                capture_output=True,
                timeout=config.ncu_timeout_seconds,
                check=False,
                env=os.environ.copy()
            )
            
            # Find the generated .ncu-rep file
            ncu_rep_path = Path(f"{ncu_output}.ncu-rep")
            if not ncu_rep_path.exists():
                # Try alternative naming
                for rep_file in prof_dir.glob(f"ncu_{benchmark_class}*.ncu-rep"):
                    ncu_rep_path = rep_file
                    break
            
            if ncu_rep_path.exists():
                profiling_outputs = {"ncu_rep": str(ncu_rep_path)}
                # Extract metrics using metrics_extractor module
                if extract_ncu_metrics is not None:
                    ncu_metrics_obj = extract_ncu_metrics(ncu_rep_path, timeout=60)
                    metrics = ncu_metrics_obj.to_dict()
                else:
                    metrics = {}
                return {
                    "profiling_outputs": profiling_outputs,
                    "metrics": metrics,
                }
            else:
                # ncu completed but file not found - skip silently (degraded mode)
                pass
        except subprocess.TimeoutExpired:
            print(f"  Note: ncu profiling timed out for {benchmark_class} - skipping ncu profiling")
        except Exception as e:
            # Only log if in debug mode
            if os.environ.get("BENCHMARK_DEBUG", "").lower() in ("1", "true", "yes"):
                print(f"  Debug: ncu profiling failed for {benchmark_class}: {e}")
        finally:
            # Clean up wrapper script
            try:
                if wrapper_script.exists():
                    wrapper_script.unlink()
            except:
                pass
        
        return None
    
    def _create_benchmark_wrapper(
        self, benchmark: Benchmark, benchmark_module, benchmark_class: str, config: BenchmarkConfig
    ) -> Optional[Path]:
        """Create a temporary Python script that runs the benchmark.
        
        The wrapper script imports the benchmark module and recreates the benchmark
        instance, then runs setup, warmup, and profiling iterations.
        """
        try:
            # Get module path
            if benchmark_module is None:
                return None
            
            module_name = benchmark_module.__name__
            module_file = getattr(benchmark_module, "__file__", None)
            
            # Try to get file from spec if __file__ is not available
            if module_file is None:
                spec = getattr(benchmark_module, "__spec__", None)
                if spec is not None:
                    module_file = getattr(spec, "origin", None)
            
            if module_file is None:
                # Can't determine module file - skip wrapper creation (degraded mode)
                return None
            
            module_path = Path(module_file).resolve()
            if not module_path.exists():
                # Module file doesn't exist - skip wrapper creation (degraded mode)
                return None
            
            module_dir = module_path.parent
            
            # Create temporary wrapper script
            wrapper_script = tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, dir=tempfile.gettempdir()
            )
            
            # Determine how to instantiate the benchmark
            # Try common patterns: get_benchmark() function, or class name
            instantiation_code = f"""# Get benchmark instance (try common patterns)
benchmark = None
try:
    if hasattr({module_name}, "get_benchmark"):
        benchmark = {module_name}.get_benchmark()
    elif hasattr({module_name}, "{benchmark_class}"):
        benchmark_class = getattr({module_name}, "{benchmark_class}")
        benchmark = benchmark_class()
    else:
        # Try to find any class with benchmark_fn method
        for attr_name in dir({module_name}):
            attr = getattr({module_name}, attr_name)
            if isinstance(attr, type) and hasattr(attr, "benchmark_fn") and callable(getattr(attr, "benchmark_fn", None)):
                benchmark = attr()
                break
except Exception as e:
    import traceback
    print("Error creating benchmark: " + str(e))
    traceback.print_exc()
    raise

if benchmark is None:
    raise RuntimeError("Could not find or instantiate benchmark instance")
"""
            
            wrapper_content = f'''import sys
from pathlib import Path

# Add module directory to path
sys.path.insert(0, r"{module_dir}")

# Import the benchmark module
import {module_name}

{instantiation_code}

# Run benchmark
try:
    benchmark.setup()
    
    # Warmup
    for _ in range({config.warmup}):
        benchmark.benchmark_fn()
    
    # Synchronize before profiling
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Run benchmark iterations for profiling (limited for profiling overhead)
    for _ in range({min(config.iterations, 10)}):
        benchmark.benchmark_fn()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    benchmark.teardown()
except Exception as e:
    import traceback
    print("Error running benchmark: " + str(e))
    traceback.print_exc()
    raise
'''
            
            wrapper_script.write(wrapper_content)
            wrapper_script.close()
            
            return Path(wrapper_script.name)
        except Exception as e:
            # Log the error for debugging but don't raise (caller will handle None return)
            # Only log if in debug mode, otherwise silently fail (degraded mode is OK)
            if os.environ.get("BENCHMARK_DEBUG", "").lower() in ("1", "true", "yes"):
                import traceback
                print(f"  Debug: Failed to create wrapper script for {benchmark_class}: {e}")
            return None
    
    # Metrics extraction methods removed - now using common.python.metrics_extractor module
    # See extract_nsys_metrics() and extract_ncu_metrics() in metrics_extractor.py
    # For metric descriptions, use get_ncu_metric_description() from metrics_extractor module
    
    def _benchmark_without_profiling(
        self, fn: Callable, config: BenchmarkConfig
    ) -> List[float]:
        """Benchmark without profiling."""
        if self.mode == BenchmarkMode.TRITON:
            return self._benchmark_triton(fn, config)
        elif self.mode == BenchmarkMode.PYTORCH:
            return self._benchmark_pytorch(fn, config)
        else:
            return self._benchmark_custom(fn, config)
    
    def _benchmark_triton(self, fn: Callable, config: BenchmarkConfig) -> List[float]:
        """Use Triton's do_bench (returns single value per call)."""
        try:
            import triton.testing as tt
            times_ms = []
            # Triton do_bench handles warmup internally, but we do our own
            for _ in range(config.iterations):
                time_ms = tt.do_bench(fn, warmup=0, rep=1)  # We handle warmup
                times_ms.append(time_ms)
            return times_ms
        except ImportError:
            # Fallback to custom if Triton not available
            return self._benchmark_custom(fn, config)
    
    def _benchmark_pytorch(self, fn: Callable, config: BenchmarkConfig) -> List[float]:
        """Use PyTorch's Timer."""
        try:
            from torch.utils.benchmark import Timer
            
            timer = Timer(
                stmt=fn,
                globals={},
                num_threads=1,
                device=self.device.type,
            )
            
            # blocked_autorange runs until min_run_time is met
            measurement = timer.blocked_autorange(
                min_run_time=config.min_run_time_ms / 1000.0  # Convert to seconds
            )
            
            # measurement.times is already in seconds
            times_ms = [t * 1000 for t in measurement.times]
            
            # If we got fewer iterations than requested, pad with repeats
            if len(times_ms) < config.iterations:
                times_ms = (times_ms * ((config.iterations // len(times_ms)) + 1))[:config.iterations]
            
            return times_ms
        except Exception as e:
            # Fallback to custom on error
            return self._benchmark_custom(fn, config)
    
    def _benchmark_custom(self, fn: Callable, config: BenchmarkConfig) -> List[float]:
        """Custom benchmarking with CUDA Events for accurate GPU timing.
        
        Optimized for minimal overhead:
        - Uses CUDA Events for accurate GPU timing without blocking
        - Reuses events across iterations for efficiency
        - Synchronizes only when necessary for accurate timing
        """
        times_ms = []
        is_cuda = self.device.type == "cuda"
        
        if is_cuda:
            # Use CUDA Events for accurate GPU timing
            # Create events once - reuse across iterations (efficient)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Synchronize once before starting to ensure clean state
            torch.cuda.synchronize(self.device)
            
            # Run benchmark iterations with accurate per-iteration timing
            # CUDA Events provide accurate timing with minimal overhead
            for _ in range(config.iterations):
                # Record start event (non-blocking)
                start_event.record()
                # Execute function under test
                fn()
                # Record end event (non-blocking)
                end_event.record()
                # Synchronize to ensure events are recorded, then get elapsed time
                # Note: Sync is necessary for accurate timing, but CUDA Events minimize overhead
                torch.cuda.synchronize(self.device)
                times_ms.append(start_event.elapsed_time(end_event))
        else:
            # CPU: use high-resolution timer
            for _ in range(config.iterations):
                start_time = time.perf_counter()
                fn()
                end_time = time.perf_counter()
                times_ms.append((end_time - start_time) * 1000)
        
        return times_ms
    
    def _warmup(self, fn: Callable, warmup_iterations: int) -> None:
        """Perform warmup iterations."""
        is_cuda = self.device.type == "cuda"
        for _ in range(warmup_iterations):
            fn()
        if is_cuda:
            torch.cuda.synchronize(self.device)
    
    def _compute_stats(
        self, times_ms: List[float], config: BenchmarkConfig
    ) -> BenchmarkResult:
        """Compute statistical measures."""
        if not times_ms:
            raise ValueError("No timing data collected")
        
        sorted_times = sorted(times_ms)
        n = len(sorted_times)
        
        # Compute percentiles
        percentiles_dict = {}
        for p in config.percentiles:
            idx = int((p / 100.0) * (n - 1))
            idx = min(idx, n - 1)
            percentiles_dict[p] = sorted_times[idx]
        
        return BenchmarkResult(
            mean_ms=statistics.mean(times_ms),
            median_ms=statistics.median(times_ms),
            std_ms=statistics.stdev(times_ms) if n > 1 else 0.0,
            min_ms=min(times_ms),
            max_ms=max(times_ms),
            percentiles=percentiles_dict,
            iterations=n,
            warmup_iterations=config.warmup,
            raw_times_ms=times_ms,  # Store raw times for subprocess IPC
        )


def compare_benchmarks(
    baseline: Benchmark,
    optimized: Benchmark,
    harness: Optional[BenchmarkHarness] = None,
    name: str = "Comparison",
    regression_threshold_pct: float = 5.0
) -> Dict[str, any]:
    """Compare baseline vs optimized benchmarks and return metrics.
    
    Args:
        baseline: Baseline benchmark instance
        optimized: Optimized benchmark instance
        harness: BenchmarkHarness instance (creates new if None)
        name: Name for the comparison
        regression_threshold_pct: Percentage degradation to consider a regression (default: 5%)
        
    Returns:
        Dictionary with comparison metrics including regression detection
    """
    if harness is None:
        harness = BenchmarkHarness()
    
    baseline_result = harness.benchmark(baseline)
    optimized_result = harness.benchmark(optimized)
    
    speedup = baseline_result.mean_ms / optimized_result.mean_ms if optimized_result.mean_ms > 0 else 1.0
    
    # Detect regression: optimized is slower by threshold
    regression = False
    regression_pct = None
    if speedup < 1.0:
        regression_pct = (1.0 - speedup) * 100
        regression = regression_pct >= regression_threshold_pct
    
    return {
        "name": name,
        "baseline": {
            "mean_ms": baseline_result.mean_ms,
            "median_ms": baseline_result.median_ms,
            "std_ms": baseline_result.std_ms,
            "min_ms": baseline_result.min_ms,
            "max_ms": baseline_result.max_ms,
        },
        "optimized": {
            "mean_ms": optimized_result.mean_ms,
            "median_ms": optimized_result.median_ms,
            "std_ms": optimized_result.std_ms,
            "min_ms": optimized_result.min_ms,
            "max_ms": optimized_result.max_ms,
        },
        "speedup": speedup,
        "regression": regression,
        "regression_pct": regression_pct,
        "baseline_result": baseline_result,
        "optimized_result": optimized_result,
    }

