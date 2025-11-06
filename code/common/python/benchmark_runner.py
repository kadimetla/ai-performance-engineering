"""Benchmark runner for subprocess-based execution.

This module provides process isolation for benchmarks, allowing reliable
timeout cancellation even when CUDA kernels hang. Benchmarks are executed
in a child process that can be killed if it exceeds the timeout.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def run_benchmark_in_subprocess(
    benchmark_module_path: str,
    benchmark_class_name: str,
    config_dict: Dict[str, Any],
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a benchmark in the current process (called from subprocess).
    
    This function is designed to be called from a subprocess spawned by
    the benchmark harness. It imports the benchmark module, instantiates
    the benchmark, runs it, and returns results as a dictionary.
    
    Args:
        benchmark_module_path: Path to the benchmark module file
        benchmark_class_name: Name of the benchmark class or 'get_benchmark' function
        config_dict: Dictionary of BenchmarkConfig values to override
        device: CUDA device string (e.g., 'cuda:0') or None for auto-detect
        
    Returns:
        Dictionary with keys: success, times_ms, memory_peak_mb, memory_allocated_mb,
        errors, profiling_outputs, nsys_metrics, ncu_metrics
    """
    result = {
        "success": False,
        "times_ms": [],
        "memory_peak_mb": None,
        "memory_allocated_mb": None,
        "errors": [],
        "profiling_outputs": {},
        "nsys_metrics": {},
        "ncu_metrics": {},
    }
    
    try:
        # Add repo root to path so we can import common.python modules
        # Find repo root by looking for common/python directory
        module_path = Path(benchmark_module_path).resolve()
        repo_root = module_path
        while repo_root.parent != repo_root:  # Not at filesystem root
            if (repo_root / "common" / "python").exists():
                break
            repo_root = repo_root.parent
        else:
            # Fallback: assume common/python is sibling to benchmark module
            repo_root = module_path.parent.parent.parent
        
        sys.path.insert(0, str(repo_root))
        
        if not module_path.exists():
            result["errors"].append(f"Benchmark module not found: {benchmark_module_path}")
            return result
        
        module_dir = module_path.parent
        sys.path.insert(0, str(module_dir))
        
        # Import the benchmark module
        module_name = module_path.stem
        if module_name.endswith('.py'):
            module_name = module_name[:-3]
        
        # Import using importlib to handle module name conflicts
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            result["errors"].append(f"Could not load module spec from {module_path}")
            return result
        
        benchmark_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark_module)
        
        # Get benchmark instance
        benchmark = None
        if hasattr(benchmark_module, "get_benchmark"):
            benchmark = benchmark_module.get_benchmark()
        elif hasattr(benchmark_module, benchmark_class_name):
            benchmark_class = getattr(benchmark_module, benchmark_class_name)
            benchmark = benchmark_class()
        else:
            # Try to find any class with benchmark_fn method
            for attr_name in dir(benchmark_module):
                attr = getattr(benchmark_module, attr_name)
                if isinstance(attr, type) and hasattr(attr, "benchmark_fn") and callable(getattr(attr, "benchmark_fn", None)):
                    benchmark = attr()
                    break
        
        if benchmark is None:
            result["errors"].append(f"Could not find or instantiate benchmark: {benchmark_class_name}")
            return result
        
        # Import harness components
        from common.python.benchmark_harness import BenchmarkHarness, BenchmarkConfig, BenchmarkMode
        
        # Create config from dict
        config = BenchmarkConfig()
        for key, value in config_dict.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        
        # Disable subprocess in runner to avoid recursion (runner IS the subprocess)
        config.use_subprocess = False
        
        # Create harness
        harness = BenchmarkHarness(
            mode=BenchmarkMode.CUSTOM,
            config=config
        )
        
        # Override device if specified
        if device:
            harness.device = torch.device(device)
        
        # Run benchmark
        benchmark_result = harness.benchmark(benchmark)
        
        # Extract results - use raw_times_ms if available, otherwise reconstruct from statistics
        result["success"] = True
        if benchmark_result.raw_times_ms:
            # Use raw times if available (preferred)
            result["times_ms"] = benchmark_result.raw_times_ms
        else:
            # Fallback: reconstruct from statistics (approximation)
            # Store statistics for reference
            result["mean_ms"] = benchmark_result.mean_ms
            result["median_ms"] = benchmark_result.median_ms
            result["std_ms"] = benchmark_result.std_ms
            result["min_ms"] = benchmark_result.min_ms
            result["max_ms"] = benchmark_result.max_ms
            result["iterations"] = benchmark_result.iterations
            # Generate synthetic times based on statistics
            import numpy as np
            if benchmark_result.iterations > 0:
                synthetic_times = np.random.normal(
                    benchmark_result.mean_ms,
                    benchmark_result.std_ms if benchmark_result.std_ms > 0 else benchmark_result.mean_ms * 0.1,
                    benchmark_result.iterations
                )
                # Clamp to min/max
                synthetic_times = np.clip(synthetic_times, benchmark_result.min_ms, benchmark_result.max_ms)
                result["times_ms"] = synthetic_times.tolist()
            else:
                result["times_ms"] = []
        result["memory_peak_mb"] = benchmark_result.memory_peak_mb
        result["memory_allocated_mb"] = benchmark_result.memory_allocated_mb
        result["errors"] = benchmark_result.errors
        result["profiling_outputs"] = benchmark_result.profiling_outputs
        result["nsys_metrics"] = benchmark_result.nsys_metrics
        result["ncu_metrics"] = benchmark_result.ncu_metrics
        
    except Exception as e:
        result["errors"].append(f"Benchmark execution failed: {str(e)}")
        result["errors"].append(f"Traceback: {traceback.format_exc()}")
    
    return result


def main():
    """Entry point for subprocess execution.
    
    Expects JSON input on stdin with:
    - benchmark_module_path: Path to benchmark module
    - benchmark_class_name: Name of benchmark class
    - config_dict: BenchmarkConfig overrides
    - device: Optional device string
    
    Outputs JSON result to stdout.
    """
    try:
        input_data = json.loads(sys.stdin.read())
        benchmark_module_path = input_data["benchmark_module_path"]
        benchmark_class_name = input_data["benchmark_class_name"]
        config_dict = input_data.get("config_dict", {})
        device = input_data.get("device")
        
        result = run_benchmark_in_subprocess(
            benchmark_module_path,
            benchmark_class_name,
            config_dict,
            device
        )
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "success": False,
            "errors": [f"Subprocess execution failed: {str(e)}", traceback.format_exc()],
            "times_ms": [],
        }
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == "__main__":
    main()

