"""Profiler wrapper script generation.

Generates wrapper scripts for nsys/ncu profiling that import and run benchmarks.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Optional, Protocol

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


def create_benchmark_wrapper(
    benchmark: BaseBenchmark,
    benchmark_module,
    benchmark_class: str,
    config: BenchmarkConfig
) -> Optional[Path]:
    """Create a temporary Python script that runs the benchmark.
    
    The wrapper script imports the benchmark module and recreates the benchmark
    instance, then runs setup, warmup, and profiling iterations.
    
    Args:
        benchmark: BaseBenchmark instance (used to get module info)
        benchmark_module: Module object containing the benchmark
        benchmark_class: Name of the benchmark class
        config: BenchmarkConfig with iterations/warmup settings
    
    Returns:
        Path to created wrapper script, or None if creation failed
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
            return None
        
        module_path = Path(module_file).resolve()
        if not module_path.exists():
            return None
        
        module_dir = module_path.parent
        
        # Create temporary wrapper script
        wrapper_script = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, dir=tempfile.gettempdir()
        )
        
        # Determine how to instantiate the benchmark
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
    except Exception:
        return None


def create_cuda_wrapper(
    cuda_executable: str,
    args: list[str],
    config: Optional[BenchmarkConfig] = None
) -> Optional[Path]:
    """Create a wrapper script for CUDA executables.
    
    Args:
        cuda_executable: Path to CUDA executable
        args: Command-line arguments for the executable
        config: Optional BenchmarkConfig (currently unused, for future extensibility)
    
    Returns:
        Path to created wrapper script, or None if creation failed
    """
    try:
        wrapper_script = tempfile.NamedTemporaryFile(
            mode='w', suffix='.sh', delete=False, dir=tempfile.gettempdir()
        )
        
        wrapper_content = f'''#!/bin/bash
# Wrapper for CUDA executable profiling

exec "{cuda_executable}" {" ".join(args)}
'''
        
        wrapper_script.write(wrapper_content)
        wrapper_script.close()
        
        # Make executable
        wrapper_path = Path(wrapper_script.name)
        wrapper_path.chmod(0o755)
        
        return wrapper_path
    except Exception:
        return None
