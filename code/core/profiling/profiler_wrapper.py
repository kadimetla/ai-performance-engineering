"""Profiler wrapper script generation.

Generates wrapper scripts for nsys/ncu profiling that import and run benchmarks.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
else:
    BaseBenchmark = Any  # type: ignore[assignment,misc]
    BenchmarkConfig = Any  # type: ignore[assignment,misc]


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

        # Ensure the import root for the module is on sys.path so `import <module_name>`
        # works for both simple modules ("foo") and package-style modules ("pkg.foo").
        # For namespace packages (no __init__.py), the parent directory containing the
        # top-level package must be present on sys.path.
        module_parts = module_name.split(".")
        up_levels = len(module_parts) - 1
        if module_path.name == "__init__.py":
            up_levels = len(module_parts)
        try:
            import_root = module_path.parents[max(up_levels, 0)]
        except Exception:
            import_root = module_dir

        # Repo code root (the directory containing `core/`). Needed because the
        # wrapper runs as a standalone script whose sys.path does not include the
        # current working directory by default.
        code_root = Path(__file__).resolve().parents[2]
        
        # Create temporary wrapper script
        wrapper_script = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, dir=tempfile.gettempdir()
        )
        
        # Determine how to instantiate the benchmark
        nvtx_includes = getattr(config, "nsys_nvtx_include", None)
        enable_profiling = getattr(config, "enable_profiling", False)
        enable_nvtx = getattr(config, "enable_nvtx", None)
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

# Attach profiling config so NVTX ranges are emitted in wrapper runs.
from core.harness.benchmark_harness import BenchmarkConfig, ReadOnlyBenchmarkConfigView
_profiling_config = BenchmarkConfig(
    enable_profiling={enable_profiling!r},
    enable_nvtx={enable_nvtx!r},
    nsys_nvtx_include={nvtx_includes!r},
)
benchmark._config = ReadOnlyBenchmarkConfigView.from_config(_profiling_config)
"""
        
        wrapper_content = f'''import sys
from pathlib import Path

# Add repo code root so `import core` works inside the wrapper process.
sys.path.insert(0, r"{code_root}")

# Add benchmark module import root so dotted imports work.
sys.path.insert(0, r"{import_root}")

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
