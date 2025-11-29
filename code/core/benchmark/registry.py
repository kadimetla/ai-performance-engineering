"""Benchmark registration and validation utilities.

Provides enforcement mechanisms to ensure all benchmark modules implement
the required get_benchmark() factory function.

Usage:
    1. Auto-export decorator (RECOMMENDED - handles deep hierarchies):
       ```python
       from core.benchmark.registry import export_benchmark
       
       @export_benchmark
       class MyBenchmark(BaseBenchmark):
           def setup(self) -> None: ...
           def benchmark_fn(self) -> None: ...
       
       # get_benchmark() is automatically created!
       ```

    2. Import check at module level:
       ```python
       from core.benchmark.registry import require_get_benchmark
       require_get_benchmark(__name__)  # At end of file
       ```

    3. Manual registration with deep hierarchies:
       ```python
       from core.benchmark.registry import register_benchmark
       
       class MyBaseBenchmark(BaseBenchmark):
           '''Intermediate base class for a category.'''
           def common_setup(self): ...
       
       @register_benchmark  
       class ConcreteBenchmark(MyBaseBenchmark):
           def setup(self) -> None: ...
           def benchmark_fn(self) -> None: ...
       
       def get_benchmark() -> BaseBenchmark:
           return ConcreteBenchmark()
       ```
    
    4. Validation script (run in CI):
       ```bash
       python -m core.benchmark.registry --validate
       ```
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

# Track registered benchmarks
_REGISTERED_BENCHMARKS: Dict[str, Type] = {}
_VALIDATED_MODULES: Set[str] = set()
_EXPORTED_BENCHMARKS: Dict[str, Type] = {}  # module_name -> benchmark class

T = TypeVar("T")


class BenchmarkRegistrationError(Exception):
    """Raised when a benchmark module doesn't implement get_benchmark()."""
    pass


def export_benchmark(cls: Type[T]) -> Type[T]:
    """Decorator that registers a benchmark class AND auto-creates get_benchmark().
    
    This is the RECOMMENDED way to create benchmark modules. It handles:
    - Deep class hierarchies (IntermediateBase -> ConcreteBenchmark)
    - Automatic get_benchmark() injection into the module
    - Registration for validation
    
    Usage:
        from core.benchmark.registry import export_benchmark
        
        @export_benchmark
        class MyBenchmark(BaseBenchmark):
            def setup(self) -> None:
                ...
            
            def benchmark_fn(self) -> None:
                ...
        
        # No need to write get_benchmark() - it's auto-generated!
        
        if __name__ == "__main__":
            main()
    
    For deep hierarchies:
        class CategoryBaseBenchmark(BaseBenchmark):
            '''Shared setup for a category of benchmarks.'''
            def _common_init(self): ...
        
        @export_benchmark
        class ConcreteBenchmark(CategoryBaseBenchmark):
            def setup(self) -> None:
                self._common_init()
                ...
    
    Args:
        cls: The benchmark class to export
        
    Returns:
        The same class (decorator is transparent)
    """
    module_name = cls.__module__
    class_name = cls.__name__
    key = f"{module_name}.{class_name}"
    
    # Register the class
    _REGISTERED_BENCHMARKS[key] = cls
    _EXPORTED_BENCHMARKS[module_name] = cls
    
    # Inject get_benchmark() into the module
    if module_name in sys.modules:
        module = sys.modules[module_name]
        
        # Create the factory function
        def get_benchmark() -> Any:
            return cls()
        
        # Add docstring
        get_benchmark.__doc__ = f"Factory function for benchmark discovery. Returns {class_name}()."
        
        # Only inject if not already defined
        if not hasattr(module, "get_benchmark"):
            setattr(module, "get_benchmark", get_benchmark)
    
    return cls


def register_benchmark(cls: Type[T]) -> Type[T]:
    """Decorator to register a benchmark class (without auto-creating get_benchmark).
    
    Use this when you want manual control over get_benchmark(), or when you have
    multiple benchmark classes in one module and need to choose which one to export.
    
    For automatic get_benchmark() creation, use @export_benchmark instead.
    
    Usage:
        @register_benchmark
        class MyBenchmark(BaseBenchmark):
            def setup(self) -> None:
                ...
            
            def benchmark_fn(self) -> None:
                ...
        
        # You must still define get_benchmark() manually:
        def get_benchmark() -> BaseBenchmark:
            return MyBenchmark()
    
    Args:
        cls: The benchmark class to register
        
    Returns:
        The same class (decorator is transparent)
    """
    module_name = cls.__module__
    class_name = cls.__name__
    key = f"{module_name}.{class_name}"
    
    _REGISTERED_BENCHMARKS[key] = cls
    
    return cls


def create_benchmark_factory(benchmark_class: Type[T], **default_kwargs: Any) -> Callable[[], T]:
    """Create a get_benchmark() factory function for a benchmark class.
    
    Useful when you need to customize benchmark instantiation or pass default arguments.
    
    Usage:
        class MyBenchmark(BaseBenchmark):
            def __init__(self, batch_size: int = 32):
                super().__init__()
                self.batch_size = batch_size
        
        # Create factory with custom defaults
        get_benchmark = create_benchmark_factory(MyBenchmark, batch_size=64)
    
    Args:
        benchmark_class: The benchmark class to instantiate
        **default_kwargs: Default keyword arguments for the constructor
        
    Returns:
        A factory function that creates benchmark instances
    """
    def factory() -> T:
        return benchmark_class(**default_kwargs)
    
    factory.__doc__ = f"Factory function for benchmark discovery. Returns {benchmark_class.__name__}()."
    return factory


def get_registered_benchmarks() -> Dict[str, Type]:
    """Get all registered benchmark classes.
    
    Returns:
        Dict mapping "module.ClassName" to the class object
    """
    return dict(_REGISTERED_BENCHMARKS)


def get_exported_benchmark(module_name: str) -> Optional[Type]:
    """Get the exported benchmark class for a module.
    
    Args:
        module_name: The module name (e.g., "ch14.optimized_sliding_window")
        
    Returns:
        The benchmark class if exported via @export_benchmark, None otherwise
    """
    return _EXPORTED_BENCHMARKS.get(module_name)


def require_get_benchmark(module_name: str) -> None:
    """Assert that the current module will have get_benchmark() defined.
    
    Call this at the end of your benchmark module (before if __name__ == "__main__")
    to ensure you don't forget to implement get_benchmark().
    
    Usage:
        # At end of your benchmark file:
        from core.benchmark.registry import require_get_benchmark
        
        def get_benchmark() -> BaseBenchmark:
            return MyBenchmark()
        
        require_get_benchmark(__name__)  # Validates get_benchmark exists
        
        if __name__ == "__main__":
            main()
    
    Args:
        module_name: Pass __name__ from your module
        
    Raises:
        BenchmarkRegistrationError: If get_benchmark() is not defined in the module
    """
    if module_name in _VALIDATED_MODULES:
        return
    
    # Get the module
    if module_name in sys.modules:
        module = sys.modules[module_name]
        if not hasattr(module, "get_benchmark"):
            raise BenchmarkRegistrationError(
                f"Module '{module_name}' is missing get_benchmark() function.\n"
                f"Add a factory function like:\n"
                f"    def get_benchmark() -> BaseBenchmark:\n"
                f"        return YourBenchmarkClass()"
            )
        _VALIDATED_MODULES.add(module_name)


def validate_benchmark_module(module_path: Path) -> Tuple[bool, List[str]]:
    """Validate a benchmark module has the required get_benchmark() function.
    
    Args:
        module_path: Path to the Python benchmark file
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors: List[str] = []
    
    if not module_path.exists():
        return False, [f"File not found: {module_path}"]
    
    if not module_path.suffix == ".py":
        return False, [f"Not a Python file: {module_path}"]
    
    # Check for get_benchmark in source (fast check without import)
    source = module_path.read_text()
    if "def get_benchmark" not in source:
        errors.append(f"Missing get_benchmark() function in {module_path.name}")
        return False, errors
    
    # Try to import and validate
    try:
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            errors.append(f"Could not create module spec for {module_path.name}")
            return False, errors
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, "get_benchmark"):
            errors.append(f"get_benchmark() not found after import in {module_path.name}")
            return False, errors
        
        get_benchmark_fn = getattr(module, "get_benchmark")
        if not callable(get_benchmark_fn):
            errors.append(f"get_benchmark is not callable in {module_path.name}")
            return False, errors
        
        # Optional: Try to call it
        try:
            benchmark = get_benchmark_fn()
            if benchmark is None:
                errors.append(f"get_benchmark() returned None in {module_path.name}")
                return False, errors
        except Exception as e:
            errors.append(f"get_benchmark() raised {type(e).__name__}: {e} in {module_path.name}")
            return False, errors
        
    except Exception as e:
        errors.append(f"Import error in {module_path.name}: {type(e).__name__}: {e}")
        return False, errors
    
    return True, []


def validate_all_benchmarks(repo_root: Path, quick: bool = True) -> Tuple[int, int, List[str]]:
    """Validate all benchmark modules in the repository.
    
    Args:
        repo_root: Path to repository root
        quick: If True, only check source for get_benchmark (no imports)
        
    Returns:
        Tuple of (num_valid, num_invalid, list_of_errors)
    """
    errors: List[str] = []
    num_valid = 0
    num_invalid = 0
    
    # Find all baseline_*.py and optimized_*.py files
    benchmark_patterns = ["baseline_*.py", "optimized_*.py"]
    
    for pattern in benchmark_patterns:
        for module_path in repo_root.rglob(pattern):
            # Skip __pycache__ and test files
            if "__pycache__" in str(module_path):
                continue
            if "test_" in module_path.name:
                continue
            
            if quick:
                # Fast check: just look for the function in source
                try:
                    source = module_path.read_text()
                    if "def get_benchmark" not in source:
                        errors.append(f"MISSING get_benchmark(): {module_path.relative_to(repo_root)}")
                        num_invalid += 1
                    else:
                        num_valid += 1
                except Exception as e:
                    errors.append(f"Could not read {module_path}: {e}")
                    num_invalid += 1
            else:
                # Full validation with import
                is_valid, module_errors = validate_benchmark_module(module_path)
                if is_valid:
                    num_valid += 1
                else:
                    num_invalid += 1
                    for err in module_errors:
                        errors.append(f"{module_path.relative_to(repo_root)}: {err}")
    
    return num_valid, num_invalid, errors


def main():
    """CLI for benchmark validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate benchmark modules")
    parser.add_argument("--validate", action="store_true", help="Validate all benchmark modules")
    parser.add_argument("--full", action="store_true", help="Full validation with imports (slower)")
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root path")
    parser.add_argument("--file", type=Path, default=None, help="Validate single file")
    
    args = parser.parse_args()
    
    if args.file:
        is_valid, errors = validate_benchmark_module(args.file)
        if is_valid:
            print(f"✓ {args.file}: Valid")
            sys.exit(0)
        else:
            print(f"✗ {args.file}: Invalid")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
    
    if args.validate:
        repo_root = args.repo_root if args.repo_root else Path(__file__).parents[2]
        
        print(f"Validating benchmarks in {repo_root}")
        print("=" * 60)
        
        num_valid, num_invalid, errors = validate_all_benchmarks(
            repo_root, quick=not args.full
        )
        
        print(f"\nResults:")
        print(f"  ✓ Valid:   {num_valid}")
        print(f"  ✗ Invalid: {num_invalid}")
        
        if errors:
            print(f"\nErrors:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
        else:
            print(f"\nAll benchmark modules have get_benchmark()!")
            sys.exit(0)
    
    parser.print_help()


if __name__ == "__main__":
    main()
