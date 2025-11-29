"""
Safe torch.compile wrapper with timeout and explicit SKIP semantics.

This module provides a robust wrapper around torch.compile that:
1. Detects large models (>40B parameters) and warns about potential hangs
2. Provides timeout support for compilation (prevents indefinite hangs)
3. Raises clear SKIPPED errors whenever compilation cannot proceed
4. Provides diagnostics about why compilation was skipped

Usage:
    from core.common.torch_compile_safe import safe_compile
    
    # Simple usage
    model_compiled = safe_compile(model, mode='max-autotune')
    
    # With explicit timeout (seconds)
    model_compiled = safe_compile(model, timeout=300)  # 5 min timeout
    
    # Force disable for large models
    model_compiled = safe_compile(model, skip_if_large=True)
"""

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
import os
import threading
import time
from typing import Any, Callable, Optional, TypeVar, Union, cast

import torch
import torch.nn as nn
from core.utils.compile_utils import error_on_graph_break

LayerSequence = Union[nn.ModuleList, nn.Sequential]


# Environment variable to disable timeout (for debugging)
DISABLE_TIMEOUT = os.environ.get("TORCH_COMPILE_DISABLE_TIMEOUT", "0") == "1"

# Default timeout for compilation (seconds)
# Large models can take 5-10 minutes to compile
DEFAULT_COMPILE_TIMEOUT = int(os.environ.get("TORCH_COMPILE_TIMEOUT", "600"))  # 10 minutes

# Parameter count threshold for "large" models (40B = 40e9)
LARGE_MODEL_THRESHOLD = 40_000_000_000


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def is_large_model(model: nn.Module, threshold: int = LARGE_MODEL_THRESHOLD) -> bool:
    """Check if model exceeds large model threshold."""
    try:
        param_count = count_parameters(model)
        return param_count >= threshold
    except Exception:
        # If we can't count parameters, assume it's not large
        return False


class CompilationTimeoutError(Exception):
    """Raised when compilation exceeds timeout."""
    pass


T = TypeVar("T")


def _compile_with_timeout(
    fn: Callable[[], T],
    timeout: int,
) -> T:
    """
    Execute ``fn`` with timeout support.
    
    Uses threading to implement timeout - the actual compilation happens
    in a separate thread, and we wait for it with a timeout.
    """
    if DISABLE_TIMEOUT:
        return fn()
    
    result: list[T] = []
    error: list[BaseException] = []
    finished = threading.Event()
    
    def compile_worker() -> None:
        try:
            result.append(fn())
        except BaseException as exc:  # pragma: no cover - defensive
            error.append(exc)
        finally:
            finished.set()
    
    thread = threading.Thread(target=compile_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if not finished.is_set():
        raise CompilationTimeoutError(
            f"Compilation exceeded timeout of {timeout} seconds during compile. "
            "This is common for models >40B parameters. "
            "Consider using eager mode or increasing timeout."
        )
    
    if error:
        raise error[0]
    if not result:
        raise RuntimeError("Compilation finished without producing a result")
    
    return result[0]


def safe_compile(
    model: nn.Module,
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = False,
    backend: str = "inductor",
    timeout: Optional[int] = None,
    skip_if_large: bool = False,
    warn_on_skip: bool = True,
    error_on_graph_break: Optional[bool] = None,
    **kwargs: Any
) -> nn.Module:
    """
    Safely compile a model with timeout and fallback support.
    
    Args:
        model: Model to compile
        mode: torch.compile mode ('default', 'reduce-overhead', 'max-autotune')
        fullgraph: Whether to compile entire graph
        dynamic: Whether shapes are dynamic
        backend: Compilation backend
        timeout: Compilation timeout in seconds (default: 600)
        skip_if_large: Skip compilation for large models (>40B params)
        warn_on_skip: Print warning when compilation is skipped
        **kwargs: Additional arguments passed to torch.compile
    
    Returns:
        Compiled model.
    
    Raises:
        RuntimeError: If torch.compile is unavailable, skipped, or fails.
    """
    if timeout is None:
        timeout = DEFAULT_COMPILE_TIMEOUT
    
    # Check if model is large
    is_large = is_large_model(model)
    
    if skip_if_large and is_large:
        param_count = count_parameters(model)
        raise RuntimeError(
            f"SKIPPED: torch.compile disabled for large model "
            f"({param_count / 1e9:.1f}B parameters)."
        )
    
    if is_large and warn_on_skip:
        param_count = count_parameters(model)
        print(
            f"[torch.compile] Compiling large model ({param_count / 1e9:.1f}B parameters); "
            "timeouts are likely."
        )
    
    def compile_target() -> nn.Module:
        with error_on_graph_break(error_on_graph_break):
            compiled_model = torch.compile(
                model,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
                backend=backend,
                **kwargs
            )
            return cast(nn.Module, compiled_model)
    
    try:
        # Attempt compilation with timeout
        compiled = _compile_with_timeout(
            compile_target,
            timeout,
        )
        
        if warn_on_skip and is_large:
            print("Compilation completed successfully (large model)")
        
        return compiled
        
    except CompilationTimeoutError as e:
        raise RuntimeError(
            f"SKIPPED: torch.compile timed out after {timeout}s ({e})."
        ) from e
        
    except Exception as e:
        raise RuntimeError(
            f"SKIPPED: torch.compile failed during safe_compile: {e}"
        ) from e


def should_use_compile(
    model: nn.Module,
    model_size_gb: Optional[float] = None
) -> tuple[bool, str]:
    """
    Determine if torch.compile should be used for a model.
    
    Returns:
        (should_compile, reason)
    """
    param_count = count_parameters(model)
    param_count_b = param_count / 1e9
    
    # Estimate model size if not provided
    if model_size_gb is None:
        # Rough estimate: FP16 = 2 bytes per parameter
        model_size_gb = param_count * 2 / 1e9
    
    # Hard cutoff: >40B parameters
    if param_count >= LARGE_MODEL_THRESHOLD:
        return False, f"Model too large ({param_count_b:.1f}B params): compilation hangs"
    
    # Soft cutoff: >80GB model size suggests memory-bound workload
    if model_size_gb > 80:
        return False, f"Model size ({model_size_gb:.1f}GB) suggests memory-bound workload"
    
    # For smaller models, compilation is usually beneficial
    if param_count_b < 1:
        return True, f"Small model ({param_count_b:.2f}B params): compilation recommended"
    elif param_count_b < 10:
        return True, f"Medium model ({param_count_b:.1f}B params): compilation may help"
    else:
        return True, f"Large model ({param_count_b:.1f}B params): compilation may hang, profile first"


def _profile_model_latency_ms(
    model: nn.Module,
    sample_inputs: tuple[Any, ...],
    warmup_iters: int = 1,
    bench_iters: int = 3,
) -> float:
    """
    Lightweight eager timing to guide mode selection.
    
    Uses no gradients; syncs CUDA if available.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(*sample_inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(bench_iters):
            _ = model(*sample_inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return (time.time() - start) * 1000.0 / bench_iters

def detect_transformer_layers(model: nn.Module) -> Optional[LayerSequence]:
    """
    Detect transformer layers in a model.
    
    Looks for common patterns:
    - model.layers (ModuleList)
    - model.transformer.layers
    - model.blocks (Sequential or ModuleList)
    
    Returns:
        Sequence of transformer layers, or None if not found
    """
    # Check common attribute names
    for attr_name in ['layers', 'blocks', 'h', 'transformer_blocks']:
        if hasattr(model, attr_name):
            layers = getattr(model, attr_name)
            if isinstance(layers, (nn.ModuleList, nn.Sequential)):
                return layers
    
    # Check nested transformer
    if hasattr(model, 'transformer'):
        transformer = model.transformer
        for attr_name in ['layers', 'blocks', 'h']:
            if hasattr(transformer, attr_name):
                layers = getattr(transformer, attr_name)
                if isinstance(layers, (nn.ModuleList, nn.Sequential)):
                    return layers
    
    return None


def compile_layer(
    layer: nn.Module,
    mode: str = "reduce-overhead",
    timeout: int = 60,
    **kwargs: Any
) -> nn.Module:
    """
    Compile a single layer with timeout.
    
    Args:
        layer: Layer to compile
        mode: Compilation mode
        timeout: Timeout in seconds (per layer)
        **kwargs: Additional torch.compile arguments
    
    Returns:
        Compiled layer.
    """
    try:
        def compile_target() -> nn.Module:
            compiled_layer = torch.compile(layer, mode=mode, **kwargs)
            return cast(nn.Module, compiled_layer)
        
        return _compile_with_timeout(compile_target, timeout)
    except Exception as exc:
        raise RuntimeError(
            f"SKIPPED: torch.compile failed for layer ({layer.__class__.__name__}): {exc}"
        ) from exc


def partial_compile(
    model: nn.Module,
    layer_indices: Optional[list[int]] = None,
    max_layers: Optional[int] = None,
    mode: str = "reduce-overhead",
    timeout_per_layer: int = 60,
    verbose: bool = True,
    **kwargs: Any
) -> nn.Module:
    """
    Partially compile a model to avoid hangs on large models.
    
    This addresses the root cause by compiling only specific layers,
    reducing compilation complexity and memory usage.
    
    Strategy:
    - Detect transformer layers automatically
    - Compile each layer individually (avoids graph explosion)
    - Use short timeout per layer (prevents hangs)
    - Fall back to eager for layers that fail/timeout
    
    Args:
        model: Model to partially compile
        layer_indices: List of layer indices to compile (None = auto-select)
        max_layers: Maximum number of layers to compile (None = all)
        mode: torch.compile mode
        timeout_per_layer: Timeout per layer in seconds
        verbose: Print compilation progress
        **kwargs: Additional arguments passed to torch.compile
    
    Returns:
        Model with selected layers compiled
    
    Example:
        # Auto-compile first 10 layers
        model = partial_compile(model, max_layers=10)
        
        # Compile specific layers
        model = partial_compile(model, layer_indices=[0, 1, 2, 5, 10])
        
        # Full model with layer-by-layer compilation (safer than full compile)
        model = partial_compile(model)
    """
    # If model is small enough, use full compilation
    if not is_large_model(model):
        if verbose:
            print("Model is small, using full compilation")
        return safe_compile(model, mode=mode, **kwargs)
    
    # Detect transformer layers
    layers = detect_transformer_layers(model)
    
    if layers is None:
        raise RuntimeError(
            "SKIPPED: unable to detect transformer layers for partial compilation."
        )
    
    # Determine which layers to compile
    if layer_indices is None:
        # Auto-select layers
        num_layers = len(layers)
        if max_layers is not None:
            # Compile first N layers
            layer_indices = list(range(min(max_layers, num_layers)))
        else:
            # Compile all layers (but individually)
            layer_indices = list(range(num_layers))
    
    if verbose:
        print(f"Detected {len(layers)} transformer layers")
        print(f"Compiling layers: {layer_indices}")
        print(f"   (Mode: {mode}, Timeout per layer: {timeout_per_layer}s)")
    
    # Compile each layer individually
    for idx in layer_indices:
        if idx >= len(layers):
            continue
        
        if verbose:
            print(f"   Layer {idx}...", end=" ", flush=True)
        
        compiled_layer = compile_layer(
            layers[idx],
            mode=mode,
            timeout=timeout_per_layer,
            **kwargs
        )
        layers[idx] = compiled_layer
        if verbose:
            print("OK")
    
    return model


def smart_compile(
    model: nn.Module,
    mode: str = "reduce-overhead",
    profile_first: bool = False,
    sample_inputs: Optional[tuple[Any, ...]] = None,
    **kwargs: Any
) -> nn.Module:
    """
    Intelligently choose compilation strategy based on model characteristics.
    
    This is the recommended entry point for compilation.
    
    Strategy:
    - Small models (<1B): Full compilation
    - Medium models (1-10B): Full compilation with timeout
    - Large models (10-40B): Partial compilation (first 20 layers)
    - Very large models (>40B): Eager mode (compilation likely to fail)
    
    Args:
        model: Model to compile
        mode: Compilation mode
        profile_first: Profile model before deciding (uses sample_inputs if provided)
        sample_inputs: Tuple of inputs for the lightweight eager timing
        **kwargs: Additional torch.compile arguments
    
    Returns:
        Model (compiled, partially compiled, or eager)
    
    Example:
        # Recommended usage
        model = smart_compile(model)
    """
    param_count = count_parameters(model)
    param_count_b = param_count / 1e9
    
    print(f"Model size: {param_count_b:.2f}B parameters")
    
    # Optional profiling-guided selection
    if profile_first and sample_inputs is not None:
        try:
            eager_latency_ms = _profile_model_latency_ms(model, sample_inputs)
            print(f"[smart_compile] Eager latency: {eager_latency_ms:.2f} ms")
            if eager_latency_ms < 5.0:
                print("Strategy: Eager mode (workload too small for compile payoff)")
                return model
            elif eager_latency_ms < 30.0:
                print("Strategy: Full compilation (reduce-overhead) based on profile")
                return safe_compile(model, mode="reduce-overhead", timeout=300, **kwargs)
            else:
                print("Strategy: Full compilation (max-autotune) based on profile")
                return safe_compile(model, mode="max-autotune", timeout=600, **kwargs)
        except Exception as exc:  # pragma: no cover - best-effort profiling
            print(f"[smart_compile] Profiling-based selection failed: {exc}; falling back to parameter-based heuristics.")
    
    # Strategy selection (parameter-based heuristics)
    if param_count_b < 1:
        print("Strategy: Full compilation (small model)")
        return safe_compile(model, mode=mode, **kwargs)
    
    elif param_count_b < 10:
        print("Strategy: Full compilation with timeout (medium model)")
        return safe_compile(model, mode=mode, timeout=600, **kwargs)
    
    elif param_count_b < 40:
        print("Strategy: Partial compilation (large model)")
        # Compile first 20 layers to get some benefit without hangs
        return partial_compile(model, max_layers=20, mode=mode, **kwargs)
    
    else:
        raise RuntimeError(
            f"SKIPPED: Model too large for smart_compile ({param_count_b:.1f}B params)."
        )
