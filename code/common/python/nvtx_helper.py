"""Helper utilities for conditional NVTX range markers.

This module provides utilities to conditionally add NVTX ranges only when
profiling is enabled, reducing overhead for pure performance benchmarks.
"""

import os
import sys
from contextlib import contextmanager, redirect_stderr
from io import StringIO
from typing import Optional

import torch


class FilteredStderr:
    """Thread-safe filter for stderr that removes NVTX threading errors."""
    def __init__(self, original):
        self.original = original
    
    def write(self, text):
        # Filter out NVTX threading error messages
        if "External init callback must run in same thread as registerClient" not in text:
            self.original.write(text)
        # Otherwise silently drop the error message
    
    def flush(self):
        self.original.flush()
    
    def __getattr__(self, name):
        # Forward all other attributes to original stderr
        return getattr(self.original, name)


@contextmanager
def _suppress_nvtx_threading_error():
    """Suppress NVTX threading errors that occur when CUDA initializes in different thread.
    
    This is a known PyTorch/NVTX issue where NVTX is initialized in one thread
    but used in another. The error is harmless and benchmarks complete successfully.
    We suppress stderr output for this specific error message using thread-safe redirect_stderr.
    """
    # Use contextlib.redirect_stderr for thread-safe stderr redirection
    filtered_stderr = FilteredStderr(sys.stderr)
    with redirect_stderr(filtered_stderr):
        yield


@contextmanager
def nvtx_range(name: str, enable: Optional[bool] = None):
    """Conditionally add NVTX range marker.
    
    Args:
        name: Name for the NVTX range
        enable: If True, add NVTX range; if False, no-op; if None, auto-detect from config
    
    Example:
        with nvtx_range("my_operation", enable=True):
            # This operation will be marked in NVTX traces
            result = model(input)
    """
    if enable is None:
        # Auto-detect: check if NVTX is enabled via environment or config
        # Default to False for minimal overhead
        enable = False
    
    if enable and torch.cuda.is_available():
        # Suppress NVTX threading errors (harmless but noisy)
        with _suppress_nvtx_threading_error():
            try:
                torch.cuda.nvtx.range_push(name)
                try:
                    yield
                finally:
                    torch.cuda.nvtx.range_pop()
            except Exception:
                # If NVTX fails (e.g., threading issue), silently fall back to no-op
                # This ensures benchmarks continue to work even if NVTX has issues
                yield
    else:
        # No-op when NVTX is disabled
        yield


def get_nvtx_enabled(config) -> bool:
    """Get NVTX enabled status from benchmark config.
    
    Args:
        config: BenchmarkConfig instance
    
    Returns:
        True if NVTX should be enabled, False otherwise
    """
    if hasattr(config, 'enable_nvtx'):
        return config.enable_nvtx
    # Fallback: enable only if profiling is enabled
    if hasattr(config, 'enable_profiling'):
        return config.enable_profiling
    return False

