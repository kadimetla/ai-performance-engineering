"""Helpers for applying environment defaults and reporting capabilities."""

from __future__ import annotations

from common.python import compile_utils as _compile_utils_patch  # noqa: F401
from common.python.cuda_capabilities import (
    pipeline_support_status,
    pipeline_runtime_allowed,
    tma_support_status,
)
from common.python.hardware_capabilities import (
    detect_capabilities,
    format_capability_report,
)
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from common.python.build_utils import ensure_clean_build_directory
except ImportError:  # pragma: no cover - psutil optional in some environments
    ensure_clean_build_directory = None  # type: ignore[assignment]

ENV_DEFAULTS: Dict[str, str] = {
    "PYTHONFAULTHANDLER": "1",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
    "TORCH_CUDNN_V8_API_ENABLED": "1",
    "CUDA_LAUNCH_BLOCKING": "0",
    "CUDA_CACHE_DISABLE": "0",
    "NCCL_IB_DISABLE": "0",
    "NCCL_P2P_DISABLE": "0",
    "NCCL_SHM_DISABLE": "0",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    "TORCH_COMPILE_DEBUG": "0",
    # "TORCH_LOGS": "",  # Disabled - remove verbose dynamo logging to reduce noise
    "CUDA_HOME": "/usr/local/cuda-13.0",
}

CUDA_PATH_SUFFIXES: Tuple[str, ...] = ("bin",)
CUDA_LIBRARY_SUFFIXES: Tuple[str, ...] = ("lib64",)

# Try to find NCCL library for current architecture
def _find_nccl_library() -> str:
    """Find NCCL library for the current architecture."""
    import platform
    machine = platform.machine()
    
    # Try architecture-specific paths
    candidates = []
    if machine == "x86_64":
        candidates = [
            "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
            "/usr/lib/x86_64-linux-gnu/libnccl.so",
        ]
    elif machine in ("aarch64", "arm64"):
        candidates = [
            "/usr/lib/aarch64-linux-gnu/libnccl.so.2",
            "/usr/lib/aarch64-linux-gnu/libnccl.so",
        ]
    
    # Also try generic paths
    candidates.extend([
        "/usr/local/lib/libnccl.so.2",
        "/usr/local/lib/libnccl.so",
        "/usr/lib/libnccl.so.2",
        "/usr/lib/libnccl.so",
    ])
    
    # Return first existing file, or default to x86_64 path (will be ignored if not found)
    for path in candidates:
        if os.path.exists(path):
            return path
    
    # Return empty string if not found (will be skipped when adding to LD_PRELOAD)
    return ""

NCCL_LIBRARY_PATH = _find_nccl_library()

REPORTED_ENV_KEYS: Tuple[str, ...] = (
    "PYTHONFAULTHANDLER",
    "TORCH_SHOW_CPP_STACKTRACES",
    "TORCH_CUDNN_V8_API_ENABLED",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_CACHE_DISABLE",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "PYTORCH_ALLOC_CONF",
    "TORCH_COMPILE_DEBUG",
    "TORCH_LOGS",
    "CUDA_HOME",
    "PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
)

_ENV_AND_CAPABILITIES_LOGGED = False


def apply_env_defaults() -> Dict[str, str]:
    """Apply default environment configuration and return the resulting values.
    
    Only sets variables that are not already set, to avoid overwriting user configurations.
    """
    applied: Dict[str, str] = {}

    for key, value in ENV_DEFAULTS.items():
        previous = os.environ.get(key)
        if previous is None:
            os.environ.setdefault(key, value)
        applied[key] = os.environ[key]

    if "PYTORCH_ALLOC_CONF" not in os.environ:
        legacy = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ["PYTORCH_ALLOC_CONF"] = legacy or "max_split_size_mb:128,expandable_segments:True"
    applied["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_ALLOC_CONF"]
    
    # Ensure PyTorch inductor cache directory exists to prevent C++ compilation errors
    # PyTorch inductor needs this directory to exist for C++ code generation
    # Use absolute path to avoid working directory issues
    inductor_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not inductor_cache_dir:
        # Default to .torch_inductor in current working directory (convert to absolute)
        inductor_cache_dir = str(Path.cwd() / ".torch_inductor")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_dir
    else:
        # Convert relative paths to absolute paths to avoid working directory issues
        if not os.path.isabs(inductor_cache_dir):
            inductor_cache_dir = str(Path.cwd() / inductor_cache_dir)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_dir
    
    if inductor_cache_dir:
        inductor_cache_path = Path(inductor_cache_dir)
        # Create directory and subdirectories (used by inductor for C++ compilation)
        # 'od' is for output directory, 'tk' is for temporary kernel files
        try:
            inductor_cache_path.mkdir(parents=True, exist_ok=True)
            (inductor_cache_path / "od").mkdir(parents=True, exist_ok=True)
            (inductor_cache_path / "tk").mkdir(parents=True, exist_ok=True)
            if ensure_clean_build_directory is not None:
                # Kill stale compiler processes/locks that keep torch.compile hanging
                ensure_clean_build_directory(inductor_cache_path)
        except (OSError, PermissionError):
            # If we can't create the directory, that's okay - PyTorch will handle it
            # or fail with a clearer error message
            pass

    # Ensure cpp-extension builds use a workspace-local cache to avoid /tmp lockups
    torch_extensions_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not torch_extensions_dir:
        torch_extensions_dir = str(Path.cwd() / ".torch_extensions")
        os.environ["TORCH_EXTENSIONS_DIR"] = torch_extensions_dir
    elif not os.path.isabs(torch_extensions_dir):
        torch_extensions_dir = str(Path.cwd() / torch_extensions_dir)
        os.environ["TORCH_EXTENSIONS_DIR"] = torch_extensions_dir

    try:
        torch_extensions_path = Path(torch_extensions_dir)
        torch_extensions_path.mkdir(parents=True, exist_ok=True)
        if ensure_clean_build_directory is not None:
            ensure_clean_build_directory(torch_extensions_path)
    except (OSError, PermissionError):
        pass

    # Only ensure CUDA paths if CUDA_HOME was not already set by user
    # This prevents overwriting user-configured CUDA installations
    if "CUDA_HOME" not in os.environ or os.environ["CUDA_HOME"] == ENV_DEFAULTS.get("CUDA_HOME"):
        _ensure_cuda_paths()
    else:
        # CUDA_HOME is set by user - only prepend paths if they're missing (don't force our defaults)
        _ensure_cuda_paths(use_existing_cuda_home=True)
    applied["PATH"] = os.environ.get("PATH", "")
    applied["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")

    _ensure_ld_preload()
    applied["LD_PRELOAD"] = os.environ.get("LD_PRELOAD", "")

    return {key: os.environ.get(key, "") for key in REPORTED_ENV_KEYS}


def _ensure_cuda_paths(use_existing_cuda_home: bool = False) -> None:
    """Ensure CUDA paths are present in PATH and LD_LIBRARY_PATH.
    
    Args:
        use_existing_cuda_home: If True, use existing CUDA_HOME value even if it matches defaults.
                                If False, only add paths if CUDA_HOME was set from defaults.
    """
    # Only use default CUDA_HOME if it's not already set
    if use_existing_cuda_home:
        cuda_home = os.environ.get("CUDA_HOME")
        if not cuda_home:
            # No CUDA_HOME set and we're not using defaults - skip
            return
    else:
        # Use default only if not set
        cuda_home = os.environ.get("CUDA_HOME", ENV_DEFAULTS.get("CUDA_HOME", ""))
        if not cuda_home:
            return

    # Only add paths if the CUDA_HOME directory actually exists
    if not os.path.exists(cuda_home):
        return

    path_prefixes = _build_paths(cuda_home, CUDA_PATH_SUFFIXES)
    lib_prefixes = _build_paths(cuda_home, CUDA_LIBRARY_SUFFIXES)

    # Only prepend if missing (this function already checks)
    for prefix in path_prefixes:
        if os.path.exists(prefix):  # Only add if path exists
            _prepend_if_missing("PATH", prefix)
    for prefix in lib_prefixes:
        if os.path.exists(prefix):  # Only add if path exists
            _prepend_if_missing("LD_LIBRARY_PATH", prefix)


def _build_paths(root: str, suffixes: Iterable[str]) -> List[str]:
    return [os.path.join(root, suffix) for suffix in suffixes]


def _prepend_if_missing(key: str, prefix: str) -> None:
    os.environ.setdefault(key, "")
    existing = os.environ.get(key, "")
    components = [segment for segment in existing.split(os.pathsep) if segment]
    if prefix not in components:
        components.insert(0, prefix)
        os.environ[key] = os.pathsep.join(components)


def _ensure_ld_preload() -> None:
    os.environ.setdefault("LD_PRELOAD", "")
    preload_entries = [segment for segment in os.environ["LD_PRELOAD"].split(os.pathsep) if segment]
    
    # Only add NCCL library if it exists and isn't already in LD_PRELOAD
    if NCCL_LIBRARY_PATH and os.path.exists(NCCL_LIBRARY_PATH):
        if NCCL_LIBRARY_PATH not in preload_entries:
            preload_entries.insert(0, NCCL_LIBRARY_PATH)
            os.environ["LD_PRELOAD"] = os.pathsep.join(preload_entries)
    elif NCCL_LIBRARY_PATH:
        # Library path was set but doesn't exist - this is okay, just skip it
        pass


def snapshot_environment() -> Dict[str, str]:
    """Return a snapshot of the relevant environment variables."""
    return {key: os.environ.get(key, "") for key in REPORTED_ENV_KEYS}


def dump_environment_and_capabilities(stream=None, *, force: bool = False) -> None:
    """Emit environment configuration and hardware capabilities."""
    global _ENV_AND_CAPABILITIES_LOGGED
    if _ENV_AND_CAPABILITIES_LOGGED and not force:
        return

    if stream is None:
        stream = sys.stdout

    env_snapshot = snapshot_environment()
    print("=" * 80, file=stream)
    print("ENVIRONMENT CONFIGURATION", file=stream)
    print("=" * 80, file=stream)
    for key in REPORTED_ENV_KEYS:
        print(f"{key}={env_snapshot.get(key, '')}", file=stream)

    print("\n" + "=" * 80, file=stream)
    print("HARDWARE CAPABILITIES", file=stream)
    print("=" * 80, file=stream)

    cap_report = format_capability_report()
    print(cap_report, file=stream)
    
    pipeline_supported, pipeline_reason = pipeline_support_status()
    tma_supported, tma_reason = tma_support_status()
    print(f"CUDA Pipeline API Support: {'yes' if pipeline_supported else 'no'} ({pipeline_reason})", file=stream)
    runtime_allowed, runtime_reason = pipeline_runtime_allowed()
    print(f"Pipeline Runtime Enabled: {'yes' if runtime_allowed else 'no'} ({runtime_reason})", file=stream)
    print(f"TMA Support: {'yes' if tma_supported else 'no'} ({tma_reason})", file=stream)
    
    # Check profiling tool availability
    print("\n" + "=" * 80, file=stream)
    print("PROFILING TOOLS", file=stream)
    print("=" * 80, file=stream)
    
    # Check nsys availability
    nsys_available = False
    try:
        import subprocess
        import shutil
        if shutil.which("nsys"):
            result = subprocess.run(
                ["nsys", "--version"],
                capture_output=True,
                timeout=5,
                check=False
            )
            nsys_available = result.returncode == 0
    except Exception:
        pass
    print(f"nsys available: {nsys_available}", file=stream)
    
    # Check ncu availability
    ncu_available = False
    try:
        if shutil.which("ncu"):
            result = subprocess.run(
                ["ncu", "--version"],
                capture_output=True,
                timeout=5,
                check=False
            )
            ncu_available = result.returncode == 0
    except Exception:
        pass
    print(f"ncu available: {ncu_available}", file=stream)
    
    _ENV_AND_CAPABILITIES_LOGGED = True
