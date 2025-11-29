"""System diagnostics shown via `aisp system ...`."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from typing import Any, Dict

try:
    import torch
except Exception:  # pragma: no cover - torch may be missing in docs builds
    torch = None  # type: ignore

from core.env import dump_environment_and_capabilities
from core.benchmark.run_manifest import get_gpu_info, get_gpu_state


def _print(payload: Dict[str, Any], json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def system_status(args: Any) -> int:
    """Dump environment + capability snapshot."""
    dump_environment_and_capabilities()
    return 0


def gpu_info(args: Any) -> int:
    """Show basic GPU details."""
    info = get_gpu_info()
    if torch and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["memory_total_gb"] = round(props.total_memory / (1024**3), 2)
        info["name"] = props.name
    state = get_gpu_state()
    payload = {**info, **state}
    _print(payload, getattr(args, "json", False))
    return 0


def show_env(args: Any) -> int:
    """Show common environment variables relevant to CUDA/torch."""
    keys = [
        "CUDA_HOME",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTORCH_CUDA_ALLOC_CONF",
        "NVCCFLAGS",
        "TORCH_LOGS",
    ]
    payload = {k: os.environ.get(k, "") for k in keys}
    payload["cwd"] = os.getcwd()
    _print(payload, getattr(args, "json", False))
    return 0


def check_deps(args: Any) -> int:
    """Print a quick dependency/version snapshot."""
    versions = {
        "python": platform.python_version(),
    }
    if torch:
        versions["torch"] = torch.__version__
        versions["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            versions["cuda_version"] = torch.version.cuda
            versions["device_name"] = torch.cuda.get_device_name(0)
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            result = subprocess.run(
                [nvcc, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            versions["nvcc"] = result.stdout.splitlines()[-1].strip() if result.stdout else "found"
        except Exception as exc:  # pragma: no cover - best-effort probe
            versions["nvcc"] = f"nvcc probe failed: {exc}"
    _print(versions, getattr(args, "json", False))
    return 0


def topo(args: Any) -> int:
    """Print GPU topology matrix via nvidia-smi (best effort)."""
    cmd = shutil.which("nvidia-smi")
    if not cmd:
        print("nvidia-smi not found")
        return 1
    try:
        result = subprocess.run([cmd, "topo", "-m"], capture_output=True, text=True, timeout=5)
    except Exception as exc:  # pragma: no cover - external tool
        print(f"Failed to query topology: {exc}")
        return 1
    if result.returncode != 0:
        print(result.stderr or "nvidia-smi topo -m failed")
        return 1
    print(result.stdout.strip())
    return 0
