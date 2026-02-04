"""Helper to build and run standalone CUDA demo binaries."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Optional

from core.benchmark.cuda_binary_benchmark import ARCH_SUFFIX, detect_supported_arch


def run_cuda_demo(
    chapter_dir: Path,
    binary_name: str,
    extra_args: Optional[Iterable[str]] = None,
) -> int:
    """Build and run a CUDA demo binary in the given chapter directory."""
    arch = detect_supported_arch()
    suffix = ARCH_SUFFIX[arch]
    target = f"{binary_name}{suffix}"

    build_cmd = ["make", f"ARCH={arch}", target]
    result = subprocess.run(build_cmd, cwd=chapter_dir)
    if result.returncode != 0:
        return result.returncode

    exec_path = chapter_dir / target
    if not exec_path.exists():
        raise FileNotFoundError(f"Built binary not found at {exec_path}")

    run_cmd = [str(exec_path)]
    if extra_args:
        run_cmd.extend(list(extra_args))
    completed = subprocess.run(run_cmd, cwd=chapter_dir)
    return completed.returncode
