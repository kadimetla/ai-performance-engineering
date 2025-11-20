#!/usr/bin/env python3
"""
Fallback Proton CLI shim.

Provides a minimal `proton profile` command that executes the requested Python script and emits a
JSON report compatible with the harness extractors. This is not a substitute for the official
Proton profiler – install the upstream CLI for real occupancy data.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _run_python_script(script: str, args: List[str]) -> float:
    """Execute the target Python script and return elapsed milliseconds."""
    cmd = [sys.executable, script, *args]
    start = time.perf_counter()
    result = subprocess.run(cmd)
    end = time.perf_counter()
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark script exited with status {result.returncode}")
    return (end - start) * 1000.0


def _write_stub_report(output: Path, script: str, duration_ms: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "results": [
            {
                "name": Path(script).stem,
                "registers_per_thread": None,
                "shared_memory_bytes": None,
                "blocks_per_sm": None,
                "occupancy_pct": None,
                "time_ms": duration_ms,
                "notes": "Proton stub fallback – install the official Proton CLI for full metrics.",
            }
        ],
        "summary_stats": {"stub_duration_ms": duration_ms},
        "metadata": {
            "stub": True,
            "proton_cli": "stub",
            "python_executable": sys.executable,
        },
    }
    output.write_text(json.dumps(report, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(prog="proton", description="Proton profiler stub")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Run benchmark and emit stub Proton report")
    profile.add_argument("--output", required=True, help="Path to Proton JSON output")
    profile.add_argument("--python-script", required=True, help="Benchmark wrapper script to execute")
    profile.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the script")

    args = parser.parse_args()

    if args.command == "profile":
        extra = args.script_args
        if extra and extra[0] == "--":
            extra = extra[1:]
        duration_ms = _run_python_script(args.python_script, extra)
        _write_stub_report(Path(args.output), args.python_script, duration_ms)
        return 0

    parser.error(f"Unsupported command {args.command}")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"[proton stub] {exc}", file=sys.stderr)
        sys.exit(1)
