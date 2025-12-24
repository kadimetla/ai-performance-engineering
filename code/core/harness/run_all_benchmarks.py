#!/usr/bin/env python3
"""Compatibility wrapper for run_benchmarks.

Historically, both run_benchmarks.py and run_all_benchmarks.py implemented the
same harness logic. This module now delegates to run_benchmarks to avoid drift
while preserving the legacy entrypoint and import surface.
"""

from __future__ import annotations

from core.harness.run_benchmarks import *  # noqa: F401,F403
from core.harness.run_benchmarks import main as _main


if __name__ == "__main__":
    _main()
