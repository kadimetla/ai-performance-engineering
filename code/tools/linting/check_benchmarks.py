#!/usr/bin/env python3
"""Thin wrapper to run the core benchmark linter."""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.scripts.linting.check_benchmarks import main


if __name__ == "__main__":
    main()
