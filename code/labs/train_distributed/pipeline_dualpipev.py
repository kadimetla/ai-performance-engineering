"""Dispatcher for baseline vs optimized DualPipeV demos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import labs.train_distributed.baseline_pipeline_dualpipev as baseline_run
import labs.train_distributed.optimized_pipeline_dualpipev as optimized_run


def main():
    parser = argparse.ArgumentParser(description="DualPipeV toy pipeline.")
    parser.add_argument("--mode", choices=["baseline", "optimized"], default="optimized")
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    if args.mode == "baseline":
        baseline_run.main()
    else:
        optimized_run.main()


if __name__ == "__main__":
    main()
