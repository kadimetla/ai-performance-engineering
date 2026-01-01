"""Dispatcher for baseline vs optimized DualPipe demos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.gpu_requirements import require_min_gpus

import labs.train_distributed.baseline_pipeline_dualpipe_multigpu as baseline_run
import labs.train_distributed.optimized_pipeline_dualpipe_multigpu as optimized_run


def main():
    require_min_gpus(2, script_name="pipeline_dualpipe_multigpu.py")
    parser = argparse.ArgumentParser(description="DualPipe toy pipeline.")
    parser.add_argument("--mode", choices=["baseline", "optimized"], default="optimized")
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    if args.mode == "baseline":
        baseline_run.main()
    else:
        optimized_run.main()


if __name__ == "__main__":
    main()
