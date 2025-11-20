"""Dispatcher for baseline vs optimized DDP runs."""

from __future__ import annotations

import argparse
import sys

import labs.train_distributed.baseline_ddp as baseline_run
import labs.train_distributed.optimized_ddp as optimized_run
import labs.train_distributed.baseline_ddp_flash as baseline_flash_run
import labs.train_distributed.optimized_ddp_flash as optimized_flash_run


def main():
    parser = argparse.ArgumentParser(description="DDP training examples.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized", "baseline_flash", "optimized_flash"],
        default="optimized",
        help="Which variant to execute.",
    )
    args, remaining = parser.parse_known_args()

    # Let the chosen script parse its own CLI flags.
    sys.argv = [sys.argv[0]] + remaining

    if args.mode == "baseline":
        baseline_run.main()
    elif args.mode == "optimized":
        optimized_run.main()
    elif args.mode == "baseline_flash":
        baseline_flash_run.main()
    else:
        optimized_flash_run.main()


if __name__ == "__main__":
    main()
