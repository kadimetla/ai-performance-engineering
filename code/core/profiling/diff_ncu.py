#!/usr/bin/env python3
"""Quick delta printer for Nsight Compute CSV exports."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python core/profiling/diff_ncu.py <baseline.csv> <candidate.csv>", file=sys.stderr)
        raise SystemExit(2)

    a_path = Path(sys.argv[1]).expanduser().resolve()
    b_path = Path(sys.argv[2]).expanduser().resolve()
    if not a_path.exists() or not b_path.exists():
        raise SystemExit(f"Missing CSV: {a_path} or {b_path}")

    baseline = pd.read_csv(a_path)
    candidate = pd.read_csv(b_path)
    joined = baseline.merge(candidate, on="Metric Name", suffixes=("_A", "_B"))

    for col in ("Metric Value", "Min", "Max", "Avg"):
        col_a = f"{col}_A"
        col_b = f"{col}_B"
        if col_a in joined and col_b in joined:
            joined[f"{col}_Δ"] = joined[col_b] - joined[col_a]

    keep_cols = ["Metric Name"]
    for col in ("Metric Value", "Avg"):
        for suffix in ("_A", "_B", "_Δ"):
            col_name = f"{col}{suffix}"
            if col_name in joined.columns:
                keep_cols.append(col_name)

    to_print = joined[keep_cols].sort_values(by=keep_cols[-1], ascending=False)
    print(to_print.to_string(index=False))


if __name__ == "__main__":
    main()
