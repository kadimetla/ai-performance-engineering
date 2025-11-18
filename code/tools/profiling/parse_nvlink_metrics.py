#!/usr/bin/env python3
"""Extract NVLink throughput summaries from Nsight Systems reports.

Usage:
    python tools/profiling/parse_nvlink_metrics.py artifacts/<run>/<file>.nsys-rep

The script shells out to ``nsys stats --report nvlink --format csv`` and prints
the rows that contain throughput/byte metrics so teams can diff the values
without opening Nsight GUI.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _ensure_nsys_available() -> str:
    path = shutil.which("nsys")
    if path is None:
        raise RuntimeError("nsys binary not found in PATH. Install Nsight Systems to use this helper.")
    return path


def _run_nsys_stats(nsys_bin: str, report: Path) -> str:
    cmd = [
        nsys_bin,
        "stats",
        "--report",
        "nvlink",
        "--format",
        "csv",
        str(report),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"nsys stats failed for {report} (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def _parse_csv_sections(raw_csv: str) -> List[Dict[str, str]]:
    """Handle CSV output that repeats headers per section."""

    entries: List[Dict[str, str]] = []
    header: Optional[List[str]] = None
    reader = csv.reader(line for line in raw_csv.splitlines())
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            header = None
            continue
        if row[0].startswith("#"):
            continue
        if header is None:
            header = [cell.strip() for cell in row]
            continue
        if len(row) != len(header):
            continue
        entry = {header[i]: row[i].strip() for i in range(len(header))}
        entries.append(entry)
    return entries


def _select_keys(entry: Dict[str, str], needle: str) -> Optional[str]:
    for key, value in entry.items():
        if needle in key.lower():
            return value
    return None


def _summarize_entries(entries: Iterable[Dict[str, str]]) -> List[Tuple[str, str, Optional[float], str]]:
    """Return tuples containing (label, metric, value, units)."""
    summary: List[Tuple[str, str, Optional[float], str]] = []
    for entry in entries:
        metric_name = _select_keys(entry, "metric")
        if not metric_name:
            continue
        lower_name = metric_name.lower()
        if "throughput" not in lower_name and "bytes" not in lower_name:
            continue
        value_str = _select_keys(entry, "value")
        units = _select_keys(entry, "unit") or ""
        section = entry.get("Section") or entry.get("section") or ""
        link = entry.get("Link") or entry.get("link") or ""
        label_parts = [part for part in (section, link) if part]
        label = " / ".join(label_parts) if label_parts else "Total"
        try:
            value = float(value_str) if value_str not in (None, "") else None
        except ValueError:
            value = None
        summary.append((label, metric_name.strip(), value, units))
    return summary


def _print_summary(report: Path, rows: List[Tuple[str, str, Optional[float], str]]) -> None:
    if not rows:
        print(f"{report}: no NVLink throughput rows found")
        return

    print(f"{report}:")
    for label, metric, value, units in rows:
        if value is None:
            print(f"  {label:<12} {metric}")
        else:
            unit_suffix = f" {units}" if units else ""
            print(f"  {label:<12} {metric:<40} {value:.3f}{unit_suffix}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize NVLink throughput from Nsight Systems reports.")
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help=".nsys-rep files emitted by the benchmark CLI (under artifacts/<run_id>/.../*.nsys-rep)",
    )
    args = parser.parse_args(argv)

    nsys_bin = _ensure_nsys_available()
    exit_code = 0
    for report in args.reports:
        try:
            raw = _run_nsys_stats(nsys_bin, report)
            entries = _parse_csv_sections(raw)
            rows = _summarize_entries(entries)
            _print_summary(report, rows)
        except Exception as exc:  # pragma: no cover - best effort helper
            print(f"{report}: error extracting NVLink metrics: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
