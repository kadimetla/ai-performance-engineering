"""Helpers for UMA memory reporting benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class MemInfoSnapshot:
    mem_total_kb: int
    mem_available_kb: int
    swap_free_kb: int

    def effective_available_kb(self) -> int:
        """Host memory that can be reclaimed without swapping."""
        return self.mem_available_kb

    def allocatable_bytes(self, reclaim_fraction: float = 0.9) -> int:
        """Estimate allocatable UMA bytes (MemAvailable + reclaimable swap)."""
        reclaimable = int(self.swap_free_kb * reclaim_fraction)
        return (self.mem_available_kb + reclaimable) * 1024


def format_bytes(num_bytes: int) -> str:
    """Human readable bytes formatter (GiB/MiB/KiB)."""
    suffixes = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for suffix in suffixes:
        if value < 1024.0 or suffix == suffixes[-1]:
            return f"{value:.2f} {suffix}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def is_integrated_gpu() -> bool:
    """Best-effort detection of an integrated/UMA GPU."""
    try:
        props = torch.cuda.get_device_properties(0)
        return bool(getattr(props, "integrated", False))
    except Exception:
        return False


def _parse_kb(line: str) -> Optional[int]:
    try:
        value = line.split(":", 1)[1].strip().split()[0]
        return int(value)
    except Exception:
        return None


def read_meminfo(path: Path | str = "/proc/meminfo") -> Optional[MemInfoSnapshot]:
    """Parse /proc/meminfo and return a snapshot. Returns None on failure."""
    try:
        mem_total_kb = mem_available_kb = swap_free_kb = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_kb = _parse_kb(line)
                elif line.startswith("MemAvailable:"):
                    mem_available_kb = _parse_kb(line)
                elif line.startswith("SwapFree:"):
                    swap_free_kb = _parse_kb(line)
        if mem_total_kb is None or mem_available_kb is None:
            return None
        if swap_free_kb is None:
            swap_free_kb = 0
        return MemInfoSnapshot(
            mem_total_kb=int(mem_total_kb),
            mem_available_kb=int(mem_available_kb),
            swap_free_kb=int(swap_free_kb),
        )
    except Exception:
        return None
