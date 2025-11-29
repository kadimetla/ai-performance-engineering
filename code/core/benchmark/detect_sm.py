#!/usr/bin/env python3
"""Best-effort detector for the active GPU SM architecture."""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Dict, Tuple

# Canonical architecture table (aligned with book specs and expectation JSONs)
# B200  -> CC 10.0, sm_100, 208 SMs
# B300  -> CC 10.3, sm_103, high SM count
# GB200/GB300 -> CC 12.1, sm_121 (canonical for all 12.x variants), high SM count
ARCH_SPECS: Dict[Tuple[int, int], Dict[str, object]] = {
    (10, 0): {
        "sm": "sm_100",
        "compute_capability": "10.0",
        "sm_count": 208,
        "label": "Blackwell B200",
    },
    (10, 3): {
        "sm": "sm_103",
        "compute_capability": "10.3",
        "sm_count": "high",
        "label": "Blackwell Ultra B300",
    },
    (12, 1): {
        "sm": "sm_121",
        "compute_capability": "12.1",
        "sm_count": "high",
        "label": "Grace-Blackwell GB200/GB300",
    },
}
# Compute capability → sm string for build tooling (keeps legacy sm_12x variants)
SM_MAP: Dict[Tuple[int, int], str] = {
    (10, 0): "sm_100",
    (10, 3): "sm_103",
    (12, 0): "sm_120",
    (12, 1): "sm_121",
    (12, 2): "sm_122",
    (12, 3): "sm_123",
}
SM_REVERSE: Dict[str, Tuple[int, int]] = {sm: cc for cc, sm in SM_MAP.items()}
# Normalize any 12.x variant to the canonical 12.1 Grace-Blackwell spec for reporting.
ARCH_CANONICAL: Dict[Tuple[int, int], Tuple[int, int]] = {
    (10, 0): (10, 0),
    (10, 3): (10, 3),
    (12, 0): (12, 1),
    (12, 1): (12, 1),
    (12, 2): (12, 1),
    (12, 3): (12, 1),
}


def map_cc(major: int, minor: int) -> str:
    # Prefer explicit mappings, but allow forward-compatible fallbacks.
    if (major, minor) in SM_MAP:
        return SM_MAP[(major, minor)]
    if major == 10:
        # Default unknown 10.x to B200 (sm_100) unless minor is clearly Ultra (>=3)
        return "sm_103" if minor >= 3 else "sm_100"
    if major == 12:
        # Normalize newer Grace-Blackwell minors to canonical sm_121
        return "sm_121"
    return ""


def get_arch_spec(major: int, minor: int) -> Dict[str, object]:
    """Return canonical architecture metadata for a given CC."""
    normalized = ARCH_CANONICAL.get((major, minor), (major, minor))
    spec = ARCH_SPECS.get(normalized)
    if spec:
        return {
            "major": major,
            "minor": minor,
            "sm": map_cc(major, minor),
            "compute_capability": spec["compute_capability"],
            "sm_count": spec["sm_count"],
            "label": spec["label"],
        }
    return {
        "major": major,
        "minor": minor,
        "sm": map_cc(major, minor),
        "compute_capability": f"{major}.{minor}",
        "sm_count": None,
        "label": "",
    }


def detect_with_torch() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return map_cc(props.major, props.minor)
    except Exception:
        return ""
    return ""


def detect_with_nvidia_smi() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        return ""
    if not out:
        return ""
    first = out.splitlines()[0]
    parts = first.strip().split(".")
    try:
        major = int(parts[0]) if parts else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return ""
    return map_cc(major, minor)


def main() -> int:
    arch = detect_with_torch() or detect_with_nvidia_smi()

    # Optional JSON output for downstream scripts: detect_sm.py --json
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        if arch:
            # map back to CC for metadata
            cc = SM_REVERSE.get(arch)
            if cc:
                major, minor = cc
                meta = get_arch_spec(major, minor)
            else:
                meta = {
                    "sm": arch,
                    "compute_capability": None,
                    "sm_count": None,
                    "label": "",
                    "major": None,
                    "minor": None,
                }
        else:
            meta = {"sm": "", "compute_capability": None, "sm_count": None, "label": ""}
        sys.stdout.write(json.dumps(meta))
        return 0

    sys.stdout.write(arch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
