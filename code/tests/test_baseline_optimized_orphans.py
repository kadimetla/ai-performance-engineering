"""Ensure baseline_/optimized_ benchmark naming is paired.

This repository reserves `baseline_*` and `optimized_*` filenames for
harness-comparable benchmark variants. This test scans the working tree for
`baseline_*.py`, `optimized_*.py`, `baseline_*.cu`, and `optimized_*.cu` and
fails if it finds:

1) A baseline file without any matching optimized variant in the same directory
2) An optimized file without any matching baseline in the same directory

Matching rule (per directory, per extension):
  - `baseline_<name>.<ext>` matches `optimized_<name>.<ext>` and any
    `optimized_<name>_*.<ext>` variants.
  - `optimized_<name>.<ext>` is considered matched if there exists any
    `baseline_<prefix>.<ext>` where `<name> == <prefix>` or `<name>` starts with
    `<prefix>_`.

The failure message annotates each orphan with whether the harness would
directly discover it, using `core.discovery.discover_all_chapters()`.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

from core.discovery import discover_all_chapters

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keep this aligned with `core.discovery._iter_benchmark_dirs()` so the test is fast
# and does not traverse huge artifact/cache trees.
IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    ".torch_inductor",
    ".torch_extensions",
    ".next",
    ".turbo",
    "build",
    "dist",
    "out",
    "artifacts",
    "benchmark_profiles",
    "benchmark_profiles_chXX",
    "profiling_results",
    "hta_output",
    "gpt-oss-20b",
    "third_party",
}

BENCHMARK_SUFFIXES = {".py", ".cu"}


def _iter_candidate_files(repo_root: Path) -> Iterable[Path]:
    """Yield baseline_/optimized_ Python/CUDA sources from the repo working tree."""
    for current, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in IGNORE_DIRS
            and not d.startswith(".")
            and not d.startswith("artifacts")
            and not d.startswith("benchmark_profiles")
        ]
        for filename in filenames:
            if not (filename.startswith("baseline_") or filename.startswith("optimized_")):
                continue
            if Path(filename).suffix not in BENCHMARK_SUFFIXES:
                continue
            yield Path(current) / filename


def _matches_prefix(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(prefix + "_")


def _nearest_discoverable_ancestor(
    parent_dir: Path,
    discoverable_dirs: set[Path],
    repo_root: Path,
) -> Path | None:
    for ancestor in parent_dir.parents:
        if ancestor == repo_root:
            break
        if ancestor in discoverable_dirs:
            return ancestor
    return None


def _harness_discoverability_label(path: Path, discoverable_dirs: set[Path], repo_root: Path) -> str:
    parent = path.parent.resolve()
    if parent in discoverable_dirs:
        return "DISCOVERABLE(parent)"
    ancestor = _nearest_discoverable_ancestor(parent, discoverable_dirs, repo_root)
    if ancestor is not None:
        return f"NESTED(under={ancestor.relative_to(repo_root)})"
    return "OUTSIDE_DISCOVERY_TREE"


def test_no_orphan_baseline_or_optimized_files() -> None:
    """Fail if any baseline_/optimized_ file is missing its counterpart."""
    # Group by directory + extension so `.py` and `.cu` pairing is independent.
    groups: DefaultDict[
        Tuple[Path, str],
        Dict[str, DefaultDict[str, List[Path]]],
    ] = defaultdict(lambda: {"baseline": defaultdict(list), "optimized": defaultdict(list)})

    for path in _iter_candidate_files(REPO_ROOT):
        ext = path.suffix
        parent = path.parent
        if path.name.startswith("baseline_"):
            key = path.stem.replace("baseline_", "", 1)
            groups[(parent, ext)]["baseline"][key].append(path)
        else:
            key = path.stem.replace("optimized_", "", 1)
            groups[(parent, ext)]["optimized"][key].append(path)

    orphan_baselines: List[Path] = []
    orphan_optimized: List[Path] = []

    for (_parent, _ext), by_kind in groups.items():
        baseline_keys = set(by_kind["baseline"].keys())
        optimized_keys = set(by_kind["optimized"].keys())

        for baseline_key in baseline_keys:
            if not any(_matches_prefix(opt_key, baseline_key) for opt_key in optimized_keys):
                orphan_baselines.extend(by_kind["baseline"][baseline_key])

        for optimized_key in optimized_keys:
            if not any(_matches_prefix(optimized_key, baseline_key) for baseline_key in baseline_keys):
                orphan_optimized.extend(by_kind["optimized"][optimized_key])

    if not orphan_baselines and not orphan_optimized:
        return

    discoverable_dirs = {p.resolve() for p in discover_all_chapters(REPO_ROOT)}

    lines: List[str] = []
    lines.append("Orphan baseline_/optimized_ files found (these names are reserved for paired benchmarks).")
    lines.append("")
    lines.append("Harness discoverability:")
    lines.append("  - DISCOVERABLE(parent): harness scans this directory directly")
    lines.append("  - NESTED(...): under a discoverable directory, but parent dir is not scanned")
    lines.append("  - OUTSIDE_DISCOVERY_TREE: not under any discoverable benchmark directory")

    if orphan_baselines:
        lines.append("")
        lines.append(f"Orphan baselines ({len(orphan_baselines)}):")
        for path in sorted(orphan_baselines):
            label = _harness_discoverability_label(path, discoverable_dirs, REPO_ROOT)
            lines.append(f"  - {path.relative_to(REPO_ROOT)} [{label}]")

    if orphan_optimized:
        lines.append("")
        lines.append(f"Orphan optimized ({len(orphan_optimized)}):")
        for path in sorted(orphan_optimized):
            label = _harness_discoverability_label(path, discoverable_dirs, REPO_ROOT)
            lines.append(f"  - {path.relative_to(REPO_ROOT)} [{label}]")

    raise AssertionError("\n".join(lines))

