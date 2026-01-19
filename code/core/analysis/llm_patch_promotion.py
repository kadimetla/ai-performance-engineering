from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from core.utils.logger import get_logger

logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _sanitize_llm_variant_name(variant_name: str) -> str:
    """Convert LLM variant names into safe filename fragments."""
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", variant_name or "")
    safe = re.sub(r"_+", "_", safe).strip("_").lower()
    return safe


def promote_best_llm_patch(
    best_patch: Dict[str, Any],
    benchmark_result: Dict[str, Any],
    chapter_dir: Path,
) -> Optional[str]:
    """Promote the best LLM patch into the chapter directory as optimized_*."""
    patched_file = best_patch.get("patched_file")
    if not patched_file:
        return None

    patched_path = Path(patched_file)
    if not patched_path.exists():
        logger.warning("    WARNING: Best patch file missing: %s", patched_file)
        return None

    verification = best_patch.get("verification")
    if isinstance(verification, dict) and verification.get("verified") is False:
        logger.warning("    WARNING: Best patch failed verification; skipping promotion.")
        return None

    example_name = benchmark_result.get("example", "unknown")
    variant_name = _sanitize_llm_variant_name(best_patch.get("variant_name", ""))
    if variant_name:
        stem = f"optimized_{example_name}_llm_{variant_name}"
    else:
        stem = f"optimized_{example_name}_llm_best"

    target_path = chapter_dir / f"{stem}{patched_path.suffix}"

    if target_path.exists():
        try:
            if target_path.read_bytes() == patched_path.read_bytes():
                return _relative_path(target_path)
        except OSError as exc:
            logger.warning("    WARNING: Unable to read existing promoted patch: %s", exc)

        for idx in range(2, 100):
            candidate = chapter_dir / f"{stem}_v{idx}{patched_path.suffix}"
            if not candidate.exists():
                target_path = candidate
                break
        else:
            logger.warning(
                "    WARNING: No available filename to promote best patch for %s",
                example_name,
            )
            return None

    shutil.copy2(patched_path, target_path)
    logger.info("    Promoted best patch to %s", target_path.name)

    return _relative_path(target_path)


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
