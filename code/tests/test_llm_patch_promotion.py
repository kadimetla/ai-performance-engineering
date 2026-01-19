from __future__ import annotations

from pathlib import Path

from core.analysis import llm_patch_promotion


def _resolve_promoted_path(promoted: str) -> Path:
    promoted_path = Path(promoted)
    if promoted_path.is_absolute():
        return promoted_path
    return llm_patch_promotion.REPO_ROOT / promoted_path


def test_promote_best_llm_patch_creates_file(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "ch10"
    patch_dir = chapter_dir / "llm_patches"
    patch_dir.mkdir(parents=True)
    patched_file = patch_dir / "optimized_toy_patch.py"
    patched_file.write_text("print('patch')\n")

    best_patch = {"patched_file": str(patched_file), "variant_name": "fast-path"}
    benchmark_result = {"example": "toy"}

    promoted = llm_patch_promotion.promote_best_llm_patch(best_patch, benchmark_result, chapter_dir)
    assert promoted is not None

    promoted_path = _resolve_promoted_path(promoted)
    assert promoted_path.exists()
    assert promoted_path.name.startswith("optimized_toy_llm_fast_path")


def test_promote_best_llm_patch_reuses_existing(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "ch10"
    patch_dir = chapter_dir / "llm_patches"
    patch_dir.mkdir(parents=True)
    patched_file = patch_dir / "optimized_toy_patch.py"
    patched_file.write_text("print('patch')\n")

    best_patch = {"patched_file": str(patched_file), "variant_name": "fast-path"}
    benchmark_result = {"example": "toy"}

    promoted_first = llm_patch_promotion.promote_best_llm_patch(best_patch, benchmark_result, chapter_dir)
    promoted_second = llm_patch_promotion.promote_best_llm_patch(best_patch, benchmark_result, chapter_dir)

    assert promoted_first is not None
    assert promoted_second is not None
    assert _resolve_promoted_path(promoted_first) == _resolve_promoted_path(promoted_second)
    assert len(list(chapter_dir.glob("optimized_toy_llm_fast_path*.py"))) == 1


def test_promote_best_llm_patch_skips_failed_verification(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "ch10"
    patch_dir = chapter_dir / "llm_patches"
    patch_dir.mkdir(parents=True)
    patched_file = patch_dir / "optimized_toy_patch.py"
    patched_file.write_text("print('patch')\n")

    best_patch = {
        "patched_file": str(patched_file),
        "variant_name": "fast-path",
        "verification": {"verified": False, "errors": ["mismatch"]},
    }
    benchmark_result = {"example": "toy"}

    promoted = llm_patch_promotion.promote_best_llm_patch(best_patch, benchmark_result, chapter_dir)
    assert promoted is None
    assert not list(chapter_dir.glob("optimized_toy_llm_fast_path*.py"))
