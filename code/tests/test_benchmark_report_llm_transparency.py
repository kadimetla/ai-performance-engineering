"""Tests for LLM transparency sections in benchmark reports."""

from pathlib import Path
import sys


repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.run_benchmarks import generate_markdown_report


def test_report_includes_llm_analysis_and_patch_diff(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    chapter_dir = bench_root / "ch99"
    chapter_dir.mkdir(parents=True)

    baseline_path = chapter_dir / "baseline_example.py"
    optimized_path = chapter_dir / "optimized_example.py"
    patch_dir = chapter_dir / "llm_patches"
    patch_dir.mkdir()
    patched_path = patch_dir / "optimized_example_test_patch.py"

    baseline_path.write_text("def foo():\n    return 1\n")
    optimized_path.write_text("def foo():\n    return 2\n")
    patched_path.write_text("def foo():\n    return 3\n")

    llm_analysis_dir = chapter_dir / "llm_analysis"
    llm_analysis_dir.mkdir()
    llm_analysis_path = llm_analysis_dir / "llm_analysis_example.md"
    llm_analysis_path.write_text(
        "# LLM Performance Analysis\n\n"
        "## Why Is It Faster?\n\n"
        "The optimized version overlaps streams.\n\n"
        "## Suggested Code Changes\n\n"
        "- Add a warmup loop.\n"
    )

    llm_explain_dir = chapter_dir / "llm_explanations"
    llm_explain_dir.mkdir()
    llm_explain_path = llm_explain_dir / "explanation_example.md"
    llm_explain_path.write_text("# Explanation\n\nDetails here.\n")

    results = [
        {
            "chapter": "ch99",
            "status": "completed",
            "benchmarks": [
                {
                    "example": "example",
                    "type": "python",
                    "baseline_file": "baseline_example.py",
                    "baseline_time_ms": 10.0,
                    "optimizations": [
                        {
                            "file": "optimized_example.py",
                            "status": "succeeded",
                            "time_ms": 5.0,
                            "speedup": 2.0,
                        }
                    ],
                    "best_speedup": 2.0,
                    "status": "succeeded",
                    "llm_analysis": {
                        "md_path": str(llm_analysis_path),
                        "provider": "test",
                        "model": "unit-test",
                        "latency_seconds": 1.2,
                        "cached": False,
                    },
                    "llm_patches": [
                        {
                            "success": True,
                            "patched_file": str(patched_path),
                            "variant_name": "test_patch",
                            "description": "Test patch",
                            "expected_speedup": "1.1x",
                            "validation_errors": [],
                            "can_benchmark": True,
                            "rebenchmark_result": {"success": True, "median_ms": 4.0},
                            "actual_speedup": 2.5,
                            "verification": {"verified": True},
                        }
                    ],
                    "best_llm_patch": {
                        "variant_name": "test_patch",
                        "patched_file": str(patched_path),
                        "actual_speedup": 2.5,
                        "median_ms": 4.0,
                    },
                }
            ],
            "summary": {
                "total_benchmarks": 1,
                "successful": 1,
                "failed": 0,
                "failed_error": 0,
                "failed_regression": 0,
                "skipped_hardware": 0,
                "skipped_distributed": 0,
                "total_skipped": 0,
                "total_speedups": 1,
                "average_speedup": 2.0,
                "max_speedup": 2.0,
                "min_speedup": 2.0,
                "informational": 0,
            },
        }
    ]

    output_md = tmp_path / "benchmark_test_results.md"
    generate_markdown_report(results, output_md, bench_root=bench_root)

    report_text = output_md.read_text()
    assert "LLM Transparency" in report_text
    assert "LLM Analysis & Patch Diffs" in report_text
    assert "Why Is It Faster?" in report_text
    assert "Patch diff" in report_text
    assert "-    return 2" in report_text
    assert "+    return 3" in report_text
