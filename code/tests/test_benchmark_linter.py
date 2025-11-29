"""Test that the benchmark linter works correctly."""

import subprocess
import sys
from pathlib import Path
import pytest


def test_linter_discovers_benchmarks():
    """Test that linter can discover benchmarks without crashing."""
    repo_root = Path(__file__).parent.parent
    linter_path = repo_root / "scripts" / "linting" / "check_benchmarks.py"
    
    # Run linter (should not crash even if files have errors)
    result = subprocess.run(
        [sys.executable, str(linter_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(repo_root), **dict(subprocess.os.environ)},
    )
    
    # Should exit with non-zero if there are errors, but not crash
    # Check that it doesn't raise TypeError about missing arguments
    assert "TypeError" not in result.stderr, f"Linter crashed with TypeError: {result.stderr}"
    assert "discover_benchmarks() missing" not in result.stderr, f"Linter has discovery bug: {result.stderr}"
    
    # Should discover some benchmarks
    assert "Checking" in result.stdout or "benchmark files" in result.stdout.lower(), \
        f"Linter should discover benchmarks: {result.stdout}"


def test_linter_with_specific_path():
    """Test that linter works with specific file paths."""
    repo_root = Path(__file__).parent.parent
    linter_path = repo_root / "tools" / "linting" / "check_benchmarks.py"
    
    # Find a valid benchmark file
    benchmark_files = list(repo_root.glob("ch*/baseline_*.py"))
    if not benchmark_files:
        pytest.skip("No benchmark files found")
    
    test_file = benchmark_files[0]
    
    result = subprocess.run(
        [sys.executable, str(linter_path), str(test_file)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(repo_root), **dict(subprocess.os.environ)},
    )
    
    # Should not crash
    assert "TypeError" not in result.stderr, f"Linter crashed: {result.stderr}"


def test_linter_run_setup_flag():
    """Test that --run-setup flag exists and works."""
    repo_root = Path(__file__).parent.parent
    linter_path = repo_root / "scripts" / "linting" / "check_benchmarks.py"
    
    # Check help output
    result = subprocess.run(
        [sys.executable, str(linter_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    
    assert "--run-setup" in result.stdout, "Linter should have --run-setup flag"


def test_linter_works_without_cuda():
    """Test that linter works without CUDA (uses AST parsing)."""
    repo_root = Path(__file__).parent.parent
    linter_path = repo_root / "tools" / "linting" / "check_benchmarks.py"
    template_path = repo_root / "templates" / "benchmark_template.py"
    
    if not template_path.exists():
        pytest.skip("Template file not found")
    
    # Run linter without --run-setup (should use AST, not require CUDA)
    result = subprocess.run(
        [sys.executable, str(linter_path), str(template_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(repo_root), "CUDA_VISIBLE_DEVICES": "", **dict(subprocess.os.environ)},
    )
    
    # Should not fail with CUDA error
    assert "CUDA required" not in result.stderr, f"Linter should not require CUDA: {result.stderr}"
    assert "RuntimeError" not in result.stderr or "CUDA" not in result.stderr, \
        f"Linter should not raise CUDA RuntimeError: {result.stderr}"
    
    # Should complete successfully (may have warnings but no CUDA errors)
    assert result.returncode == 0 or "CUDA" not in result.stderr, \
        f"Linter should not fail due to CUDA: {result.stderr}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
