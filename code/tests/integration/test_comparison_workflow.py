"""Integration tests for baseline vs optimized comparison workflows.

Tests that comparisons work correctly, speedup calculations are accurate,
and edge cases are handled properly.
"""

import pytest
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.env import apply_env_defaults
apply_env_defaults()

import torch
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig
from core.utils.chapter_compare_template import discover_benchmarks, load_benchmark
from core.discovery import discover_all_chapters
from core.benchmark.comparison import compare_results, ComparisonResult


# Skip tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required - NVIDIA GPU and tools must be available"
)


class TestComparisonWorkflowIntegration:
    """Integration tests for comparison workflows."""
    
    def test_baseline_optimized_comparison(self):
        """Test comparing baseline and optimized benchmarks."""
        # Find a real benchmark pair
        chapters = discover_all_chapters(repo_root)
        if not chapters:
            pytest.skip("No chapters found")
        
        pairs = None
        for chapter_dir in chapters:
            chapter_pairs = discover_benchmarks(chapter_dir)
            if chapter_pairs:
                pairs = chapter_pairs
                break
        
        if not pairs:
            pytest.skip("No benchmark pairs found")
        
        baseline_path, optimized_paths, _ = pairs[0]
        
        baseline = load_benchmark(baseline_path)
        if baseline is None:
            pytest.skip("Failed to load baseline")
        
        if not optimized_paths:
            pytest.skip("No optimized benchmarks found")
        
        optimized = load_benchmark(optimized_paths[0])
        if optimized is None:
            pytest.skip("Failed to load optimized benchmark")
        
        config = BenchmarkConfig(
            iterations=5,
            warmup=5,
            enable_profiling=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        # Run both benchmarks
        baseline_result = harness.benchmark(baseline)
        optimized_result = harness.benchmark(optimized)
        
        # Compare results
        comparison = compare_results(baseline_result, optimized_result)
        
        # Verify comparison structure
        assert comparison.speedup > 0
        assert comparison.baseline_mean_ms > 0
        assert comparison.optimized_mean_ms > 0
        if comparison.improvement_pct is not None:
            assert comparison.improvement_pct >= 0
    
    def test_comparison_with_same_performance(self):
        """Test comparison when both benchmarks have same performance."""
        from core.benchmark.models import BenchmarkResult, TimingStats
        
        # Create two results with identical timing
        timing = TimingStats(
            mean_ms=100.0,
            median_ms=95.0,
            std_ms=5.0,
            min_ms=90.0,
            max_ms=110.0,
            iterations=10,
            warmup_iterations=2,
        )
        
        result1 = BenchmarkResult(timing=timing)
        result2 = BenchmarkResult(timing=timing)
        
        comparison = compare_results(result1, result2)
        
        # Speedup should be 1.0 (no improvement)
        assert comparison.speedup == pytest.approx(1.0, rel=0.01)
        assert comparison.improvement_pct is None
    
    def test_comparison_with_regression(self):
        """Test comparison when optimized is slower (regression)."""
        from core.benchmark.models import BenchmarkResult, TimingStats
        
        # Baseline is faster
        baseline_timing = TimingStats(
            mean_ms=50.0,
            median_ms=48.0,
            std_ms=3.0,
            min_ms=45.0,
            max_ms=55.0,
            iterations=10,
            warmup_iterations=2,
        )
        
        # Optimized is slower (regression)
        optimized_timing = TimingStats(
            mean_ms=100.0,
            median_ms=95.0,
            std_ms=5.0,
            min_ms=90.0,
            max_ms=110.0,
            iterations=10,
            warmup_iterations=2,
        )
        
        baseline_result = BenchmarkResult(timing=baseline_timing)
        optimized_result = BenchmarkResult(timing=optimized_timing)
        
        comparison = compare_results(baseline_result, optimized_result)
        
        # Speedup should be < 1.0 (regression)
        assert comparison.speedup < 1.0
        assert comparison.regression is True
        assert comparison.regression_pct is not None and comparison.regression_pct > 0
    
    def test_comparison_with_missing_timing(self):
        """Test comparison handles missing timing gracefully."""
        from core.benchmark.models import BenchmarkResult
        
        # Creating results without timing should raise validation errors
        with pytest.raises(Exception):
            BenchmarkResult(timing=None)  # type: ignore[arg-type]
    
    def test_multiple_optimizations_comparison(self):
        """Test comparing baseline against multiple optimizations."""
        # Find a benchmark pair with multiple optimizations
        chapters = discover_all_chapters(repo_root)
        if not chapters:
            pytest.skip("No chapters found")
        
        pairs = None
        for chapter_dir in chapters:
            chapter_pairs = discover_benchmarks(chapter_dir)
            # Look for pairs with multiple optimizations
            for pair in chapter_pairs:
                if len(pair[1]) > 1:  # Multiple optimized versions
                    pairs = [pair]
                    break
            if pairs:
                break
        
        if not pairs:
            pytest.skip("No benchmark pairs with multiple optimizations found")
        
        baseline_path, optimized_paths, _ = pairs[0]
        
        baseline = load_benchmark(baseline_path)
        if baseline is None:
            pytest.skip("Failed to load baseline")
        
        config = BenchmarkConfig(
            iterations=5,
            warmup=5,
            enable_profiling=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        try:
            baseline_result = harness.benchmark(baseline)
        except RuntimeError as e:
            msg = str(e)
            if "skipped" in msg.lower() or "requires >=2 gpus" in msg.lower():
                pytest.skip(msg)
            raise
        
        # Compare against each optimization
        comparisons = []
        for opt_path in optimized_paths[:2]:  # Limit to first 2 for speed
            optimized = load_benchmark(opt_path)
            if optimized is None:
                continue
            
            try:
                optimized_result = harness.benchmark(optimized)
            except RuntimeError as e:
                msg = str(e).lower()
                if "multi gpu" in msg or "multiple gpu" in msg:
                    continue
                raise
            comparison = compare_results(baseline_result, optimized_result)
            comparisons.append(comparison)
        
        # Verify we got comparisons
        assert len(comparisons) > 0
        for comparison in comparisons:
            assert comparison.speedup > 0
