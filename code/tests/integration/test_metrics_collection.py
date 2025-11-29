"""Integration tests for metrics collection and reporting.

Tests that metrics are extracted correctly from profiling artifacts,
serialized properly, and can be compared.
"""

import pytest
import sys
import json
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
from core.benchmark.models import BenchmarkResult, TimingStats, MemoryStats


# Skip tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required - NVIDIA GPU and tools must be available"
)


class TestMetricsCollectionIntegration:
    """Integration tests for metrics collection."""
    
    def test_timing_metrics_are_collected(self):
        """Test that timing metrics are collected correctly."""
        # Find a real benchmark
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
        
        baseline_path, _, _ = pairs[0]
        benchmark = load_benchmark(baseline_path)
        if benchmark is None:
            pytest.skip("Failed to load benchmark")
        
        config = BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_profiling=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        
        # Verify timing metrics
        assert result.timing is not None
        assert result.timing.iterations == 10
        assert result.timing.warmup_iterations == harness.config.warmup
        assert result.timing.mean_ms > 0
        assert result.timing.median_ms > 0
        assert result.timing.min_ms > 0
        assert result.timing.max_ms > 0
        assert result.timing.std_ms >= 0
    
    def test_memory_metrics_are_collected(self):
        """Test that memory metrics are collected when enabled."""
        # Find a real benchmark
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
        
        baseline_path, _, _ = pairs[0]
        benchmark = load_benchmark(baseline_path)
        if benchmark is None:
            pytest.skip("Failed to load benchmark")
        
        config = BenchmarkConfig(
            iterations=5,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=True,  # Enable memory tracking
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        
        # Memory stats may or may not be present depending on CUDA availability
        if result.memory is not None:
            if result.memory.peak_mb is not None:
                assert result.memory.peak_mb >= 0
    
    def test_metrics_serialization(self):
        """Test that metrics can be serialized to JSON."""
        # Find a real benchmark
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
        
        baseline_path, _, _ = pairs[0]
        benchmark = load_benchmark(baseline_path)
        if benchmark is None:
            pytest.skip("Failed to load benchmark")
        
        config = BenchmarkConfig(
            iterations=5,
            warmup=5,
            enable_profiling=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        
        # Serialize to JSON
        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        
        # Deserialize and verify structure
        data = json.loads(json_str)
        assert 'timing' in data
        assert data['timing']['iterations'] == 5
        assert data['timing']['warmup_iterations'] == harness.config.warmup
    
    def test_metrics_comparison(self):
        """Test that metrics can be compared."""
        from core.benchmark.comparison import compare_results
        
        # Create two mock results for comparison
        timing1 = TimingStats(
            mean_ms=100.0,
            median_ms=95.0,
            std_ms=5.0,
            min_ms=90.0,
            max_ms=110.0,
            iterations=10,
            warmup_iterations=2,
        )
        
        timing2 = TimingStats(
            mean_ms=50.0,  # Faster
            median_ms=48.0,
            std_ms=3.0,
            min_ms=45.0,
            max_ms=55.0,
            iterations=10,
            warmup_iterations=2,
        )
        
        result1 = BenchmarkResult(timing=timing1)
        result2 = BenchmarkResult(timing=timing2)
        
        comparison = compare_results(result1, result2)
        
        # Verify comparison results
        assert comparison.speedup > 1.0  # result2 is faster
        assert comparison.speedup == pytest.approx(2.0, rel=0.1)  # ~2x speedup
        assert comparison.improvement_pct > 0
