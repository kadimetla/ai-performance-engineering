"""Integration tests for full benchmark execution pipeline.

Tests that benchmarks can be executed end-to-end with manifest generation,
artifact management, and error handling.
"""

import pytest
import sys
import json
from pathlib import Path
from tempfile import TemporaryDirectory

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
from core.benchmark.artifact_manager import ArtifactManager


# Skip tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required - NVIDIA GPU and tools must be available"
)


class TestBenchmarkExecutionPipeline:
    """Integration tests for full benchmark execution."""
    
    def test_benchmark_with_manifest_generates_manifest(self):
        """Test that benchmark_with_manifest generates a complete manifest."""
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
            enable_memory_tracking=True,
            adaptive_iterations=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        # Run with manifest
        run = harness.benchmark_with_manifest(benchmark, run_id="test_run")
        
        # Verify manifest exists
        assert run.manifest is not None
        assert run.manifest.hardware is not None
        assert run.manifest.software is not None
        assert run.manifest.git is not None
        
        # Verify result exists
        assert run.result is not None
        assert run.result.timing is not None
        assert run.result.timing.iterations == 5
    
    def test_benchmark_execution_produces_valid_result(self):
        """Test that benchmark execution produces valid results."""
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
            enable_memory_tracking=True,
            adaptive_iterations=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        
        # Verify result structure
        assert result.timing is not None
        assert result.timing.iterations == 10
        assert result.timing.warmup_iterations == harness.config.warmup
        assert result.timing.mean_ms > 0
        assert result.timing.median_ms > 0
        
        # Memory stats may or may not be present
        if result.memory is not None:
            assert result.memory.peak_mb >= 0
    
    def test_benchmark_execution_handles_errors_gracefully(self):
        """Test that benchmark execution handles errors gracefully."""
        # Create a benchmark that will fail
        from core.harness.benchmark_harness import BaseBenchmark
        
        class FailingBenchmark(BaseBenchmark):
            def setup(self):
                pass
            
            def benchmark_fn(self):
                raise RuntimeError("Intentional failure")
            
            def get_verify_inputs(self):
                return {"input": torch.tensor([0.0], device=self.device)}
            
            def get_verify_output(self):
                raise RuntimeError("benchmark_fn() must set output")
            
            def get_output_tolerance(self):
                return (1e-5, 1e-8)
        
        benchmark = FailingBenchmark()
        config = BenchmarkConfig(
            iterations=5,
            warmup=5,
            enable_profiling=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        # Should raise RuntimeError or return result with errors
        try:
            result = harness.benchmark(benchmark)
            # If we get a result, it should have errors
            assert len(result.errors) > 0
        except RuntimeError:
            # RuntimeError is also acceptable
            pass
    
    def test_artifact_manager_creates_directories(self):
        """Test that ArtifactManager creates proper directory structure."""
        with TemporaryDirectory() as tmpdir:
            artifact_manager = ArtifactManager(base_dir=Path(tmpdir))
            
            # Verify directories are created
            assert artifact_manager.run_dir.exists()
            assert artifact_manager.results_dir.exists()
            assert artifact_manager.logs_dir.exists()
            assert artifact_manager.reports_dir.exists()
            
            # Verify log path exists
            log_path = artifact_manager.get_log_path()
            assert log_path.parent.exists()
    
    def test_benchmark_result_serialization(self):
        """Test that benchmark results can be serialized to JSON."""
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
        
        run = harness.benchmark_with_manifest(benchmark, run_id="test_serialization")
        
        # Serialize to JSON
        json_str = run.model_dump_json()
        assert isinstance(json_str, str)
        
        # Deserialize and verify
        data = json.loads(json_str)
        assert 'manifest' in data
        assert 'result' in data
        assert 'run_id' in data
        assert data['run_id'] == "test_serialization"
