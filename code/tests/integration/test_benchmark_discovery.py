"""Integration tests for benchmark discovery across chapters.

Tests that discovery works correctly across multiple chapters and handles
edge cases like missing files, malformed pairs, etc.
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
from core.discovery import (
    discover_benchmarks,
    discover_cuda_benchmarks,
    discover_all_chapters,
    discover_benchmark_pairs,
)
from core.utils.chapter_compare_template import load_benchmark


# Skip tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required - NVIDIA GPU and tools must be available"
)


class TestBenchmarkDiscoveryIntegration:
    """Integration tests for benchmark discovery."""
    
    def test_discover_all_chapters_finds_real_chapters(self):
        """Test that discovery finds real chapter directories."""
        chapters = discover_all_chapters(repo_root)
        
        assert len(chapters) > 0
        for chapter_dir in chapters:
            relative = chapter_dir.relative_to(repo_root)
            assert chapter_dir.exists()
            assert chapter_dir.is_dir()
            if relative.parts[0] == "labs":
                # Labs live under labs/ and may not use ch* naming
                assert len(relative.parts) >= 2
                continue
            # Benchmarks may live in nested dirs under a chapter; require the
            # top-level component to be a real chXX directory.
            top = relative.parts[0]
            assert top.startswith("ch")
            assert top[2:].isdigit()
    
    def test_discover_benchmarks_finds_real_pairs(self):
        """Test that discovery finds real baseline/optimized pairs."""
        # Find a chapter that exists
        chapters = discover_all_chapters(repo_root)
        if not chapters:
            pytest.skip("No chapters found")
        
        # Test first available chapter
        chapter_dir = chapters[0]
        pairs = discover_benchmarks(chapter_dir)
        
        # Should find at least some pairs if chapter has benchmarks
        assert isinstance(pairs, list)
        
        # If pairs exist, verify they're valid
        for baseline_path, optimized_paths, example_name in pairs:
            assert baseline_path.exists()
            assert baseline_path.name.startswith("baseline_")
            assert len(optimized_paths) > 0
            for opt_path in optimized_paths:
                assert opt_path.exists()
                assert opt_path.name.startswith("optimized_")
            assert isinstance(example_name, str)
            assert len(example_name) > 0
    
    def test_discover_benchmark_pairs_all_chapters(self):
        """Test discovering pairs across all chapters."""
        all_pairs = discover_benchmark_pairs(repo_root, chapter="all")
        
        assert isinstance(all_pairs, list)
        
        # Verify all pairs are valid
        for baseline_path, optimized_paths, example_name in all_pairs:
            assert baseline_path.exists()
            assert len(optimized_paths) > 0
    
    def test_discover_benchmark_pairs_specific_chapter(self):
        """Test discovering pairs in a specific chapter."""
        chapters = discover_all_chapters(repo_root)
        if not chapters:
            pytest.skip("No chapters found")
        
        chapter_name = chapters[0].name
        pairs = discover_benchmark_pairs(repo_root, chapter=chapter_name)
        
        assert isinstance(pairs, list)
        # All pairs should be from the specified chapter
        for baseline_path, _, _ in pairs:
            assert baseline_path.parent.name == chapter_name
    
    def test_load_benchmark_from_discovered_pair(self):
        """Test that discovered benchmarks can actually be loaded."""
        chapters = discover_all_chapters(repo_root)
        if not chapters:
            pytest.skip("No chapters found")
        
        # Find a chapter with pairs
        pairs = None
        for chapter_dir in chapters:
            chapter_pairs = discover_benchmarks(chapter_dir)
            if chapter_pairs:
                pairs = chapter_pairs
                break
        
        if not pairs:
            pytest.skip("No benchmark pairs found")
        
        # Try to load the first baseline
        baseline_path, _, _ = pairs[0]
        benchmark = load_benchmark(baseline_path)
        
        assert benchmark is not None
        # Verify it has required methods
        assert hasattr(benchmark, 'benchmark_fn')
        assert callable(benchmark.benchmark_fn)


class TestCudaBenchmarkDiscoveryIntegration:
    """Integration tests for CUDA benchmark discovery."""
    
    def test_discover_cuda_benchmarks_finds_real_files(self):
        """Test that CUDA discovery finds real .cu files if they exist."""
        cuda_benchmarks = discover_cuda_benchmarks(repo_root)
        
        assert isinstance(cuda_benchmarks, list)
        
        # If CUDA files exist, verify they're valid
        for cuda_file in cuda_benchmarks:
            assert cuda_file.exists()
            assert cuda_file.suffix == '.cu'


class TestDiscoveryEdgeCases:
    """Test discovery handles edge cases correctly."""
    
    def test_discover_nonexistent_chapter(self):
        """Test that discovery handles nonexistent chapters gracefully."""
        pairs = discover_benchmark_pairs(repo_root, chapter="ch99999")
        
        assert isinstance(pairs, list)
        assert len(pairs) == 0
    
    def test_discover_empty_chapter(self, tmp_path):
        """Test that discovery handles empty chapters gracefully."""
        empty_chapter = tmp_path / "ch99"
        empty_chapter.mkdir()
        
        pairs = discover_benchmarks(empty_chapter)
        
        assert isinstance(pairs, list)
        assert len(pairs) == 0
    
    def test_discover_chapter_with_only_baseline(self, tmp_path):
        """Test that discovery handles chapters with only baseline files."""
        chapter_dir = tmp_path / "ch99"
        chapter_dir.mkdir()
        
        baseline_file = chapter_dir / "baseline_test.py"
        baseline_file.write_text("# baseline only")
        
        pairs = discover_benchmarks(chapter_dir)
        
        # Should not find pairs without optimized files
        assert len(pairs) == 0
