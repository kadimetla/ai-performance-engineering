"""Unit tests for benchmark discovery functionality.

Tests discovery of Python and CUDA benchmark pairs as specified in Part 2.10.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add repo root to path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.discovery import (
    discover_benchmarks,
    discover_cuda_benchmarks,
    discover_all_chapters,
    discover_benchmark_pairs,
)


class TestPythonBenchmarkDiscovery:
    """Test discovery of Python benchmark pairs."""
    
    def test_discover_benchmarks_finds_baseline_optimized_pairs(self, tmp_path):
        """Test that discover_benchmarks finds baseline/optimized pairs."""
        # Create a test chapter directory
        chapter_dir = tmp_path / "ch1"
        chapter_dir.mkdir()
        
        # Create baseline file
        baseline_file = chapter_dir / "baseline_attention.py"
        baseline_file.write_text("# baseline")
        
        # Create optimized file
        optimized_file = chapter_dir / "optimized_attention.py"
        optimized_file.write_text("# optimized")
        
        pairs = discover_benchmarks(chapter_dir)
        
        assert len(pairs) == 1
        baseline_path, optimized_paths, example_name = pairs[0]
        assert baseline_path.name == "baseline_attention.py"
        assert len(optimized_paths) == 1
        assert optimized_paths[0].name == "optimized_attention.py"
        assert example_name == "attention"
    
    def test_discover_benchmarks_finds_multiple_optimizations(self, tmp_path):
        """Test that discover_benchmarks finds multiple optimized variants."""
        chapter_dir = tmp_path / "ch1"
        chapter_dir.mkdir()
        
        baseline_file = chapter_dir / "baseline_moe.py"
        baseline_file.write_text("# baseline")
        
        optimized1 = chapter_dir / "optimized_moe_sparse.py"
        optimized1.write_text("# optimized1")
        
        optimized2 = chapter_dir / "optimized_moe_dense.py"
        optimized2.write_text("# optimized2")
        
        pairs = discover_benchmarks(chapter_dir)

        names = {example_name for _, _, example_name in pairs}
        assert {"moe", "moe_sparse", "moe_dense"} <= names

        primary = next(p for p in pairs if p[2] == "moe")
        baseline_path, optimized_paths, _ = primary
        assert len(optimized_paths) == 2
        assert any(p.name == "optimized_moe_sparse.py" for p in optimized_paths)
        assert any(p.name == "optimized_moe_dense.py" for p in optimized_paths)
    
    def test_discover_benchmarks_handles_no_baseline(self, tmp_path):
        """Test that discover_benchmarks handles missing baseline files."""
        chapter_dir = tmp_path / "ch1"
        chapter_dir.mkdir()
        
        # Only optimized file, no baseline
        optimized_file = chapter_dir / "optimized_attention.py"
        optimized_file.write_text("# optimized")
        
        pairs = discover_benchmarks(chapter_dir)
        
        assert len(pairs) == 0
    
    def test_discover_benchmarks_handles_no_optimized(self, tmp_path):
        """Test that discover_benchmarks handles missing optimized files."""
        chapter_dir = tmp_path / "ch1"
        chapter_dir.mkdir()
        
        # Only baseline file, no optimized
        baseline_file = chapter_dir / "baseline_attention.py"
        baseline_file.write_text("# baseline")
        
        pairs = discover_benchmarks(chapter_dir)
        
        assert len(pairs) == 0
    
    def test_discover_benchmarks_extracts_example_name_correctly(self, tmp_path):
        """Test that example name is extracted correctly from baseline filename."""
        chapter_dir = tmp_path / "ch1"
        chapter_dir.mkdir()
        
        baseline_file = chapter_dir / "baseline_speculative_decoding.py"
        baseline_file.write_text("# baseline")
        
        optimized_file = chapter_dir / "optimized_speculative_decoding.py"
        optimized_file.write_text("# optimized")
        
        pairs = discover_benchmarks(chapter_dir)
        
        assert len(pairs) == 1
        _, _, example_name = pairs[0]
        assert example_name == "speculative_decoding"
    
    def test_discover_benchmarks_real_chapter(self):
        """Test discovery on a real chapter directory if it exists."""
        ch1_dir = repo_root / "ch1"
        if not ch1_dir.exists():
            pytest.skip("ch1 directory not found")
        
        pairs = discover_benchmarks(ch1_dir)
        
        # Should find at least some pairs if chapter exists
        assert isinstance(pairs, list)
        for baseline_path, optimized_paths, example_name in pairs:
            assert baseline_path.exists()
            assert baseline_path.name.startswith("baseline_")
            assert len(optimized_paths) > 0
            for opt_path in optimized_paths:
                assert opt_path.exists()
                assert opt_path.name.startswith("optimized_")
            assert isinstance(example_name, str)
            assert len(example_name) > 0


class TestCudaBenchmarkDiscovery:
    """Test discovery of CUDA benchmarks."""
    
    def test_discover_cuda_benchmarks_finds_cu_files(self, tmp_path):
        """Test that discover_cuda_benchmarks finds .cu files."""
        ch1_dir = tmp_path / "ch1"
        ch1_dir.mkdir()
        
        cuda_file = ch1_dir / "test.cu"
        cuda_file.write_text("// CUDA code")
        
        cuda_benchmarks = discover_cuda_benchmarks(tmp_path)
        
        assert len(cuda_benchmarks) == 1
        assert cuda_benchmarks[0].name == "test.cu"
    
    def test_discover_cuda_benchmarks_finds_in_subdir(self, tmp_path):
        """Test that discover_cuda_benchmarks finds .cu files in cuda/ subdir."""
        ch1_dir = tmp_path / "ch1"
        ch1_dir.mkdir()
        
        cuda_subdir = ch1_dir / "cuda"
        cuda_subdir.mkdir()
        
        cuda_file = cuda_subdir / "test.cu"
        cuda_file.write_text("// CUDA code")
        
        cuda_benchmarks = discover_cuda_benchmarks(tmp_path)
        
        assert len(cuda_benchmarks) == 1
        assert cuda_benchmarks[0].name == "test.cu"
    
    def test_discover_cuda_benchmarks_handles_no_cuda_files(self, tmp_path):
        """Test that discover_cuda_benchmarks handles no CUDA files."""
        ch1_dir = tmp_path / "ch1"
        ch1_dir.mkdir()
        
        cuda_benchmarks = discover_cuda_benchmarks(tmp_path)
        
        assert len(cuda_benchmarks) == 0
    
    def test_discover_cuda_benchmarks_returns_sorted(self, tmp_path):
        """Test that discover_cuda_benchmarks returns sorted list."""
        ch1_dir = tmp_path / "ch1"
        ch1_dir.mkdir()
        
        cuda_file1 = ch1_dir / "z_test.cu"
        cuda_file1.write_text("// CUDA code")
        
        cuda_file2 = ch1_dir / "a_test.cu"
        cuda_file2.write_text("// CUDA code")
        
        cuda_benchmarks = discover_cuda_benchmarks(tmp_path)
        
        assert len(cuda_benchmarks) == 2
        # Should be sorted
        assert cuda_benchmarks[0].name < cuda_benchmarks[1].name


class TestChapterDiscovery:
    """Test discovery of all chapters."""
    
    def test_discover_all_chapters_finds_ch_directories(self, tmp_path):
        """Test that discover_all_chapters finds ch* directories."""
        ch1 = tmp_path / "ch1"
        ch1.mkdir()
        
        ch2 = tmp_path / "ch2"
        ch2.mkdir()
        
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        
        chapters = discover_all_chapters(tmp_path)
        
        assert len(chapters) == 2
        chapter_names = [ch.name for ch in chapters]
        assert "ch1" in chapter_names
        assert "ch2" in chapter_names
        assert "other" not in chapter_names
    
    def test_discover_all_chapters_filters_by_number(self, tmp_path):
        """Test that discover_all_chapters only finds ch* with numbers."""
        ch1 = tmp_path / "ch1"
        ch1.mkdir()
        
        ch_abc = tmp_path / "chabc"
        ch_abc.mkdir()
        
        chapters = discover_all_chapters(tmp_path)
        
        assert len(chapters) == 1
        assert chapters[0].name == "ch1"
    
    def test_discover_all_chapters_returns_sorted(self, tmp_path):
        """Test that discover_all_chapters returns sorted list."""
        ch10 = tmp_path / "ch10"
        ch10.mkdir()
        
        ch2 = tmp_path / "ch2"
        ch2.mkdir()
        
        ch1 = tmp_path / "ch1"
        ch1.mkdir()
        
        chapters = discover_all_chapters(tmp_path)
        
        assert len(chapters) == 3
        # Should be sorted
        assert chapters[0].name == "ch1"
        assert chapters[1].name == "ch2"
        assert chapters[2].name == "ch10"


class TestBenchmarkPairDiscovery:
    """Test discovery of benchmark pairs across chapters."""
    
    def test_discover_benchmark_pairs_all_chapters(self, tmp_path):
        """Test that discover_benchmark_pairs finds pairs across all chapters."""
        ch1 = tmp_path / "ch1"
        ch1.mkdir()
        
        baseline1 = ch1 / "baseline_test.py"
        baseline1.write_text("# baseline")
        optimized1 = ch1 / "optimized_test.py"
        optimized1.write_text("# optimized")
        
        ch2 = tmp_path / "ch2"
        ch2.mkdir()
        
        baseline2 = ch2 / "baseline_test.py"
        baseline2.write_text("# baseline")
        optimized2 = ch2 / "optimized_test.py"
        optimized2.write_text("# optimized")
        
        pairs = discover_benchmark_pairs(tmp_path, chapter="all")
        
        assert len(pairs) == 2
    
    def test_discover_benchmark_pairs_specific_chapter(self, tmp_path):
        """Test that discover_benchmark_pairs finds pairs in specific chapter."""
        ch1 = tmp_path / "ch1"
        ch1.mkdir()
        
        baseline1 = ch1 / "baseline_test.py"
        baseline1.write_text("# baseline")
        optimized1 = ch1 / "optimized_test.py"
        optimized1.write_text("# optimized")
        
        ch2 = tmp_path / "ch2"
        ch2.mkdir()
        
        baseline2 = ch2 / "baseline_test.py"
        baseline2.write_text("# baseline")
        optimized2 = ch2 / "optimized_test.py"
        optimized2.write_text("# optimized")
        
        pairs = discover_benchmark_pairs(tmp_path, chapter="ch1")
        
        assert len(pairs) == 1
        assert pairs[0][0].parent.name == "ch1"
    
    def test_discover_benchmark_pairs_normalizes_chapter_name(self, tmp_path):
        """Test that discover_benchmark_pairs normalizes chapter name."""
        ch1 = tmp_path / "ch1"
        ch1.mkdir()
        
        baseline1 = ch1 / "baseline_test.py"
        baseline1.write_text("# baseline")
        optimized1 = ch1 / "optimized_test.py"
        optimized1.write_text("# optimized")
        
        # Test with number only
        pairs1 = discover_benchmark_pairs(tmp_path, chapter="1")
        assert len(pairs1) == 1
        
        # Test with ch prefix
        pairs2 = discover_benchmark_pairs(tmp_path, chapter="ch1")
        assert len(pairs2) == 1
        
        # Test with just number string
        pairs3 = discover_benchmark_pairs(tmp_path, chapter="1")
        assert len(pairs3) == 1
    
    def test_discover_benchmark_pairs_handles_nonexistent_chapter(self, tmp_path):
        """Test that discover_benchmark_pairs handles nonexistent chapter."""
        pairs = discover_benchmark_pairs(tmp_path, chapter="ch999")
        
        assert len(pairs) == 0
