"""Unit tests for metrics_extractor module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from common.python.metrics_extractor import (
    NsysMetrics,
    NcuMetrics,
    extract_nsys_metrics,
    extract_ncu_metrics,
    get_ncu_metric_description,
    _parse_nsys_csv,
    _parse_ncu_csv,
)


class TestNsysMetrics:
    """Tests for NsysMetrics dataclass."""
    
    def test_nsys_metrics_to_dict(self):
        """Test conversion to dictionary."""
        metrics = NsysMetrics(
            total_gpu_time_ms=123.45,
            raw_metrics={"kernel_time": 100.0, "memory_throughput": 50.0}
        )
        result = metrics.to_dict()
        
        assert result["nsys_total_gpu_time_ms"] == 123.45
        assert result["nsys_kernel_time"] == 100.0
        assert result["nsys_memory_throughput"] == 50.0
    
    def test_nsys_metrics_empty(self):
        """Test empty metrics."""
        metrics = NsysMetrics()
        result = metrics.to_dict()
        
        assert len(result) == 0


class TestNcuMetrics:
    """Tests for NcuMetrics dataclass."""
    
    def test_ncu_metrics_to_dict(self):
        """Test conversion to dictionary."""
        metrics = NcuMetrics(
            kernel_time_ms=10.5,
            sm_throughput_pct=85.0,
            dram_throughput_pct=60.0,
            l2_throughput_pct=70.0,
            occupancy_pct=90.0,
            raw_metrics={"tensor_cores": 100.0}
        )
        result = metrics.to_dict()
        
        assert result["ncu_kernel_time_ms"] == 10.5
        assert result["ncu_sm_throughput_pct"] == 85.0
        assert result["ncu_dram_throughput_pct"] == 60.0
        assert result["ncu_l2_throughput_pct"] == 70.0
        assert result["ncu_occupancy_pct"] == 90.0
        assert result["ncu_tensor_cores"] == 100.0
    
    def test_ncu_metrics_empty(self):
        """Test empty metrics."""
        metrics = NcuMetrics()
        result = metrics.to_dict()
        
        assert len(result) == 0


class TestParseNsysCsv:
    """Tests for nsys CSV parsing."""
    
    def test_parse_nsys_csv_with_total_gpu_time(self):
        """Test parsing nsys CSV with total GPU time."""
        csv_text = "Metric,Value\nTotal GPU Time,123.45"
        result = _parse_nsys_csv(csv_text)
        
        assert "nsys_total_gpu_time_ms" in result
        assert result["nsys_total_gpu_time_ms"] == 123.45
    
    def test_parse_nsys_csv_empty(self):
        """Test parsing empty CSV."""
        result = _parse_nsys_csv("")
        assert len(result) == 0
    
    def test_parse_nsys_csv_no_match(self):
        """Test parsing CSV without matching pattern."""
        csv_text = "Metric,Value\nSome Other Metric,100"
        result = _parse_nsys_csv(csv_text)
        assert len(result) == 0


class TestParseNcuCsv:
    """Tests for ncu CSV parsing."""
    
    def test_parse_ncu_csv_simple(self):
        """Test parsing simple ncu CSV."""
        csv_text = '"gpu__time_duration.avg","100.5"\n"sm__throughput.avg.pct_of_peak_sustained_elapsed","85.0"'
        result = _parse_ncu_csv(csv_text)
        
        assert "gpu__time_duration.avg" in result
        assert result["gpu__time_duration.avg"] == 100.5
        assert "sm__throughput.avg.pct_of_peak_sustained_elapsed" in result
        assert result["sm__throughput.avg.pct_of_peak_sustained_elapsed"] == 85.0
    
    def test_parse_ncu_csv_empty(self):
        """Test parsing empty CSV."""
        result = _parse_ncu_csv("")
        assert len(result) == 0
    
    def test_parse_ncu_csv_malformed(self):
        """Test parsing malformed CSV."""
        csv_text = "not,a,valid,csv"
        result = _parse_ncu_csv(csv_text)
        # Should handle gracefully without crashing
        assert isinstance(result, dict)


class TestGetNcuMetricDescription:
    """Tests for metric description lookup."""
    
    def test_get_known_metric_description(self):
        """Test getting description for known metric."""
        desc = get_ncu_metric_description("gpu__time_duration.avg")
        assert desc == "Kernel Execution Time"
    
    def test_get_unknown_metric_description(self):
        """Test getting description for unknown metric."""
        desc = get_ncu_metric_description("unknown_metric_id")
        # Should return cleaned version
        assert isinstance(desc, str)
        assert len(desc) > 0
    
    def test_get_clean_metric_name(self):
        """Test getting description for clean metric name."""
        desc = get_ncu_metric_description("ncu_sm_throughput_pct")
        # Should match to known metric
        assert "throughput" in desc.lower() or "SM" in desc


class TestExtractNsysMetrics:
    """Tests for nsys metrics extraction."""
    
    @patch('subprocess.run')
    def test_extract_nsys_metrics_success(self, mock_run):
        """Test successful nsys metrics extraction."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Metric,Value\nTotal GPU Time,123.45"
        )
        
        nsys_path = Path("/tmp/test.nsys-rep")
        nsys_path.touch()  # Create file
        
        metrics = extract_nsys_metrics(nsys_path)
        
        assert metrics.total_gpu_time_ms == 123.45
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_extract_nsys_metrics_file_not_found(self, mock_run):
        """Test extraction when file doesn't exist."""
        nsys_path = Path("/tmp/nonexistent.nsys-rep")
        
        metrics = extract_nsys_metrics(nsys_path)
        
        assert metrics.total_gpu_time_ms is None
        mock_run.assert_not_called()
    
    @patch('subprocess.run')
    def test_extract_nsys_metrics_timeout(self, mock_run):
        """Test extraction when subprocess times out."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("nsys", 60)
        
        nsys_path = Path("/tmp/test.nsys-rep")
        nsys_path.touch()
        
        metrics = extract_nsys_metrics(nsys_path)
        
        assert metrics.total_gpu_time_ms is None


class TestExtractNcuMetrics:
    """Tests for ncu metrics extraction."""
    
    @patch('subprocess.run')
    def test_extract_ncu_metrics_success(self, mock_run):
        """Test successful ncu metrics extraction."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='"gpu__time_duration.avg","10.5"\n"sm__throughput.avg.pct_of_peak_sustained_elapsed","85.0"'
        )
        
        ncu_path = Path("/tmp/test.ncu-rep")
        ncu_path.touch()
        
        metrics = extract_ncu_metrics(ncu_path)
        
        assert metrics.kernel_time_ms == 10.5
        assert metrics.sm_throughput_pct == 85.0
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_extract_ncu_metrics_file_not_found(self, mock_run):
        """Test extraction when file doesn't exist."""
        ncu_path = Path("/tmp/nonexistent.ncu-rep")
        
        metrics = extract_ncu_metrics(ncu_path)
        
        assert metrics.kernel_time_ms is None
        mock_run.assert_not_called()
    
    @patch('subprocess.run')
    def test_extract_ncu_metrics_companion_csv(self, mock_run):
        """Test extraction from companion CSV file."""
        ncu_path = Path("/tmp/test.ncu-rep")
        ncu_path.touch()
        
        csv_path = Path("/tmp/test.csv")
        csv_path.write_text('"gpu__time_duration.avg","20.0"')
        
        metrics = extract_ncu_metrics(ncu_path)
        
        # Should read from companion CSV
        assert metrics.kernel_time_ms == 20.0

