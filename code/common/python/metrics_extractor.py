"""Metrics extraction from profiling tools (nsys, ncu).

Provides typed classes and functions for extracting metrics from nsys and ncu reports.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class NsysMetrics:
    """Extracted metrics from nsys profiling report."""
    
    total_gpu_time_ms: Optional[float] = None
    raw_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        result = {}
        if self.total_gpu_time_ms is not None:
            result["nsys_total_gpu_time_ms"] = self.total_gpu_time_ms
        result.update({f"nsys_{k}": v for k, v in self.raw_metrics.items()})
        return result


@dataclass
class NcuMetrics:
    """Extracted metrics from ncu profiling report."""
    
    kernel_time_ms: Optional[float] = None
    sm_throughput_pct: Optional[float] = None
    dram_throughput_pct: Optional[float] = None
    l2_throughput_pct: Optional[float] = None
    occupancy_pct: Optional[float] = None
    raw_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        result = {}
        if self.kernel_time_ms is not None:
            result["ncu_kernel_time_ms"] = self.kernel_time_ms
        if self.sm_throughput_pct is not None:
            result["ncu_sm_throughput_pct"] = self.sm_throughput_pct
        if self.dram_throughput_pct is not None:
            result["ncu_dram_throughput_pct"] = self.dram_throughput_pct
        if self.l2_throughput_pct is not None:
            result["ncu_l2_throughput_pct"] = self.l2_throughput_pct
        if self.occupancy_pct is not None:
            result["ncu_occupancy_pct"] = self.occupancy_pct
        result.update({f"ncu_{k}": v for k, v in self.raw_metrics.items()})
        return result


# Mapping of metric identifiers to natural language descriptions
NCU_METRIC_DESCRIPTIONS = {
    "gpu__time_duration.avg": "Kernel Execution Time",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM Compute Throughput (% of peak)",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "DRAM/HBM Memory Throughput (% of peak)",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "L2 Cache Throughput (% of peak)",
    "sm__sass_thread_inst_executed_op_fp32_pred_on.sum": "FP32 Instructions Executed (compute proxy)",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "Achieved Occupancy (% active warps)",
    "dram__sectors_read.sum": "DRAM Sectors Read",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "L1 Global Memory Load Sectors",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": "L1 Global Memory Store Sectors",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "Shared Memory Bank Conflicts",
    "sm__inst_executed_pipe_tensor.sum": "Tensor Core Instructions Executed",
}


def get_ncu_metric_description(metric_key: str, fallback_to_key: bool = True) -> str:
    """Get natural language description for a metric key.
    
    Args:
        metric_key: The metric identifier (cryptic ID or clean name)
        fallback_to_key: If True, return the key itself if no description found
    
    Returns:
        Natural language description, or the key itself if not found and fallback_to_key=True
    """
    # First check if it's directly in our mapping
    if metric_key in NCU_METRIC_DESCRIPTIONS:
        return NCU_METRIC_DESCRIPTIONS[metric_key]
    
    # Try to find matching cryptic ID
    clean_key = metric_key.replace("ncu_", "").replace("_pct", "").replace("_ms", "")
    for cryptic_id, description in NCU_METRIC_DESCRIPTIONS.items():
        cryptic_parts = cryptic_id.replace("__", "_").replace(".", "_").split("_")
        key_parts = clean_key.split("_")
        
        # Check if significant parts match
        if len(set(cryptic_parts) & set(key_parts)) >= 2:
            return description
        if cryptic_id.replace("__", "_").replace(".", "_") in clean_key or clean_key in cryptic_id.replace("__", "_").replace(".", "_"):
            return description
    
    # If no match found and fallback is enabled, return a cleaned version of the key
    if fallback_to_key:
        cleaned = metric_key.replace("ncu_", "").replace("__", " ").replace("_", " ").replace(".", " ")
        return cleaned.title()
    
    return metric_key


def extract_nsys_metrics(nsys_rep_path: Path, timeout: int = 60) -> NsysMetrics:
    """Extract metrics from nsys report file.
    
    Args:
        nsys_rep_path: Path to .nsys-rep file
        timeout: Timeout for nsys stats command in seconds
        
    Returns:
        NsysMetrics object with extracted metrics
    """
    metrics = NsysMetrics()
    
    if not nsys_rep_path.exists():
        return metrics
    
    # Try using nsys stats command
    try:
        result = subprocess.run(
            ["nsys", "stats", "--report", "cuda_gpu_sum", "--format", "csv", str(nsys_rep_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            csv_metrics = _parse_nsys_csv(result.stdout)
            if "nsys_total_gpu_time_ms" in csv_metrics:
                metrics.total_gpu_time_ms = csv_metrics["nsys_total_gpu_time_ms"]
            # Store other metrics in raw_metrics
            for k, v in csv_metrics.items():
                if k != "nsys_total_gpu_time_ms":
                    clean_key = k.replace("nsys_", "")
                    metrics.raw_metrics[clean_key] = v
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Also try using the extract_nsys_summary module if available
    try:
        repo_root = Path(__file__).parent.parent.parent.parent
        tools_path = repo_root / "tools" / "profiling"
        if str(tools_path) not in sys.path:
            sys.path.insert(0, str(tools_path))
        
        from extract_nsys_summary import harvest
        harvested = harvest(nsys_rep_path)
        
        # Convert harvested metrics to dict format
        for entry in harvested:
            metric_name = entry.get("metric", "")
            value_str = entry.get("value", "")
            if metric_name and value_str:
                try:
                    value = float(value_str.replace(",", "").replace("%", ""))
                    clean_name = metric_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                    metrics.raw_metrics[clean_name] = value
                except (ValueError, AttributeError):
                    pass
    except (ImportError, Exception):
        pass
    
    return metrics


def extract_ncu_metrics(ncu_rep_path: Path, timeout: int = 60) -> NcuMetrics:
    """Extract metrics from ncu report file.
    
    Args:
        ncu_rep_path: Path to .ncu-rep file
        timeout: Timeout for ncu command in seconds
        
    Returns:
        NcuMetrics object with extracted metrics
    """
    metrics = NcuMetrics()
    
    if not ncu_rep_path.exists():
        return metrics
    
    # Try using ncu CLI to export metrics
    try:
        result = subprocess.run(
            ["ncu", "--csv", "--page", "details", "--import", str(ncu_rep_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            csv_metrics = _parse_ncu_csv(result.stdout)
            _populate_ncu_metrics(metrics, csv_metrics)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Also check for companion CSV file
    companion_csv = ncu_rep_path.with_suffix(".csv")
    if companion_csv.exists():
        try:
            csv_text = companion_csv.read_text()
            csv_metrics = _parse_ncu_csv(csv_text)
            _populate_ncu_metrics(metrics, csv_metrics)
        except Exception:
            pass
    
    return metrics


def _parse_nsys_csv(csv_text: str) -> Dict[str, float]:
    """Parse nsys CSV output for timing metrics.
    
    Args:
        csv_text: CSV text from nsys stats command
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    # Extract total GPU time
    match = re.search(r"Total GPU Time.*?,(\d+\.?\d*)", csv_text, re.IGNORECASE)
    if match:
        try:
            metrics["nsys_total_gpu_time_ms"] = float(match.group(1))
        except (ValueError, IndexError):
            pass
    
    return metrics


def _parse_ncu_csv(csv_text: str) -> Dict[str, float]:
    """Parse ncu CSV output for comprehensive roofline and performance metrics.
    
    Args:
        csv_text: CSV text from ncu export or companion CSV file
        
    Returns:
        Dictionary of metric identifiers to values
    """
    metrics = {}
    
    # Parse CSV lines
    lines = csv_text.strip().split("\n")
    if len(lines) < 2:
        return metrics
    
    # Find header row (usually first line)
    header = lines[0].split(",")
    
    # Find data rows
    for line in lines[1:]:
        if not line.strip():
            continue
        
        values = line.split(",")
        if len(values) != len(header):
            continue
        
        # Map header to values
        for i, col_name in enumerate(header):
            col_name = col_name.strip().strip('"')
            if i < len(values):
                value_str = values[i].strip().strip('"')
                if value_str and col_name:
                    try:
                        # Try to parse as float
                        value = float(value_str)
                        # Use metric ID as key
                        clean_name = col_name.replace(" ", "_").lower()
                        metrics[col_name] = value  # Keep original ID
                        metrics[f"ncu_{clean_name}"] = value  # Also add clean name
                    except ValueError:
                        pass
    
    return metrics


def _populate_ncu_metrics(metrics: NcuMetrics, csv_metrics: Dict[str, float]) -> None:
    """Populate NcuMetrics object from parsed CSV metrics.
    
    Args:
        metrics: NcuMetrics object to populate
        csv_metrics: Dictionary of metric identifiers to values
    """
    # Map known metric IDs to fields
    if "gpu__time_duration.avg" in csv_metrics:
        metrics.kernel_time_ms = csv_metrics["gpu__time_duration.avg"]
    
    if "sm__throughput.avg.pct_of_peak_sustained_elapsed" in csv_metrics:
        metrics.sm_throughput_pct = csv_metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
    
    if "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed" in csv_metrics:
        metrics.dram_throughput_pct = csv_metrics["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"]
    
    if "lts__throughput.avg.pct_of_peak_sustained_elapsed" in csv_metrics:
        metrics.l2_throughput_pct = csv_metrics["lts__throughput.avg.pct_of_peak_sustained_elapsed"]
    
    if "sm__warps_active.avg.pct_of_peak_sustained_active" in csv_metrics:
        metrics.occupancy_pct = csv_metrics["sm__warps_active.avg.pct_of_peak_sustained_active"]
    
    # Store all other metrics in raw_metrics
    for key, value in csv_metrics.items():
        if key not in [
            "gpu__time_duration.avg",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]:
            clean_key = key.replace("ncu_", "") if key.startswith("ncu_") else key
            metrics.raw_metrics[clean_key] = value

