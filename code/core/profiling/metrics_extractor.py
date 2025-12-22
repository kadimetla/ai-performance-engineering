"""Metrics extraction from profiling tools (nsys, ncu, proton, torch).

Provides functions for extracting metrics from profiling reports and returning Pydantic models.
"""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Any, Type

try:
    from core.benchmark.models import NsysMetrics, NcuMetrics, TorchMetrics, ProtonMetrics
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    NsysMetrics: Any = None  # type: ignore[no-redef]
    NcuMetrics: Any = None  # type: ignore[no-redef]
    TorchMetrics: Any = None  # type: ignore[no-redef]
    ProtonMetrics: Any = None  # type: ignore[no-redef]


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

NSYS_METRIC_KEY_MAP = {
    "total_gpu_time": ["nsys_total_gpu_time_ms", "nsys_total_gpu_time"],
    "kernel_time": ["nsys_kernel_time_ms", "nsys_kernel_time"],
    "memory_throughput_gb_per_s": ["nsys_memory_throughput_gb_per_s"],
    "compute_throughput_gflops": ["nsys_compute_throughput_gflops"],
    "memory_bandwidth_utilization": ["nsys_memory_bandwidth_utilization_pct"],
    "compute_utilization": ["nsys_compute_utilization_pct"],
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


def extract_nsys_metrics(nsys_rep_path: Path, timeout: int = 180) -> NsysMetrics:
    """Extract metrics from nsys report file.
    
    Args:
        nsys_rep_path: Path to .nsys-rep file
        timeout: Timeout for nsys stats command in seconds
        
    Returns:
        NsysMetrics (Pydantic) object with extracted metrics
    """
    if not PYDANTIC_AVAILABLE or NsysMetrics is None:
        raise ImportError("pydantic and NsysMetrics are required for extract_nsys_metrics")
    
    if not nsys_rep_path.exists():
        return NsysMetrics(total_gpu_time_ms=None, raw_metrics={}, schemaVersion="1.0")
    
    total_gpu_time_ms = None
    raw_metrics = {}
    
    # Try using nsys stats command
    try:
        result = subprocess.run(
            [
                "nsys",
                "stats",
                "--force-export=true",
                "--report",
                "cuda_gpu_sum",
                "--format",
                "csv",
                str(nsys_rep_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            csv_metrics = _parse_nsys_csv(result.stdout)
            if "nsys_total_gpu_time_ms" in csv_metrics:
                total_gpu_time_ms = csv_metrics["nsys_total_gpu_time_ms"]
            # Store other metrics in raw_metrics
            for k, v in csv_metrics.items():
                if k != "nsys_total_gpu_time_ms":
                    clean_key = k.replace("nsys_", "")
                    raw_metrics[clean_key] = v
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Also try using the extract_nsys_summary module if available
    try:
        repo_root = Path(__file__).resolve().parent.parent
        profiling_path = repo_root / "profiling"
        if str(profiling_path) not in sys.path:
            sys.path.insert(0, str(profiling_path))
        
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
                    raw_metrics[clean_name] = value
                except (ValueError, AttributeError):
                    pass
    except (ImportError, SystemExit, Exception):
        pass
    
    return NsysMetrics(total_gpu_time_ms=total_gpu_time_ms, raw_metrics=raw_metrics, schemaVersion="1.0")


def extract_ncu_metrics(ncu_rep_path: Path, timeout: int = 300) -> NcuMetrics:
    """Extract metrics from ncu report file.
    
    Args:
        ncu_rep_path: Path to .ncu-rep file
        timeout: Timeout for ncu command in seconds
        
    Returns:
        NcuMetrics (Pydantic) object with extracted metrics
    """
    if not PYDANTIC_AVAILABLE or NcuMetrics is None:
        raise ImportError("pydantic and NcuMetrics are required for extract_ncu_metrics")
    
    if not ncu_rep_path.exists():
        return NcuMetrics(
            kernel_time_ms=None,
            sm_throughput_pct=None,
            dram_throughput_pct=None,
            l2_throughput_pct=None,
            occupancy_pct=None,
            raw_metrics={},
            schemaVersion="1.0"
        )
    
    kernel_time_ms = None
    sm_throughput_pct = None
    dram_throughput_pct = None
    l2_throughput_pct = None
    occupancy_pct = None
    raw_metrics = {}
    
    # Try using ncu CLI to export metrics
    try:
        # Use --page details (Metric Name/Unit/Value rows) so we can honor units
        # while keeping output small via --metrics filtering.
        metrics = [
            "gpu__time_duration.avg",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
        result = subprocess.run(
            ["ncu", "--csv", "--page", "details", "--metrics", ",".join(metrics), "--import", str(ncu_rep_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            csv_metrics = _parse_ncu_csv(result.stdout)
            kernel_time_ms, sm_throughput_pct, dram_throughput_pct, l2_throughput_pct, occupancy_pct, raw_metrics = _populate_ncu_metrics(csv_metrics)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Also check for companion CSV file
    companion_csv = ncu_rep_path.with_suffix(".csv")
    if companion_csv.exists():
        try:
            csv_text = companion_csv.read_text()
            csv_metrics = _parse_ncu_csv(csv_text)
            kt, sm, dram, l2, occ, raw = _populate_ncu_metrics(csv_metrics)
            if kernel_time_ms is None and kt is not None:
                kernel_time_ms = kt
            if sm_throughput_pct is None and sm is not None:
                sm_throughput_pct = sm
            if dram_throughput_pct is None and dram is not None:
                dram_throughput_pct = dram
            if l2_throughput_pct is None and l2 is not None:
                l2_throughput_pct = l2
            if occupancy_pct is None and occ is not None:
                occupancy_pct = occ
            raw_metrics.update(raw)
        except (ValueError, KeyError, OSError):
            pass  # CSV parsing failed or file error
    
    return NcuMetrics(
        kernel_time_ms=kernel_time_ms,
        sm_throughput_pct=sm_throughput_pct,
        dram_throughput_pct=dram_throughput_pct,
        l2_throughput_pct=l2_throughput_pct,
        occupancy_pct=occupancy_pct,
        raw_metrics=raw_metrics,
        schemaVersion="1.0"
    )


def extract_proton_metrics(report_path: Path) -> ProtonMetrics:
    """Extract Proton kernel summaries from a JSON report."""
    if not PYDANTIC_AVAILABLE or ProtonMetrics is None:
        raise ImportError("pydantic and ProtonMetrics are required for extract_proton_metrics")
    
    if not report_path.exists():
        return ProtonMetrics(
            kernel_count=None,
            occupancy_limited_kernels=[],
            summary_stats={},
            kernel_summaries=[],
            schemaVersion="1.0",
        )
    
    try:
        data = json.loads(report_path.read_text())
    except Exception:
        # Return minimal object with parse failure note
        return ProtonMetrics(
            kernel_count=None,
            occupancy_limited_kernels=[],
            summary_stats={"parse_error": 1.0},
            kernel_summaries=[],
            schemaVersion="1.0",
        )
    
    kernel_entries = []
    if isinstance(data, dict):
        for key in ("kernels", "kernel_reports", "results"):
            if key in data and isinstance(data[key], list):
                kernel_entries = data[key]
                break
        if not kernel_entries and "data" in data and isinstance(data["data"], list):
            kernel_entries = data["data"]
    elif isinstance(data, list):
        kernel_entries = data
    
    summaries = []
    occupancy_limited: list[str] = []
    summary_stats: Dict[str, float] = {}
    
    def _maybe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    
    for entry in kernel_entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("kernel") or entry.get("kernel_name")
        regs = entry.get("registers_per_thread") or entry.get("regs_per_thread") or entry.get("registers")
        smem = entry.get("shared_memory_bytes") or entry.get("shared_memory") or entry.get("smem_bytes")
        blocks_per_sm = entry.get("blocks_per_sm") or entry.get("cta_per_sm")
        occupancy = entry.get("occupancy_pct") or entry.get("theoretical_occupancy") or entry.get("occupancy")
        time_ms = entry.get("time_ms") or entry.get("duration_ms") or entry.get("total_time_ms")
        tma_desc = entry.get("tma_descriptors") or entry.get("tma") or entry.get("mma_tma")
        
        reg_f = _maybe_float(regs)
        smem_f = _maybe_float(smem)
        occupancy_f = _maybe_float(occupancy)
        blocks_f = _maybe_float(blocks_per_sm)
        time_f = _maybe_float(time_ms)
        
        summaries.append(
            {
                "name": name,
                "regs_per_thread": reg_f,
                "shared_mem_bytes": smem_f,
                "blocks_per_sm": blocks_f,
                "occupancy_pct": occupancy_f,
                "time_ms": time_f,
                "tma_descriptors": tma_desc if isinstance(tma_desc, (int, float, str)) else None,
            }
        )
        
        if occupancy_f is not None and occupancy_f < 40.0:
            occupancy_limited.append(name or "unknown_kernel")
    
    if summaries:
        regs_max = max((s.get("regs_per_thread") for s in summaries if s.get("regs_per_thread") is not None), default=None)
        smem_max = max((s.get("shared_mem_bytes") for s in summaries if s.get("shared_mem_bytes") is not None), default=None)
        blocks_max = max((s.get("blocks_per_sm") for s in summaries if s.get("blocks_per_sm") is not None), default=None)
        time_max = max((s.get("time_ms") for s in summaries if s.get("time_ms") is not None), default=None)
        if regs_max is not None:
            summary_stats["max_regs_per_thread"] = regs_max
        if smem_max is not None:
            summary_stats["max_shared_mem_bytes"] = smem_max
        if blocks_max is not None:
            summary_stats["max_blocks_per_sm"] = blocks_max
        if time_max is not None:
            summary_stats["max_time_ms"] = time_max
    
    return ProtonMetrics(
        kernel_count=len(summaries),
        occupancy_limited_kernels=occupancy_limited,
        summary_stats=summary_stats,
        kernel_summaries=summaries,
        schemaVersion="1.0",
    )


def _parse_nsys_csv(csv_text: str) -> Dict[str, float]:
    """Parse nsys CSV output for timing and bandwidth metrics.
    
    NSYS CSV format has header row: "Metric,Value"
    Example:
        Metric,Value
        Total GPU Time,1234.56
        Memory Throughput GB/s,500.25
    
    Args:
        csv_text: CSV text from nsys stats command
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        return metrics

    # Format A (legacy): "Metric,Value" with rows like "Total GPU Time,123.45"
    header = [c.strip() for c in lines[0].split(",")]
    if len(header) >= 2 and header[0].strip().lower() == "metric":
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            metric_name = parts[0]
            value_str = parts[1]
            if not metric_name or not value_str:
                continue
            try:
                value = float(value_str)
            except ValueError:
                continue
            clean_name = (
                metric_name.lower()
                .replace(" ", "_")
                .replace("/", "_per_")
                .replace("(", "")
                .replace(")", "")
            )
            mapped_keys = NSYS_METRIC_KEY_MAP.get(clean_name)
            if not mapped_keys:
                continue
            for target in mapped_keys:
                metrics[target] = value
        return metrics

    # Format B (current): cuda_gpu_sum table CSV with header like:
    # "Time (%),Total Time (ns),Instances,...,Category,Operation"
    header_idx = None
    for idx, line in enumerate(lines):
        if "Total Time (ns)" in line and "Category" in line and "Operation" in line:
            header_idx = idx
            break
    if header_idx is None:
        return metrics

    table_lines = lines[header_idx:]
    try:
        reader = csv.DictReader(io.StringIO("\n".join(table_lines)))
        total_time_ns = 0.0
        kernel_time_ns = 0.0
        for row in reader:
            try:
                time_ns = float(row.get("Total Time (ns)", "") or 0.0)
            except ValueError:
                continue
            total_time_ns += time_ns
            if (row.get("Category") or "").strip() == "CUDA_KERNEL":
                kernel_time_ns += time_ns
        if total_time_ns > 0:
            metrics["nsys_total_gpu_time_ms"] = total_time_ns / 1e6
        if kernel_time_ns > 0:
            metrics["nsys_kernel_time_ms"] = kernel_time_ns / 1e6
    except Exception:
        return metrics
    
    return metrics


def _parse_ncu_csv(csv_text: str) -> Dict[str, float]:
    """Parse ncu CSV output for comprehensive roofline and performance metrics.
    
    NCU CSV format is "metric,value" per line with no header.
    Example:
        "gpu__time_duration.avg","10.500"
        "sm__throughput.avg.pct_of_peak_sustained_elapsed","85.25"
    
    Args:
        csv_text: CSV text from ncu export or companion CSV file
        
    Returns:
        Dictionary of metric identifiers to values
    """
    metrics: Dict[str, float] = {}

    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        return metrics

    # Format A (legacy): "metric","value" per line (no header)
    if len(lines) >= 1 and lines[0].count(",") == 1 and lines[0].startswith('"') and lines[0].endswith('"'):
        for line in lines:
            try:
                row = next(csv.reader(io.StringIO(line)))
            except Exception:
                continue
            if len(row) < 2:
                continue
            metric_name = (row[0] or "").strip()
            value_str = (row[1] or "").strip()
            if not metric_name or not value_str:
                continue
            try:
                metrics[metric_name] = float(value_str)
            except ValueError:
                continue
        return metrics

    def _parse_float(text: str) -> Optional[float]:
        stripped = (text or "").strip().replace(",", "")
        if not stripped:
            return None
        if stripped.endswith("%"):
            stripped = stripped[:-1]
        match = re.match(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", stripped)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _time_to_ms(value: float, unit: str) -> float:
        u = (unit or "").strip().lower()
        if u.endswith("ns"):
            return value / 1e6
        if u.endswith("us"):
            return value / 1e3
        if u.endswith("ms"):
            return value
        if u.endswith("s"):
            return value * 1e3
        # Nsight Compute CSV usually uses base units; treat unknown as-is.
        return value

    # Format B (current): header + rows (ncu --page details/raw --import ...)
    try:
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
    except Exception:
        return metrics

    fieldnames = reader.fieldnames or []
    has_metric_name = "Metric Name" in fieldnames and "Metric Value" in fieldnames

    # B1) Details page: one row per (kernel, metric) with units.
    if has_metric_name:
        per_kernel: Dict[str, Dict[str, float]] = {}
        per_kernel_time: Dict[str, float] = {}

        for row in reader:
            kernel_id = (row.get("ID") or "").strip()
            if not kernel_id.isdigit():
                continue
            metric_name = (row.get("Metric Name") or "").strip()
            metric_value_raw = (row.get("Metric Value") or "").strip()
            if not metric_name or not metric_value_raw:
                continue
            value = _parse_float(metric_value_raw)
            if value is None:
                continue
            unit = (row.get("Metric Unit") or "").strip()
            if metric_name.startswith("gpu__time_duration"):
                value = _time_to_ms(value, unit)
            per_kernel.setdefault(kernel_id, {})[metric_name] = value
            if metric_name == "gpu__time_duration.avg":
                per_kernel_time[kernel_id] = value

        if not per_kernel:
            return metrics

        # Choose the kernel with the highest avg duration (dominant kernel).
        best_kernel_id = max(per_kernel_time, key=per_kernel_time.get) if per_kernel_time else sorted(per_kernel.keys())[0]
        metrics.update(per_kernel[best_kernel_id])
        return metrics

    # B2) Raw page: one row per kernel with metrics as columns (no units).
    best_row: Optional[Dict[str, str]] = None
    best_time = -1.0
    for row in reader:
        row_id = (row.get("ID") or "").strip()
        if not row_id.isdigit():
            continue
        time_val = _parse_float((row.get("gpu__time_duration.avg") or "").strip() if row.get("gpu__time_duration.avg") else "")
        time_val_num = time_val if time_val is not None else 0.0
        if best_row is None or time_val_num > best_time:
            best_row = row
            best_time = time_val_num

    if best_row is None:
        return metrics

    for key, value_str in best_row.items():
        if not key:
            continue
        value = _parse_float(str(value_str))
        if value is None:
            continue
        # Heuristic: gpu__time_duration.* is usually printed in microseconds in CSV mode.
        if key.startswith("gpu__time_duration"):
            value = value / 1e3
        metrics[key] = value

    return metrics


def _populate_ncu_metrics(csv_metrics: Dict[str, float]) -> tuple:
    """Extract NcuMetrics fields from parsed CSV metrics.
    
    Args:
        csv_metrics: Dictionary of metric identifiers to values
        
    Returns:
        Tuple of (kernel_time_ms, sm_throughput_pct, dram_throughput_pct, l2_throughput_pct, occupancy_pct, raw_metrics)
    """
    kernel_time_ms = None
    sm_throughput_pct = None
    dram_throughput_pct = None
    l2_throughput_pct = None
    occupancy_pct = None
    raw_metrics = {}
    recognized_keys = set()
    
    # Map known metric IDs to fields
    kernel_key = "gpu__time_duration.avg"
    if kernel_key in csv_metrics:
        kernel_time_ms = csv_metrics[kernel_key]
        recognized_keys.add(kernel_key)
    
    sm_key = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    if sm_key in csv_metrics:
        sm_throughput_pct = csv_metrics[sm_key]
        recognized_keys.add(sm_key)
    
    dram_keys = [
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ]
    for key in dram_keys:
        if key in csv_metrics:
            dram_throughput_pct = csv_metrics[key]
            recognized_keys.add(key)
            break
    
    l2_keys = [
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "l2__throughput.avg.pct_of_peak_sustained_elapsed",
    ]
    for key in l2_keys:
        if key in csv_metrics:
            l2_throughput_pct = csv_metrics[key]
            recognized_keys.add(key)
            break
    
    occupancy_key = "sm__warps_active.avg.pct_of_peak_sustained_active"
    if occupancy_key in csv_metrics:
        occupancy_pct = csv_metrics[occupancy_key]
        recognized_keys.add(occupancy_key)
    
    # Store all other metrics in raw_metrics
    for key, value in csv_metrics.items():
        if key in recognized_keys:
            continue
        clean_key = key.replace("ncu_", "") if key.startswith("ncu_") else key
        raw_metrics[clean_key] = value
    
    return (kernel_time_ms, sm_throughput_pct, dram_throughput_pct, l2_throughput_pct, occupancy_pct, raw_metrics)
