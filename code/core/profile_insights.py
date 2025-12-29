"""
Shared profiling insights and heuristics used across dashboard, CLI, and MCP.

These helpers stay free of HTTP/handler state and operate on plain data
structures so they can be reused by any interface.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def _extract_ncu_sources(ncu_path: Path, limit: int = 12) -> List[Dict[str, str]]:
    """
    Best-effort extraction of source/line info from an ncu-rep.
    Parses the Source page CSV and keeps the first few entries with a Source column.
    """
    try:
        result = subprocess.run(
            ["ncu", "--import", str(ncu_path), "--csv", "--page", "source"],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if result.returncode != 0:
            return []
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        kernel_name: Optional[str] = None
        header: Optional[List[str]] = None
        samples: List[Dict[str, str]] = []
        for row in csv.reader(lines):
            if not row:
                continue
            if row[0] == "Kernel Name":
                kernel_name = row[1] if len(row) > 1 else kernel_name
                header = None
                continue
            if "Source" in row and "Address" in row:
                header = row
                continue
            if header and len(row) >= len(header):
                entry = dict(zip(header, row))
                source = entry.get("Source", "").strip()
                if not source:
                    continue
                samples.append(
                    {
                        "kernel": kernel_name or "",
                        "source": source,
                        "address": entry.get("Address", "").strip(),
                        "stall": entry.get("Warp Stall Sampling (All Samples)", "").strip(),
                    }
                )
                if len(samples) >= limit:
                    break
        return samples
    except Exception:
        return []


def _extract_ncu_disassembly(ncu_path: Path, limit: int = 40) -> List[str]:
    """
    Best-effort disassembly from an ncu-rep via the Source page (SASS view).
    Returns a list of text lines (truncated) for quick inspection.
    """
    try:
        result = subprocess.run(
            ["ncu", "--import", str(ncu_path), "--csv", "--page", "source", "--print-source", "sass"],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if result.returncode != 0:
            return []
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        payload: List[str] = []
        header: Optional[List[str]] = None
        for row in csv.reader(lines):
            if not row:
                continue
            if row[0] == "Kernel Name":
                header = None
                continue
            if "Address" in row and "Source" in row:
                header = row
                continue
            if header and len(row) >= 2:
                address = row[0].strip()
                source = row[1].strip()
                if not source:
                    continue
                payload.append(f"{address} {source}".strip())
                if len(payload) >= limit:
                    break
        return payload
    except Exception:
        return []


def _extract_nsys_sources(nsys_path: Path, limit: int = 12) -> List[Dict[str, Any]]:
    """
    Best-effort extraction of source/line info from an nsys-rep by exporting trace JSON.
    Broadened to capture CPU/GPU/NVTX/driver events with any file/line/symbol hints.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="nsys_src_") as tmp:
            out_prefix = Path(tmp) / "trace"
            export_cmd = [
                "nsys",
                "export",
                "--type",
                "timeline",  # richer than stats; still JSON
                "--format",
                "json",
                "--force",
                "true",
                "--output",
                str(out_prefix),
                str(nsys_path),
            ]
            subprocess.run(export_cmd, check=True, capture_output=True, text=True, timeout=90)
            json_path = out_prefix.with_suffix(".json")
            if not json_path.exists():
                return []
            data = json.loads(json_path.read_text())
            events = data if isinstance(data, list) else data.get("traceEvents", [])
            samples: List[Dict[str, Any]] = []
            seen = set()
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                cat = (ev.get("cat") or "").lower()
                # Broader cat coverage: cuda, gpu, cpu, nvtx, driver/cu/rt, cupti
                if not any(tok in cat for tok in ("cuda", "gpu", "cpu", "nvtx", "driver", "cupti")):
                    continue
                args = ev.get("args") or {}
                file = ev.get("file") or args.get("file") or args.get("Source File")
                line = ev.get("line") or args.get("line") or args.get("Source Line")
                symbol = args.get("External Symbol") or args.get("Symbol") or ev.get("name")
                key = (file, line, symbol, ev.get("pid"), ev.get("tid"))
                if key in seen:
                    continue
                seen.add(key)
                samples.append(
                    {
                        "name": ev.get("name", ""),
                        "cat": ev.get("cat", ""),
                        "file": file,
                        "line": line,
                        "symbol": symbol,
                        "ts": ev.get("ts"),
                        "dur": ev.get("dur"),
                        "pid": ev.get("pid"),
                        "tid": ev.get("tid"),
                    }
                )
                if len(samples) >= limit:
                    break
            return samples
    except Exception:
        return []


def detect_bottlenecks(
    flame_data: Optional[Dict[str, Any]],
    kernel_data: Optional[Dict[str, Any]],
    hw_caps: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze profile data to detect performance bottlenecks."""
    flame_data = flame_data or {}
    kernel_data = kernel_data or {}
    hw_caps = hw_caps or {}

    bottlenecks: List[Dict[str, Any]] = []
    total_time = float(flame_data.get("value") or 0)
    if total_time <= 0:
        return {"bottlenecks": [], "message": "No profile data available"}

    features = hw_caps.get("features", []) or []
    feature_names = [f.get("name", "") for f in features if isinstance(f, dict)]

    # Analyze by category
    for category in flame_data.get("children", []):
        cat_name = category.get("name", "")
        cat_time = float(category.get("value") or 0)
        cat_pct = (cat_time / total_time * 100) if total_time else 0.0

        # Memory transfer bottleneck
        if cat_name in ["gpu_memcpy", "cuda_runtime"] and cat_pct > 10:
            bottlenecks.append(
                {
                    "type": "memory_transfer",
                    "severity": "high" if cat_pct > 20 else "medium",
                    "category": cat_name,
                    "time_us": cat_time,
                    "percentage": round(cat_pct, 1),
                    "description": f"Memory transfers taking {cat_pct:.1f}% of execution time",
                    "suggestions": [
                        "Use pinned memory for host-to-device transfers",
                        "Overlap transfers with compute using CUDA streams",
                        "Consider using TMA for async memory operations"
                        if any(name.startswith("TMA") for name in feature_names)
                        else "Batch small transfers into larger ones",
                        "Use cudaMemcpyAsync instead of synchronous copies",
                    ],
                    "potential_speedup": f"{cat_pct * 0.5:.0f}% reduction possible",
                }
            )

        # CPU overhead bottleneck
        if cat_name == "python_function" and cat_pct > 15:
            bottlenecks.append(
                {
                    "type": "cpu_overhead",
                    "severity": "high" if cat_pct > 30 else "medium",
                    "category": cat_name,
                    "time_us": cat_time,
                    "percentage": round(cat_pct, 1),
                    "description": f"Python/CPU overhead taking {cat_pct:.1f}% of execution time",
                    "suggestions": [
                        "Use torch.compile() to fuse operations",
                        "Enable CUDA graphs for repeated kernel launches",
                        "Reduce Python-level loops - vectorize operations",
                        "Use torch.jit.script for hot paths",
                    ],
                    "potential_speedup": f"{cat_pct * 0.7:.0f}% reduction possible",
                }
            )

        # Profiler overhead (informational)
        if cat_name == "overhead" and cat_pct > 5:
            bottlenecks.append(
                {
                    "type": "profiler_overhead",
                    "severity": "info",
                    "category": cat_name,
                    "time_us": cat_time,
                    "percentage": round(cat_pct, 1),
                    "description": f"Profiler overhead is {cat_pct:.1f}% - actual performance is better",
                    "suggestions": [
                        "This overhead is from profiling only",
                        f"Real execution will be faster by ~{cat_pct:.0f}%",
                    ],
                    "potential_speedup": "N/A - profiling artifact",
                }
            )

    # Analyze individual kernels for optimization opportunities
    kernels = kernel_data.get("kernels", []) or []
    total_kernel_time = float(kernel_data.get("summary", {}).get("total_time_us", 1) or 1)

    for kernel in kernels[:5]:  # Top 5 kernels
        k_name = kernel.get("name", "")
        k_time = float(kernel.get("time_us", 0) or 0)
        k_pct = (k_time / total_kernel_time * 100) if total_kernel_time else 0.0

        if k_pct > 15:  # Significant kernel
            suggestions: List[str] = []
            lower_name = k_name.lower()

            # Pattern-based suggestions
            if any(term in lower_name for term in ["gemm", "matmul", "mm_"]):
                suggestions.extend(
                    [
                        "Ensure matrix dimensions are multiples of 16",
                        "Consider using cuBLAS or cuBLASLt for GEMM",
                        "Try FP8 for inference if accuracy permits"
                        if (hw_caps.get("architecture") or "").lower().startswith("blackwell")
                        else "Use TF32 for faster FP32 operations",
                    ]
                )
            elif "conv" in lower_name:
                suggestions.extend(
                    [
                        "Use cuDNN with autotuning enabled",
                        "Consider channels-last memory format (NHWC)",
                        "Try depthwise-separable convolutions if applicable",
                    ]
                )
            elif any(term in lower_name for term in ["softmax", "attention", "sdpa"]):
                suggestions.extend(
                    [
                        "Use Flash Attention for memory efficiency",
                        "Consider fused attention kernels",
                        "Use scaled_dot_product_attention with efficient backend",
                    ]
                )
            elif any(term in lower_name for term in ["copy", "memcpy", "_to_"]):
                suggestions.extend(
                    [
                        "Minimize data movement between CPU and GPU",
                        "Use in-place operations where possible",
                        "Consider memory layout optimization",
                    ]
                )
            else:
                suggestions.extend(
                    [
                        "Profile with NCU for detailed metrics",
                        "Check occupancy and register usage",
                        "Consider kernel fusion opportunities",
                    ]
                )

            bottlenecks.append(
                {
                    "type": "hot_kernel",
                    "severity": "high" if k_pct > 25 else "medium",
                    "category": "kernel",
                    "kernel_name": k_name,
                    "time_us": k_time,
                    "percentage": round(k_pct, 1),
                    "description": f"Kernel '{k_name[:50]}' taking {k_pct:.1f}% of GPU time",
                    "suggestions": suggestions,
                    "potential_speedup": "Varies by optimization",
                }
            )

    severity_order = {"high": 0, "medium": 1, "info": 2}
    bottlenecks.sort(key=lambda x: severity_order.get(x.get("severity", "info"), 3))

    return {
        "bottlenecks": bottlenecks,
        "total_time_us": total_time,
        "hardware": {
            "name": hw_caps.get("gpu", {}).get("name", "Unknown"),
            "architecture": hw_caps.get("architecture", "unknown"),
            "features_available": len(features),
        },
    }


def calculate_optimization_score(
    hw_caps: Dict[str, Any], bottlenecks: Dict[str, Any], kernel_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate an optimization opportunity score based on profile analysis."""
    hw_caps = hw_caps or {}
    bottlenecks = bottlenecks or {}
    kernel_data = kernel_data or {}

    score = 100  # Start with perfect score
    deductions: List[str] = []
    quick_wins: List[Dict[str, Any]] = []
    advanced_opts: List[Dict[str, Any]] = []
    already_optimized: List[str] = []

    # Check for bottlenecks and deduct points
    for bottleneck in bottlenecks.get("bottlenecks", []) or []:
        severity = bottleneck.get("severity")
        if severity == "high":
            score -= 15
            deductions.append(f"{bottleneck.get('type')}: -15 pts")
        elif severity == "medium":
            score -= 8
            deductions.append(f"{bottleneck.get('type')}: -8 pts")

    # Check for unused hardware features
    for feature in hw_caps.get("features", []) or []:
        if not isinstance(feature, dict) or not feature.get("supported"):
            continue
        name = feature.get("name", "")
        opt_text = feature.get("optimization", "")
        if "TMA" in name:
            quick_wins.append(
                {
                    "feature": name,
                    "potential": "+15% speedup",
                    "effort": "medium",
                    "description": opt_text,
                }
            )
        elif "FP8" in name:
            quick_wins.append(
                {
                    "feature": name,
                    "potential": "+25% throughput",
                    "effort": "low",
                    "description": opt_text,
                }
            )
        elif "Cluster" in name:
            advanced_opts.append(
                {
                    "feature": name,
                    "potential": "+10% speedup",
                    "effort": "high",
                    "description": opt_text,
                }
            )
        elif "DSMEM" in name:
            advanced_opts.append(
                {
                    "feature": name,
                    "potential": "+5-15% speedup",
                    "effort": "high",
                    "description": opt_text,
                }
            )

    # Check what's already optimized (look for patterns in kernel names)
    kernels = kernel_data.get("kernels", []) or []
    kernel_names = [k.get("name", "").lower() for k in kernels]

    if any("flash" in k for k in kernel_names):
        already_optimized.append("Flash Attention enabled")
        score += 5
    if any("fused" in k for k in kernel_names):
        already_optimized.append("Kernel fusion detected")
        score += 3
    if any("cudnn" in k for k in kernel_names):
        already_optimized.append("cuDNN optimized kernels")
        score += 3

    score = max(0, min(100, score))  # Clamp to 0-100

    return {
        "score": score,
        "grade": "A"
        if score >= 90
        else "B"
        if score >= 75
        else "C"
        if score >= 60
        else "D"
        if score >= 40
        else "F",
        "deductions": deductions,
        "quick_wins": quick_wins,
        "advanced_optimizations": advanced_opts,
        "already_optimized": already_optimized,
        "summary": {
            "quick_win_count": len(quick_wins),
            "advanced_opt_count": len(advanced_opts),
            "already_optimized_count": len(already_optimized),
        },
    }


def compare_nsys_files(
    profiles_dir: Path,
    pair_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Extract and compare nsys metrics between baseline and optimized."""
    baseline_nsys = list(profiles_dir.glob("*baseline*.nsys-rep"))
    optimized_nsys = list(profiles_dir.glob("*optimized*.nsys-rep"))

    if not baseline_nsys or not optimized_nsys:
        return None

    pair, _, error = _select_profile_pair(
        baseline_nsys,
        optimized_nsys,
        pair_key=pair_key,
        label="nsys",
    )
    if error:
        return error
    if not pair:
        return None

    baseline_path, optimized_path = pair

    try:
        from core.profiling.extract_nsys_summary import harvest

        baseline_metrics = harvest(baseline_path)
        optimized_metrics = harvest(optimized_path)

        comparison = {
            "baseline_file": baseline_path.name,
            "optimized_file": optimized_path.name,
            "metrics": [],
            "baseline_sources": _extract_nsys_sources(baseline_path),
            "optimized_sources": _extract_nsys_sources(optimized_path),
        }

        opt_lookup = {m.get("metric", ""): m.get("value", "") for m in optimized_metrics}

        for base_metric in baseline_metrics:
            metric = base_metric.get("metric", "")
            baseline_val = base_metric.get("value", "")
            optimized_val = opt_lookup.get(metric, "")

            delta = None
            delta_pct = None
            try:
                b_num = float(str(baseline_val).replace(",", ""))
                o_num = float(str(optimized_val).replace(",", ""))
                delta = o_num - b_num
                delta_pct = ((o_num - b_num) / b_num * 100) if b_num != 0 else 0
            except (ValueError, TypeError):
                pass

            comparison["metrics"].append(
                {
                    "name": metric,
                    "baseline": baseline_val,
                    "optimized": optimized_val,
                    "delta": delta,
                    "delta_pct": delta_pct,
                }
            )

        return comparison
    except Exception as exc:  # pragma: no cover - best-effort helper
        return {"error": str(exc)}


def compare_ncu_files(
    profiles_dir: Path,
    pair_key: Optional[str] = None,
    include_ncu_details: bool = False,
) -> Optional[Dict[str, Any]]:
    """Extract and compare ncu metrics between baseline and optimized."""
    baseline_ncu = list(profiles_dir.glob("*baseline*.ncu-rep"))
    optimized_ncu = list(profiles_dir.glob("*optimized*.ncu-rep"))

    baseline_csv = list(profiles_dir.glob("*baseline*ncu*.csv"))
    optimized_csv = list(profiles_dir.glob("*optimized*ncu*.csv"))

    if baseline_csv and optimized_csv:
        pair, selected_key, error = _select_profile_pair(
            baseline_csv,
            optimized_csv,
            pair_key=pair_key,
            label="ncu csv",
        )
        if error:
            return error
        if not pair:
            return None

        baseline_csv_path, optimized_csv_path = pair
        selected_pair_key = pair_key or selected_key

        try:
            def read_ncu_csv(path: Path) -> Dict[str, Any]:
                metrics: Dict[str, Any] = {}
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = row.get("Metric Name", row.get("Name", ""))
                        value = row.get("Metric Value", row.get("Avg", row.get("Value", "")))
                        if name:
                            metrics[name] = value
                return metrics

            baseline_metrics = read_ncu_csv(baseline_csv_path)
            optimized_metrics = read_ncu_csv(optimized_csv_path)

            comparison = {
                "baseline_file": baseline_csv_path.name,
                "optimized_file": optimized_csv_path.name,
                "metrics": [],
            }
            if include_ncu_details and baseline_ncu and optimized_ncu:
                rep_pair, _, rep_error = _select_profile_pair(
                    baseline_ncu,
                    optimized_ncu,
                    pair_key=selected_pair_key,
                    label="ncu",
                )
                if rep_error:
                    return rep_error
                if rep_pair:
                    baseline_rep, optimized_rep = rep_pair
                    comparison["baseline_sources"] = _extract_ncu_sources(baseline_rep)
                    comparison["optimized_sources"] = _extract_ncu_sources(optimized_rep)
                    comparison["baseline_disassembly"] = _extract_ncu_disassembly(baseline_rep)
                    comparison["optimized_disassembly"] = _extract_ncu_disassembly(optimized_rep)

            all_keys = set(baseline_metrics.keys()) | set(optimized_metrics.keys())
            for key in sorted(all_keys):
                b_val = baseline_metrics.get(key, "")
                o_val = optimized_metrics.get(key, "")

                delta = None
                try:
                    b_num = float(str(b_val).replace(",", ""))
                    o_num = float(str(o_val).replace(",", ""))
                    delta = o_num - b_num
                except (ValueError, TypeError):
                    pass

                comparison["metrics"].append(
                    {"name": key, "baseline": b_val, "optimized": o_val, "delta": delta}
                )

            return comparison
        except Exception as exc:
            return {"error": str(exc)}

    if not baseline_ncu or not optimized_ncu:
        return None

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
        return value

    def _load_metrics(ncu_path: Path) -> Dict[str, Dict[str, float]]:
        metrics = [
            "gpu__time_duration.avg",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
        result = subprocess.run(
            ["ncu", "--import", str(ncu_path), "--csv", "--page", "details", "--metrics", ",".join(metrics)],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        reader = csv.DictReader(io.StringIO(result.stdout))
        per_kernel: Dict[str, Dict[str, float]] = {}
        for row in reader:
            kernel = (row.get("Kernel Name") or "").strip() or "kernel"
            metric = (row.get("Metric Name") or "").strip()
            value_raw = (row.get("Metric Value") or "").strip()
            if not metric or not value_raw:
                continue
            value = _parse_float(value_raw)
            if value is None:
                continue
            unit = (row.get("Metric Unit") or "").strip()
            if metric.startswith("gpu__time_duration"):
                value = _time_to_ms(value, unit)
            per_kernel.setdefault(kernel, {})[metric] = value
        return per_kernel

    pair, _, error = _select_profile_pair(
        baseline_ncu,
        optimized_ncu,
        pair_key=pair_key,
        label="ncu",
    )
    if error:
        return error
    if not pair:
        return None

    baseline_path, optimized_path = pair

    try:
        baseline_per_kernel = _load_metrics(baseline_path)
        optimized_per_kernel = _load_metrics(optimized_path)
    except Exception as exc:  # pragma: no cover - tool availability varies
        return {"error": f"NCU extraction failed: {exc}"}

    if not baseline_per_kernel and not optimized_per_kernel:
        return None

    # Rank kernels by baseline time if available.
    def _kernel_time(per_kernel: Dict[str, Dict[str, float]], name: str) -> float:
        try:
            return float(per_kernel.get(name, {}).get("gpu__time_duration.avg", 0.0) or 0.0)
        except Exception:
            return 0.0

    all_kernels = set(baseline_per_kernel.keys()) | set(optimized_per_kernel.keys())
    ranked = sorted(all_kernels, key=lambda k: _kernel_time(baseline_per_kernel, k), reverse=True)
    ranked = ranked[:12]

    def _kernel_payload(name: str) -> Dict[str, Any]:
        b = baseline_per_kernel.get(name, {})
        o = optimized_per_kernel.get(name, {})
        keys = set(b.keys()) | set(o.keys())
        diffs: Dict[str, Any] = {}
        for key in sorted(keys):
            bv = b.get(key)
            ov = o.get(key)
            delta = None
            ratio = None
            if isinstance(bv, (int, float)) and isinstance(ov, (int, float)):
                delta = ov - bv
                ratio = (ov / bv) if bv != 0 else None
            diffs[key] = {"baseline": bv, "optimized": ov, "delta": delta, "ratio": ratio}
        return {"kernel": name, "metrics": diffs}

    comparison = {
        "baseline_file": baseline_path.name,
        "optimized_file": optimized_path.name,
        "kernel_comparison": [_kernel_payload(k) for k in ranked],
    }
    if include_ncu_details:
        comparison["baseline_sources"] = _extract_ncu_sources(baseline_path)
        comparison["optimized_sources"] = _extract_ncu_sources(optimized_path)
        comparison["baseline_disassembly"] = _extract_ncu_disassembly(baseline_path)
        comparison["optimized_disassembly"] = _extract_ncu_disassembly(optimized_path)
    return comparison

    return None


def _extract_nsys_cuda_api_stats(nsys_path: Path) -> Dict[str, Any]:
    """Extract CUDA API summary stats from nsys report."""
    try:
        with tempfile.TemporaryDirectory(prefix="nsys_api_") as tmp_dir:
            output_prefix = Path(tmp_dir) / "report"
            result = subprocess.run(
                ["nsys", "stats", "--report", "cuda_api_sum", "--format", "csv",
                 "--force-export", "true", "-o", str(output_prefix), str(nsys_path)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                return {}
            
            # Find the generated CSV file
            csv_files = sorted(Path(tmp_dir).glob("report*cuda_api_sum.csv"))
            if not csv_files:
                return {}
            
            api_stats = {}
            with open(csv_files[0], newline='') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    name = row.get('Name', '')
                    if name:
                        try:
                            api_stats[name] = {
                                'time_pct': float(row.get('Time (%)', 0) or 0),
                                'total_time_ns': int(float(row.get('Total Time (ns)', 0) or 0)),
                                'num_calls': int(row.get('Num Calls', 0) or 0),
                                'avg_ns': float(row.get('Avg (ns)', 0) or 0),
                            }
                        except (ValueError, TypeError):
                            continue
            return api_stats
    except Exception:
        return {}


def _extract_nsys_kernel_stats(nsys_path: Path) -> Dict[str, Any]:
    """Extract GPU kernel summary stats from nsys report."""
    try:
        with tempfile.TemporaryDirectory(prefix="nsys_kern_") as tmp_dir:
            output_prefix = Path(tmp_dir) / "report"
            result = subprocess.run(
                ["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                 "--force-export", "true", "-o", str(output_prefix), str(nsys_path)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                return {}
            
            # Find the generated CSV file
            csv_files = sorted(Path(tmp_dir).glob("report*cuda_gpu_kern_sum.csv"))
            if not csv_files:
                return {}
            
            kernel_stats = {}
            with open(csv_files[0], newline='') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    name = row.get('Name', '')
                    if name:
                        try:
                            # Shorten kernel names for display
                            short_name = name.split('<')[0].split('(')[0][-40:]
                            kernel_stats[short_name] = {
                                'time_pct': float(row.get('Time (%)', 0) or 0),
                                'total_time_ns': int(float(row.get('Total Time (ns)', 0) or 0)),
                                'instances': int(row.get('Instances', 0) or 0),
                                'avg_ns': float(row.get('Avg (ns)', 0) or 0),
                            }
                        except (ValueError, TypeError):
                            continue
            return kernel_stats
    except Exception:
        return {}


def _strip_profile_suffix(name: str) -> str:
    if name.endswith(".nsys-rep"):
        return name[:-9]
    if name.endswith(".ncu-rep"):
        return name[:-8]
    if name.endswith(".csv"):
        return name[:-4]
    if name.endswith(".sqlite"):
        return name[:-7]
    return name


def _snake_case(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return name


def _normalize_profile_name(name: str) -> str:
    name = _strip_profile_suffix(name)
    if name.startswith("nsys_"):
        name = name[5:]
    name = re.sub(r"^(baseline|optimized)_", "", name, flags=re.IGNORECASE)
    name = re.sub(r"(baseline|optimized)$", "", name, flags=re.IGNORECASE)
    name = name.replace("Baseline", "").replace("Optimized", "").replace("Benchmark", "")
    name = _snake_case(name)
    name = re.sub(r"baseline|optimized", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[_]+", "_", name)
    normalized = name.strip("_").lower()
    tokens = [token for token in normalized.split("_") if token]
    if tokens and len(tokens) % 2 == 0:
        half = len(tokens) // 2
        if tokens[:half] == tokens[half:]:
            normalized = "_".join(tokens[:half])
    return normalized


def _tokenize_profile_name(name: str) -> List[str]:
    normalized = _normalize_profile_name(name)
    tokens = [token for token in normalized.split("_") if token]
    return tokens


def _group_profile_pairs(
    baseline_files: List[Path],
    optimized_files: List[Path],
) -> Dict[str, Dict[str, List[Path]]]:
    grouped: Dict[str, Dict[str, List[Path]]] = {}

    def _key_from_path(path: Path) -> str:
        key = _normalize_profile_name(path.name)
        if not key:
            key = _strip_profile_suffix(path.name).lower()
        return key

    for path in baseline_files:
        key = _key_from_path(path)
        grouped.setdefault(key, {}).setdefault("baseline", []).append(path)
    for path in optimized_files:
        key = _key_from_path(path)
        grouped.setdefault(key, {}).setdefault("optimized", []).append(path)
    return {
        key: value
        for key, value in grouped.items()
        if value.get("baseline") and value.get("optimized")
    }


def _select_profile_pair(
    baseline_files: List[Path],
    optimized_files: List[Path],
    pair_key: Optional[str],
    label: str,
) -> tuple[Optional[tuple[Path, Path]], Optional[str], Optional[Dict[str, Any]]]:
    if not baseline_files or not optimized_files:
        return None, None, None

    candidates = _group_profile_pairs(baseline_files, optimized_files)
    if not candidates:
        return None, None, None

    candidate_keys = sorted(candidates.keys())
    if pair_key is None:
        if len(candidate_keys) > 1:
            return (
                None,
                None,
                {
                    "error": (
                        f"Multiple {label} profile pairs found; provide --pair to select one."
                    ),
                    "candidates": candidate_keys,
                },
            )
        selected_key = candidate_keys[0]
        return (
            (
                max(candidates[selected_key]["baseline"], key=lambda p: p.stat().st_mtime),
                max(candidates[selected_key]["optimized"], key=lambda p: p.stat().st_mtime),
            ),
            selected_key,
            None,
        )

    key_tokens = set(_tokenize_profile_name(pair_key))
    pair_key_norm = _normalize_profile_name(pair_key) or pair_key.lower()

    best_key: Optional[str] = None
    best_score = -1.0
    for key in candidate_keys:
        candidate_tokens = set(_tokenize_profile_name(key))
        overlap = len(key_tokens & candidate_tokens)
        if overlap == 0 and pair_key_norm not in key:
            continue
        score = 0.0
        if key == pair_key_norm:
            score += 1.5
        if pair_key_norm in key:
            score += 1.0
        if overlap:
            score += (2.0 * overlap) / (len(key_tokens) + len(candidate_tokens))
        if score > best_score:
            best_score = score
            best_key = key

    if best_key is None:
        return (
            None,
            None,
            {
                "error": f"No matching {label} profile pair found for --pair={pair_key}.",
                "candidates": candidate_keys,
            },
        )

    return (
        (
            max(candidates[best_key]["baseline"], key=lambda p: p.stat().st_mtime),
            max(candidates[best_key]["optimized"], key=lambda p: p.stat().st_mtime),
        ),
        best_key,
        None,
    )


def _select_flamegraph_pair(
    profiles_dir: Path,
    pair_key: Optional[str] = None,
) -> tuple[Optional[tuple[Path, Path]], Optional[Dict[str, Any]]]:
    nsys_files = sorted(profiles_dir.glob("*.nsys-rep"))
    baseline_nsys = [path for path in nsys_files if "baseline" in path.name.lower()]
    optimized_nsys = [path for path in nsys_files if "optimized" in path.name.lower()]

    if not baseline_nsys or not optimized_nsys:
        return None, None

    pair, _, error = _select_profile_pair(
        baseline_nsys,
        optimized_nsys,
        pair_key=pair_key,
        label="nsys",
    )
    return pair, error


def generate_flamegraph_comparison(
    profiles_dir: Path,
    pair_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Generate flame graph comparison data for baseline vs optimized profiles.
    
    Returns structured data for the FlameGraphComparison React component.
    """
    pair, error = _select_flamegraph_pair(profiles_dir, pair_key=pair_key)
    if error:
        return error
    if pair is None:
        return None
    baseline_path, optimized_path = pair
    
    try:
        baseline_api = _extract_nsys_cuda_api_stats(baseline_path)
        optimized_api = _extract_nsys_cuda_api_stats(optimized_path)
        baseline_kernels = _extract_nsys_kernel_stats(baseline_path)
        optimized_kernels = _extract_nsys_kernel_stats(optimized_path)
        
        # Calculate total times
        baseline_total_ns = sum(s.get('total_time_ns', 0) for s in baseline_api.values())
        optimized_total_ns = sum(s.get('total_time_ns', 0) for s in optimized_api.values())
        
        # Calculate speedup
        baseline_total_ns_f = float(baseline_total_ns)
        optimized_total_ns_f = float(optimized_total_ns)
        if (
            not math.isfinite(baseline_total_ns_f)
            or not math.isfinite(optimized_total_ns_f)
            or baseline_total_ns_f <= 0
            or optimized_total_ns_f <= 0
        ):
            speedup = 1.0
        else:
            speedup = baseline_total_ns_f / optimized_total_ns_f
        
        # Build API breakdown for flame bars
        def build_api_bars(api_stats: Dict) -> List[Dict]:
            bars = []
            for name, stats in sorted(api_stats.items(), key=lambda x: -x[1].get('time_pct', 0)):
                bar_type = 'sync' if 'Sync' in name else \
                          'malloc' if 'Malloc' in name or 'Alloc' in name or 'Free' in name else \
                          'launch' if 'Launch' in name else \
                          'memcpy' if 'Memcpy' in name else \
                          'wait' if 'Wait' in name else 'other'
                bars.append({
                    'name': name,
                    'time_pct': stats.get('time_pct', 0),
                    'total_time_ns': stats.get('total_time_ns', 0),
                    'num_calls': stats.get('num_calls', 0),
                    'type': bar_type,
                })
            return bars[:10]  # Top 10 API calls
        
        def build_kernel_bars(kernel_stats: Dict) -> List[Dict]:
            bars = []
            for name, stats in sorted(kernel_stats.items(), key=lambda x: -x[1].get('time_pct', 0)):
                bars.append({
                    'name': name,
                    'time_pct': stats.get('time_pct', 0),
                    'total_time_ns': stats.get('total_time_ns', 0),
                    'instances': stats.get('instances', 0),
                })
            return bars[:8]  # Top 8 kernels
        
        # Key metrics for comparison
        baseline_sync_calls = sum(
            s.get('num_calls', 0) for n, s in baseline_api.items() 
            if 'Sync' in n
        )
        optimized_sync_calls = sum(
            s.get('num_calls', 0) for n, s in optimized_api.items() 
            if 'Sync' in n
        )
        
        baseline_device_sync = baseline_api.get('cudaDeviceSynchronize', {}).get('num_calls', 0)
        optimized_device_sync = optimized_api.get('cudaDeviceSynchronize', {}).get('num_calls', 0)
        
        optimized_wait_events = optimized_api.get('cudaStreamWaitEvent', {}).get('num_calls', 0)
        
        return {
            'baseline': {
                'file': baseline_path.name,
                'total_time_ms': baseline_total_ns / 1_000_000,
                'api_bars': build_api_bars(baseline_api),
                'kernel_bars': build_kernel_bars(baseline_kernels),
            },
            'optimized': {
                'file': optimized_path.name,
                'total_time_ms': optimized_total_ns / 1_000_000,
                'api_bars': build_api_bars(optimized_api),
                'kernel_bars': build_kernel_bars(optimized_kernels),
            },
            'speedup': round(speedup, 2),
            'metrics': {
                'baseline_sync_calls': baseline_sync_calls,
                'optimized_sync_calls': optimized_sync_calls,
                'sync_reduction_pct': round((1 - optimized_sync_calls / baseline_sync_calls) * 100, 1) if baseline_sync_calls > 0 else 0,
                'baseline_device_sync': baseline_device_sync,
                'optimized_device_sync': optimized_device_sync,
                'device_sync_reduction_pct': round((1 - optimized_device_sync / baseline_device_sync) * 100, 1) if baseline_device_sync > 0 else 0,
                'optimized_wait_events': optimized_wait_events,
            },
            'insight': _generate_optimization_insight(baseline_api, optimized_api),
        }
    except Exception as exc:
        return {'error': str(exc)}


def _generate_optimization_insight(baseline_api: Dict, optimized_api: Dict) -> str:
    """Generate a human-readable insight about the optimization."""
    insights = []
    
    # Check sync reduction
    baseline_sync = baseline_api.get('cudaStreamSynchronize', {}).get('num_calls', 0)
    optimized_sync = optimized_api.get('cudaStreamSynchronize', {}).get('num_calls', 0)
    if baseline_sync > 0 and optimized_sync < baseline_sync:
        reduction = (1 - optimized_sync / baseline_sync) * 100
        insights.append(f"Stream syncs reduced by {reduction:.0f}% ({baseline_sync} → {optimized_sync})")
    
    # Check device sync reduction
    baseline_dev = baseline_api.get('cudaDeviceSynchronize', {}).get('num_calls', 0)
    optimized_dev = optimized_api.get('cudaDeviceSynchronize', {}).get('num_calls', 0)
    if baseline_dev > 0 and optimized_dev < baseline_dev:
        reduction = (1 - optimized_dev / baseline_dev) * 100
        insights.append(f"Device syncs reduced by {reduction:.0f}% ({baseline_dev} → {optimized_dev})")
    
    # Check for stream wait events (pipelining indicator)
    wait_events = optimized_api.get('cudaStreamWaitEvent', {}).get('num_calls', 0)
    if wait_events > 0:
        insights.append(f"Uses {wait_events} stream wait events for lightweight coordination")
    
    return "; ".join(insights) if insights else "Profile comparison available"


def generate_recommendations_from_profiles(result: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on profile comparison data."""
    recommendations: List[str] = []

    nsys = result.get("nsys_comparison", {})
    if nsys and isinstance(nsys, dict) and "metrics" in nsys:
        for metric in nsys.get("metrics", []):
            name = metric.get("name", "").lower()
            delta = metric.get("delta")

            if delta and "dram" in name and delta < -10:
                recommendations.append(
                    "Memory traffic reduced: Consider further optimization with TMA prefetching"
                )
            elif delta and "sm" in name and delta > 10:
                recommendations.append(
                    f"SM utilization improved by {delta:.1f}%: Good progress on compute efficiency"
                )

    ncu = result.get("ncu_comparison", {})
    if ncu and isinstance(ncu, dict) and "metrics" in ncu:
        for metric in ncu.get("metrics", []):
            name = metric.get("name", "").lower()
            if "occupancy" in name:
                try:
                    opt_val = float(str(metric.get("optimized", "0")).replace("%", ""))
                    if opt_val < 50:
                        recommendations.append(
                            f"Occupancy at {opt_val:.0f}%: Consider tuning block size or reducing register pressure"
                        )
                except (ValueError, TypeError):
                    pass

    if not recommendations:
        recommendations.append("Profile both baseline and optimized to get detailed recommendations")

    return recommendations
