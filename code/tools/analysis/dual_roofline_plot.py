#!/usr/bin/env python3
"""
Generate a dual-roofline plot for Blackwell-era GPUs.

The script combines Nsight Compute CSV exports (for SM and TMEM counters)
with a kernel metadata file that lists arithmetic intensity and achieved FLOP/s
per kernel. We then determine which roof (SM compute or tensor memory) is
binding for each kernel and emit both a plot and a textual summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - handled at runtime
    plt = None


ARCH_DEFAULTS = {
    "blackwell-b200": {
        "peak_flops_tflops": 450.0,
        "tmem_bandwidth_gbs": 9000.0,
        "l2_bandwidth_gbs": 12500.0,
        "dram_bandwidth_gbs": 7800.0,
    },
    "blackwell-b300": {
        "peak_flops_tflops": 450.0,
        "tmem_bandwidth_gbs": 10000.0,
        "l2_bandwidth_gbs": 13500.0,
        "dram_bandwidth_gbs": 9000.0,
    },
    "gb200": {
        "peak_flops_tflops": 450.0,
        "tmem_bandwidth_gbs": 9000.0,
        "l2_bandwidth_gbs": 12500.0,
        "dram_bandwidth_gbs": 7800.0,
    },
    "hopper-h100": {
        "peak_flops_tflops": 395.0,
        "tmem_bandwidth_gbs": 3500.0,
        "l2_bandwidth_gbs": 7000.0,
        "dram_bandwidth_gbs": 3350.0,
    },
}


@dataclass
class MetricSample:
    """Single Nsight Compute metric sample."""

    value: float
    unit: Optional[str]
    name: str


@dataclass
class KernelMetadata:
    """User-provided arithmetic intensity + achieved TFLOP/s summary."""

    name: str
    intensity: float
    achieved_tflops: float
    label: Optional[str] = None


@dataclass
class DualRooflinePoint:
    """Derived per-kernel point for plotting."""

    kernel: str
    intensity: float
    achieved_tflops: float
    sm_util_pct: float
    tmem_gbs: float
    l2_gbs: float
    dram_gbs: float
    compute_roof_tflops: float
    tmem_roof_tflops: float
    binding: str
    efficiency_pct: float
    label: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dual compute/TMEM rooflines.")
    parser.add_argument(
        "--ncu-csv",
        required=True,
        type=Path,
        help="Nsight Compute CSV with SM/TMEM metrics (use ncu --csv export).",
    )
    parser.add_argument(
        "--kernel-meta",
        required=True,
        type=Path,
        help="CSV/JSON file with kernel,intensity(FLOPs/byte),achieved_tflops columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dual_roofline.png"),
        help="Output PNG path for the plot.",
    )
    parser.add_argument(
        "--arch",
        choices=sorted(ARCH_DEFAULTS.keys()),
        default="blackwell-b200",
        help="Architecture preset for default peak FLOPs and bandwidths.",
    )
    parser.add_argument(
        "--peak-flops",
        type=float,
        default=None,
        help="Peak Tensor Core throughput in TFLOP/s for the precision of interest.",
    )
    parser.add_argument(
        "--tmem-bandwidth",
        type=float,
        default=None,
        help="Maximum TMEM sustained throughput in GB/s.",
    )
    parser.add_argument(
        "--l2-bandwidth",
        type=float,
        default=None,
        help="Maximum L2 sustained throughput in GB/s.",
    )
    parser.add_argument(
        "--dram-bandwidth",
        type=float,
        default=None,
        help="Maximum HBM/DRAM sustained throughput in GB/s.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print the textual summary; skip matplotlib plotting.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to writing a PNG.",
    )
    return parser.parse_args()


def ensure_defaults(args: argparse.Namespace) -> None:
    defaults = ARCH_DEFAULTS[args.arch]
    if args.peak_flops is None:
        args.peak_flops = defaults["peak_flops_tflops"]
    if args.tmem_bandwidth is None:
        args.tmem_bandwidth = defaults["tmem_bandwidth_gbs"]
    if args.l2_bandwidth is None:
        args.l2_bandwidth = defaults["l2_bandwidth_gbs"]
    if args.dram_bandwidth is None:
        args.dram_bandwidth = defaults["dram_bandwidth_gbs"]


def _safe_float(value: str) -> Optional[float]:
    try:
        if value is None:
            return None
        token = value.strip()
        if not token:
            return None
        return float(token)
    except ValueError:
        return None


def parse_ncu_csv(csv_path: Path) -> Dict[str, Dict[str, MetricSample]]:
    """Parse Nsight Compute CSV exports grouped by kernel name."""
    metrics_by_kernel: Dict[str, Dict[str, MetricSample]] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if not first:
                continue
            if first.startswith("==PROF==") or first.startswith("#"):
                continue
            cells = [cell.strip().strip('"') for cell in row]
            if cells[:2] == ["ID", "Kernel Name"]:
                # Header line
                continue
            if cells[0].isdigit() and len(cells) >= 5:
                _, kernel_name, metric_name, metric_unit, metric_value = cells[:5]
                value = _safe_float(metric_value)
                if value is None:
                    continue
                kernel_metrics = metrics_by_kernel.setdefault(kernel_name, {})
                kernel_metrics[metric_name] = MetricSample(
                    value=value,
                    unit=metric_unit or None,
                    name=metric_name,
                )
                continue
            # Key/value CSV without kernel names - fall back to aggregate bucket.
            if len(cells) >= 2:
                metric_name = cells[0]
                metric_value = _safe_float(cells[-1])
                if metric_value is None:
                    continue
                kernel_metrics = metrics_by_kernel.setdefault("__aggregate__", {})
                kernel_metrics[metric_name] = MetricSample(
                    value=metric_value,
                    unit=None,
                    name=metric_name,
                )
    return metrics_by_kernel


def load_kernel_metadata(meta_path: Path) -> List[KernelMetadata]:
    """Load kernel metadata from CSV or JSON."""
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    suffix = meta_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(meta_path.read_text())
        return [
            KernelMetadata(
                name=entry.get("kernel") or entry["name"],
                intensity=float(entry["intensity"]),
                achieved_tflops=float(entry["achieved_tflops"]),
                label=entry.get("label"),
            )
            for entry in data
        ]
    # Default to CSV
    results: List[KernelMetadata] = []
    with meta_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            kernel = row.get("kernel") or row.get("name")
            if not kernel:
                raise ValueError("Metadata file must include a 'kernel' column.")
            intensity = row.get("intensity")
            achieved = (
                row.get("achieved_tflops")
                or row.get("achieved_tfops")
                or row.get("achieved")
            )
            if intensity is None or achieved is None:
                raise ValueError(
                    f"Missing intensity or achieved_tflops for kernel {kernel}"
                )
            results.append(
                KernelMetadata(
                    name=kernel.strip(),
                    intensity=float(intensity),
                    achieved_tflops=float(achieved),
                    label=row.get("label") or row.get("tag"),
                )
            )
    return results


def _find_metric(
    metrics: Dict[str, MetricSample], base_name: str
) -> Optional[MetricSample]:
    """Return the first metric whose name starts with the base token."""
    if base_name in metrics:
        return metrics[base_name]
    for key, sample in metrics.items():
        if key.startswith(base_name):
            return sample
    return None


def _resolve_throughput(
    sample: Optional[MetricSample], peak_limit_gbs: float
) -> float:
    """Convert Nsight counters (percent or GB/s) into GB/s."""
    if sample is None:
        return peak_limit_gbs
    if sample.unit and sample.unit.strip() == "%":
        return peak_limit_gbs * sample.value / 100.0
    if sample.name.endswith("pct_of_peak_sustained_elapsed"):
        return peak_limit_gbs * sample.value / 100.0
    return sample.value


def _resolve_sm_pct(metrics: Dict[str, MetricSample]) -> float:
    sample = _find_metric(metrics, "sm__throughput")
    if sample is None:
        return 0.0
    if sample.unit and sample.unit.strip() == "%":
        return sample.value
    if sample.name.endswith("pct_of_peak_sustained_elapsed"):
        return sample.value
    # Already reported as a ratio (0-1)
    return sample.value * 100.0


def _min_ignore_none(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    return min(filtered) if filtered else None


def build_points(
    kernel_meta: List[KernelMetadata],
    metrics_by_kernel: Dict[str, Dict[str, MetricSample]],
    peak_flops_tflops: float,
    tmem_limit_gbs: float,
    l2_limit_gbs: float,
    dram_limit_gbs: float,
) -> List[DualRooflinePoint]:
    points: List[DualRooflinePoint] = []
    for meta in kernel_meta:
        metrics = metrics_by_kernel.get(meta.name)
        if metrics is None:
            # Try substring match if exact name is unavailable.
            matches = [
                (k, v) for k, v in metrics_by_kernel.items() if meta.name in k
            ]
            if len(matches) == 1:
                metrics = matches[0][1]
            elif len(matches) > 1:
                raise KeyError(
                    f"Kernel '{meta.name}' matches multiple Nsight entries: "
                    f"{[name for name, _ in matches]}"
                )
            else:
                raise KeyError(
                    f"Kernel '{meta.name}' not found in Nsight CSV. "
                    "Ensure kernel names match exactly."
                )
        sm_pct = _resolve_sm_pct(metrics)
        compute_roof_tflops = peak_flops_tflops * sm_pct / 100.0

        tmem_sample = _find_metric(metrics, "tmem__throughput")
        l2_sample = _find_metric(metrics, "l2__throughput")
        dram_sample = _find_metric(metrics, "dram__throughput")

        tmem_gbs = min(
            _resolve_throughput(tmem_sample, tmem_limit_gbs), tmem_limit_gbs
        )
        l2_gbs = min(_resolve_throughput(l2_sample, l2_limit_gbs), l2_limit_gbs)
        dram_gbs = min(
            _resolve_throughput(dram_sample, dram_limit_gbs), dram_limit_gbs
        )
        effective_tmem_gbs = _min_ignore_none([tmem_gbs, l2_gbs, dram_gbs])
        if effective_tmem_gbs is None:
            effective_tmem_gbs = min(tmem_limit_gbs, l2_limit_gbs, dram_limit_gbs)
        tmem_roof_tflops = effective_tmem_gbs * meta.intensity / 1000.0

        binding = "compute"
        ceiling = compute_roof_tflops
        if tmem_roof_tflops < compute_roof_tflops:
            binding = "tmem"
            ceiling = tmem_roof_tflops
        efficiency = (
            (meta.achieved_tflops / ceiling) * 100.0 if ceiling > 0 else 0.0
        )
        if efficiency > 120.0:
            print(
                f"Warning: {meta.name} reports {efficiency:.1f}% of the "
                f"{binding.upper()} roof. Verify FLOP/byte math and Nsight metrics."
            )

        points.append(
            DualRooflinePoint(
                kernel=meta.name,
                intensity=meta.intensity,
                achieved_tflops=meta.achieved_tflops,
                sm_util_pct=sm_pct,
                tmem_gbs=tmem_gbs,
                l2_gbs=l2_gbs,
                dram_gbs=dram_gbs,
                compute_roof_tflops=compute_roof_tflops,
                tmem_roof_tflops=tmem_roof_tflops,
                binding=binding,
                efficiency_pct=efficiency,
                label=meta.label or meta.name,
            )
        )
    return points


def print_summary(points: List[DualRooflinePoint]) -> None:
    header = (
        f"{'Kernel':<32} {'AI(F/B)':>10} {'Ach TFLOPs':>12} "
        f"{'SM%':>6} {'TMEM GB/s':>11} {'Bind':>7} {'Ceiling TFLOPs':>16} {'Eff%':>7}"
    )
    print(header)
    print("-" * len(header))
    for point in points:
        ceiling = (
            point.compute_roof_tflops
            if point.binding == "compute"
            else point.tmem_roof_tflops
        )
        print(
            f"{point.kernel:<32} "
            f"{point.intensity:>10.2f} "
            f"{point.achieved_tflops:>12.1f} "
            f"{point.sm_util_pct:>6.1f} "
            f"{point.tmem_gbs:>11.0f} "
            f"{point.binding.upper():>7} "
            f"{ceiling:>16.1f} "
            f"{point.efficiency_pct:>7.1f}"
        )
    print()


def plot_roofline(
    points: List[DualRooflinePoint],
    peak_flops_tflops: float,
    tmem_limit_gbs: float,
    l2_limit_gbs: float,
    dram_limit_gbs: float,
    output_path: Path,
    show: bool = False,
) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping plot generation.")
        return
    if not points:
        print("No points to plot.")
        return
    intensities = [p.intensity for p in points]
    x_min = min(intensities) * 0.5
    x_max = max(intensities) * 2.0
    x_min = max(x_min, 0.1)
    x_values = [x_min * (2 ** (i / 50.0)) for i in range(200)]

    effective_tmem_limit = min(tmem_limit_gbs, l2_limit_gbs, dram_limit_gbs)
    compute_line = [peak_flops_tflops for _ in x_values]
    tmem_line = [effective_tmem_limit * x / 1000.0 for x in x_values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs/byte)")
    ax.set_ylabel("FLOP/s (TFLOPs)")
    ax.set_title("Dual roofline (SM compute vs TMEM throughput)")
    ax.plot(x_values, compute_line, linestyle="--", color="grey", label="SM peak")
    ax.plot(
        x_values,
        tmem_line,
        linestyle="-.",
        color="orange",
        label=f"TMEM/L2/DRAM limit ({effective_tmem_limit:.0f} GB/s)",
    )

    colors = {"compute": "#d62728", "tmem": "#1f77b4"}
    seen_labels: Dict[str, bool] = {}
    for point in points:
        label = point.binding.capitalize()
        legend_label = label if label not in seen_labels else ""
        seen_labels[label] = True
        ax.scatter(
            point.intensity,
            point.achieved_tflops,
            color=colors.get(point.binding, "#2ca02c"),
            edgecolor="white",
            linewidth=0.8,
            s=60,
            label=legend_label,
            zorder=3,
        )
        ax.annotate(
            point.label or point.kernel,
            (point.intensity, point.achieved_tflops),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    ax.legend(loc="lower right")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Wrote dual-roofline plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_defaults(args)
    metrics = parse_ncu_csv(args.ncu_csv)
    kernel_meta = load_kernel_metadata(args.kernel_meta)
    points = build_points(
        kernel_meta=kernel_meta,
        metrics_by_kernel=metrics,
        peak_flops_tflops=args.peak_flops,
        tmem_limit_gbs=args.tmem_bandwidth,
        l2_limit_gbs=args.l2_bandwidth,
        dram_limit_gbs=args.dram_bandwidth,
    )
    print_summary(points)
    if not args.no_plot:
        plot_roofline(
            points,
            peak_flops_tflops=args.peak_flops,
            tmem_limit_gbs=args.tmem_bandwidth,
            l2_limit_gbs=args.l2_bandwidth,
            dram_limit_gbs=args.dram_bandwidth,
            output_path=args.output,
            show=args.show,
        )


if __name__ == "__main__":
    main()
