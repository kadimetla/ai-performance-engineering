"""Utilities for comparing benchmark results and detecting regressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ComparisonResult:
    """Result of comparing two benchmark runs."""
    baseline_mean_ms: float
    optimized_mean_ms: float
    speedup: float
    regression: bool
    regression_pct: Optional[float] = None
    improvement_pct: Optional[float] = None


def compare_results(
    baseline_result,
    optimized_result,
    regression_threshold_pct: float = 5.0,
    improvement_threshold_pct: float = 5.0
) -> ComparisonResult:
    """Compare baseline and optimized benchmark results.
    
    Args:
        baseline_result: BenchmarkResult from baseline run
        optimized_result: BenchmarkResult from optimized run
        regression_threshold_pct: Percentage degradation to consider a regression (default: 5%)
        improvement_threshold_pct: Percentage improvement to consider significant (default: 5%)
        
    Returns:
        ComparisonResult with speedup and regression detection
    """
    baseline_mean = baseline_result.mean_ms
    optimized_mean = optimized_result.mean_ms
    
    if optimized_mean <= 0:
        speedup = 0.0
    else:
        speedup = baseline_mean / optimized_mean
    
    # Detect regression: optimized is slower by threshold
    regression = False
    regression_pct = None
    if speedup < 1.0:
        regression_pct = (1.0 - speedup) * 100
        regression = regression_pct >= regression_threshold_pct
    
    # Detect improvement: optimized is faster by threshold
    improvement_pct = None
    if speedup > 1.0:
        improvement_pct = (speedup - 1.0) * 100
    
    return ComparisonResult(
        baseline_mean_ms=baseline_mean,
        optimized_mean_ms=optimized_mean,
        speedup=speedup,
        regression=regression,
        regression_pct=regression_pct,
        improvement_pct=improvement_pct if improvement_pct and improvement_pct >= improvement_threshold_pct else None,
    )


def detect_regressions(
    comparisons: List[ComparisonResult],
    regression_threshold_pct: float = 5.0
) -> List[ComparisonResult]:
    """Detect regressions from a list of comparisons.
    
    Args:
        comparisons: List of ComparisonResult objects
        regression_threshold_pct: Percentage degradation to consider a regression
        
    Returns:
        List of ComparisonResult objects that represent regressions
    """
    return [c for c in comparisons if c.regression]


def format_comparison(comparison: ComparisonResult, name: str = "Benchmark") -> str:
    """Format a comparison result as a human-readable string.
    
    Args:
        comparison: ComparisonResult to format
        name: Name of the benchmark
        
    Returns:
        Formatted string
    """
    lines = [
        f"{name}:",
        f"  Baseline: {comparison.baseline_mean_ms:.3f} ms",
        f"  Optimized: {comparison.optimized_mean_ms:.3f} ms",
        f"  Speedup: {comparison.speedup:.2f}x",
    ]
    
    if comparison.regression:
        lines.append(f"  ⚠ REGRESSION: {comparison.regression_pct:.1f}% slower")
    elif comparison.improvement_pct:
        lines.append(f"  ✓ Improvement: {comparison.improvement_pct:.1f}% faster")
    else:
        lines.append("  → No significant change")
    
    return "\n".join(lines)

