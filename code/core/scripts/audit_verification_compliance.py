#!/usr/bin/env python3
"""Audit script for benchmark verification compliance.

This script scans all benchmark files in the repository and reports on
verification compliance status:

- Benchmarks with/without get_input_signature()
- Benchmarks with/without validate_result()
- Benchmarks with/without get_workload_metadata()
- Benchmarks with skip flags (skip_output_check, skip_input_check, skip_verification)

Usage:
    python -m core.scripts.audit_verification_compliance [--chapter CHAPTER] [--json]
    
Output:
    Summary report by chapter showing compliance gaps
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class BenchmarkInfo:
    """Information about a single benchmark file."""
    path: str
    has_get_input_signature: bool = False
    has_validate_result: bool = False
    has_get_workload_metadata: bool = False
    has_get_verify_output: bool = False  # STRICT: Mandatory for verification
    has_skip_output_check: bool = False
    has_skip_input_check: bool = False
    has_skip_verification: bool = False
    skip_input_verification: bool = False
    skip_output_verification: bool = False
    benchmark_class: Optional[str] = None
    errors: List[str] = field(default_factory=list)


@dataclass 
class ChapterReport:
    """Compliance report for a chapter."""
    chapter: str
    total_baseline: int = 0
    total_optimized: int = 0
    baseline_with_signature: int = 0
    optimized_with_signature: int = 0
    baseline_with_validate: int = 0
    optimized_with_validate: int = 0
    baseline_with_workload: int = 0
    optimized_with_workload: int = 0
    baseline_with_verify_output: int = 0  # STRICT: Mandatory for verification
    optimized_with_verify_output: int = 0  # STRICT: Mandatory for verification
    benchmarks_with_skip_flags: List[str] = field(default_factory=list)
    benchmarks: List[BenchmarkInfo] = field(default_factory=list)


def analyze_benchmark_file(file_path: Path) -> BenchmarkInfo:
    """Analyze a benchmark file for verification compliance.
    
    Uses AST parsing to avoid importing modules (which may have dependencies).
    """
    info = BenchmarkInfo(path=str(file_path))
    
    try:
        source = file_path.read_text()
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        info.errors.append(f"Syntax error: {e}")
        return info
    except Exception as e:
        info.errors.append(f"Parse error: {e}")
        return info
    
    # Base classes that provide get_verify_output() by default
    bases_with_verify_output = {
        "CudaBinaryBenchmark", "OccupancyBinaryBenchmark",
        "NvshmemIbgdaMicrobench",
    }
    
    # Base classes that provide get_input_signature() by default
    bases_with_input_signature = {
        "CudaBinaryBenchmark",  # Has default returning _workload_params
    }
    
    # Base classes that inherit get_workload_metadata() from BaseBenchmark
    # (via register_workload_metadata)
    bases_with_workload_metadata = {
        "BaseBenchmark", "CudaBinaryBenchmark", "OccupancyBinaryBenchmark",
        "LoopUnrollingBenchmarkBase",
    }
    
    # Base classes that are benchmarks (inherit from BaseBenchmark)
    known_benchmark_bases = {
        "BaseBenchmark", "CudaBinaryBenchmark", "OccupancyBinaryBenchmark",
        "AiOptimizationBenchmarkBase", "ThresholdBenchmarkBase", "TilingBenchmarkBase",
        "HBMBenchmarkBase", "LoopUnrollingBenchmarkBase", "ThresholdBenchmarkBaseTMA",
        "TilingBenchmarkBaseTCGen05", "BaselineMatmulTCGen05Benchmark",
        "StridedStreamBaseline", "ConcurrentStreamOptimized", "StreamOrderedBase",
        "TorchrunScriptBenchmark", "MoEJourneyBenchmark", "MoEBenchmarkBase",
        "NvshmemIbgdaMicrobench", "BenchmarkBase", "FlexDecodingHarness",
    }
    
    # Find benchmark class and analyze its methods
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if this looks like a benchmark class
            has_benchmark_fn = any(
                isinstance(item, ast.FunctionDef) and item.name == "benchmark_fn"
                for item in node.body
            )
            
            # Also check if it inherits from a known benchmark base
            inherits_from_benchmark = False
            inherits_from_verify_provider = False
            inherits_from_signature_provider = False
            inherits_from_workload_provider = False
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name:
                    if any(b in base_name for b in known_benchmark_bases):
                        inherits_from_benchmark = True
                    if any(b in base_name for b in bases_with_verify_output):
                        inherits_from_verify_provider = True
                    if any(b in base_name for b in bases_with_input_signature):
                        inherits_from_signature_provider = True
                    if any(b in base_name for b in bases_with_workload_metadata):
                        inherits_from_workload_provider = True
            
            if not has_benchmark_fn and not inherits_from_benchmark:
                continue
            
            info.benchmark_class = node.name
            
            # Mark inherited methods
            if inherits_from_verify_provider:
                info.has_get_verify_output = True
            if inherits_from_signature_provider:
                info.has_get_input_signature = True
            
            # Check for register_workload_metadata() calls
            has_register_workload = False
            for item in ast.walk(node):
                if isinstance(item, ast.Call):
                    if isinstance(item.func, ast.Attribute):
                        if item.func.attr == "register_workload_metadata":
                            has_register_workload = True
                            break
            
            if has_register_workload or inherits_from_workload_provider:
                info.has_get_workload_metadata = True
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == "get_input_signature":
                        info.has_get_input_signature = True
                    elif item.name == "validate_result":
                        info.has_validate_result = True
                    elif item.name == "get_workload_metadata":
                        info.has_get_workload_metadata = True
                    elif item.name == "get_verify_output":
                        info.has_get_verify_output = True
                    elif item.name == "skip_input_verification":
                        info.skip_input_verification = True
                    elif item.name == "skip_output_verification":
                        info.skip_output_verification = True
                
                # Check for skip flag attributes in __init__ assignments
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                                    if target.value.id == "self":
                                        if target.attr == "skip_output_check":
                                            info.has_skip_output_check = True
                                        elif target.attr == "skip_input_check":
                                            info.has_skip_input_check = True
                                        elif target.attr == "skip_verification":
                                            info.has_skip_verification = True
                
                # Check for skip flag attributes (class-level)
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            if target.id == "skip_output_check":
                                info.has_skip_output_check = True
                            elif target.id == "skip_input_check":
                                info.has_skip_input_check = True
                            elif target.id == "skip_verification":
                                info.has_skip_verification = True
    
    return info


def find_benchmark_files(root_dir: Path, chapter: Optional[str] = None) -> List[Path]:
    """Find all benchmark files (baseline_*.py, optimized_*.py)."""
    files: List[Path] = []
    
    # Patterns to match
    patterns = ["baseline_*.py", "optimized_*.py"]
    
    # Directories to search
    if chapter:
        search_dirs = [root_dir / chapter]
    else:
        search_dirs = [
            d for d in root_dir.iterdir()
            if d.is_dir() and (d.name.startswith("ch") or d.name == "labs")
        ]
        # Also search labs subdirectories
        labs_dir = root_dir / "labs"
        if labs_dir.exists():
            search_dirs.extend(
                d for d in labs_dir.iterdir() if d.is_dir()
            )
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            files.extend(search_dir.glob(pattern))
            # Also check subdirectories
            files.extend(search_dir.glob(f"**/{pattern}"))
    
    return sorted(set(files))


def generate_report(root_dir: Path, chapter: Optional[str] = None) -> Dict[str, ChapterReport]:
    """Generate compliance report for all chapters."""
    files = find_benchmark_files(root_dir, chapter)
    
    reports: Dict[str, ChapterReport] = defaultdict(lambda: ChapterReport(chapter=""))
    
    for file_path in files:
        # Determine chapter from path
        relative = file_path.relative_to(root_dir)
        parts = relative.parts
        if parts[0] == "labs" and len(parts) > 1:
            chapter_name = f"labs/{parts[1]}"
        else:
            chapter_name = parts[0]
        
        if chapter_name not in reports:
            reports[chapter_name] = ChapterReport(chapter=chapter_name)
        
        report = reports[chapter_name]
        
        # Analyze the file
        info = analyze_benchmark_file(file_path)
        report.benchmarks.append(info)
        
        # Update counts
        is_baseline = file_path.name.startswith("baseline_")
        is_optimized = file_path.name.startswith("optimized_")
        
        if is_baseline:
            report.total_baseline += 1
            if info.has_get_input_signature:
                report.baseline_with_signature += 1
            if info.has_validate_result:
                report.baseline_with_validate += 1
            if info.has_get_workload_metadata:
                report.baseline_with_workload += 1
            if info.has_get_verify_output:
                report.baseline_with_verify_output += 1
        elif is_optimized:
            report.total_optimized += 1
            if info.has_get_input_signature:
                report.optimized_with_signature += 1
            if info.has_validate_result:
                report.optimized_with_validate += 1
            if info.has_get_workload_metadata:
                report.optimized_with_workload += 1
            if info.has_get_verify_output:
                report.optimized_with_verify_output += 1
        
        # Track skip flags
        if any([
            info.has_skip_output_check,
            info.has_skip_input_check,
            info.has_skip_verification,
            info.skip_input_verification,
            info.skip_output_verification,
        ]):
            report.benchmarks_with_skip_flags.append(str(file_path))
    
    return dict(reports)


def print_summary(reports: Dict[str, ChapterReport]) -> int:
    """Print human-readable summary report. Returns number of compliance issues."""
    print("=" * 80)
    print("BENCHMARK VERIFICATION COMPLIANCE AUDIT")
    print("=" * 80)
    print()
    
    total_baseline = 0
    total_optimized = 0
    total_with_signature = 0
    total_with_validate = 0
    total_with_workload = 0
    total_with_verify_output = 0  # STRICT: Mandatory for verification
    all_skip_flags: List[str] = []
    
    for chapter_name in sorted(reports.keys()):
        report = reports[chapter_name]
        
        total_baseline += report.total_baseline
        total_optimized += report.total_optimized
        total_with_signature += report.baseline_with_signature + report.optimized_with_signature
        total_with_validate += report.baseline_with_validate + report.optimized_with_validate
        total_with_workload += report.baseline_with_workload + report.optimized_with_workload
        total_with_verify_output += report.baseline_with_verify_output + report.optimized_with_verify_output
        all_skip_flags.extend(report.benchmarks_with_skip_flags)
        
        chapter_total = report.total_baseline + report.total_optimized
        if chapter_total == 0:
            continue
        
        print(f"\n{chapter_name}")
        print("-" * 40)
        print(f"  Baseline:  {report.total_baseline:3d} files")
        print(f"  Optimized: {report.total_optimized:3d} files")
        
        # Signature coverage
        sig_coverage = report.baseline_with_signature + report.optimized_with_signature
        sig_pct = (sig_coverage / chapter_total * 100) if chapter_total > 0 else 0
        print(f"  get_input_signature(): {sig_coverage}/{chapter_total} ({sig_pct:.0f}%)")
        
        # Validate coverage
        val_coverage = report.baseline_with_validate + report.optimized_with_validate
        val_pct = (val_coverage / chapter_total * 100) if chapter_total > 0 else 0
        print(f"  validate_result():     {val_coverage}/{chapter_total} ({val_pct:.0f}%)")
        
        # Workload coverage
        work_coverage = report.baseline_with_workload + report.optimized_with_workload
        work_pct = (work_coverage / chapter_total * 100) if chapter_total > 0 else 0
        print(f"  get_workload_metadata(): {work_coverage}/{chapter_total} ({work_pct:.0f}%)")
        
        # STRICT: Verify output coverage
        verify_coverage = report.baseline_with_verify_output + report.optimized_with_verify_output
        verify_pct = (verify_coverage / chapter_total * 100) if chapter_total > 0 else 0
        print(f"  get_verify_output():   {verify_coverage}/{chapter_total} ({verify_pct:.0f}%) ** STRICT REQUIRED **")
        
        # Skip flags (now deprecated)
        if report.benchmarks_with_skip_flags:
            print(f"  ⚠️  DEPRECATED skip flags: {len(report.benchmarks_with_skip_flags)} files")
            for path in report.benchmarks_with_skip_flags:
                print(f"    - {path}")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    grand_total = total_baseline + total_optimized
    print(f"\nTotal benchmark files: {grand_total}")
    print(f"  Baseline:  {total_baseline}")
    print(f"  Optimized: {total_optimized}")
    
    if grand_total > 0:
        print(f"\nget_input_signature() coverage: {total_with_signature}/{grand_total} "
              f"({total_with_signature / grand_total * 100:.1f}%)")
        print(f"validate_result() coverage: {total_with_validate}/{grand_total} "
              f"({total_with_validate / grand_total * 100:.1f}%)")
        print(f"get_workload_metadata() coverage: {total_with_workload}/{grand_total} "
              f"({total_with_workload / grand_total * 100:.1f}%)")
        print(f"get_verify_output() coverage: {total_with_verify_output}/{grand_total} "
              f"({total_with_verify_output / grand_total * 100:.1f}%) ** STRICT REQUIRED **")
    
    # Calculate issues
    issues = 0
    
    # STRICT: Missing get_verify_output()
    missing_verify_output = grand_total - total_with_verify_output
    if missing_verify_output > 0:
        issues += missing_verify_output
        print(f"\n⚠️  STRICT MODE FAILURES: {missing_verify_output} benchmarks MISSING get_verify_output()")
        print("    These benchmarks will FAIL verification until they implement get_verify_output()!")
    
    # Deprecated skip flags
    if all_skip_flags:
        issues += len(all_skip_flags)
        print(f"\n⚠️  DEPRECATED: {len(all_skip_flags)} files still use deprecated skip flags")
        print("  These need migration to jitter_exemption_reason or removal:")
        for path in sorted(set(all_skip_flags)):
            print(f"    - {path}")
    
    # Missing get_input_signature() (recommended but not strict)
    missing_signature = grand_total - total_with_signature
    if missing_signature > 0:
        print(f"\n📋 RECOMMENDED: {missing_signature} benchmarks missing get_input_signature()")
    
    if issues == 0:
        print("\n✅ All benchmarks are compliant!")
    else:
        print(f"\n❌ Total compliance issues: {issues}")
    
    return issues


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Audit benchmark files for verification compliance"
    )
    parser.add_argument(
        "--chapter", "-c",
        help="Audit only a specific chapter (e.g., ch01, labs/moe_cuda)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output JSON instead of human-readable format"
    )
    parser.add_argument(
        "--root", "-r",
        default=".",
        help="Root directory (default: current directory)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code if any compliance issues found"
    )
    
    args = parser.parse_args()
    root_dir = Path(args.root).resolve()
    
    reports = generate_report(root_dir, args.chapter)
    
    if args.json:
        # Convert to JSON-serializable format
        output = {}
        for chapter, report in reports.items():
            output[chapter] = {
                "total_baseline": report.total_baseline,
                "total_optimized": report.total_optimized,
                "baseline_with_signature": report.baseline_with_signature,
                "optimized_with_signature": report.optimized_with_signature,
                "baseline_with_validate": report.baseline_with_validate,
                "optimized_with_validate": report.optimized_with_validate,
                "baseline_with_workload": report.baseline_with_workload,
                "optimized_with_workload": report.optimized_with_workload,
                "baseline_with_verify_output": report.baseline_with_verify_output,
                "optimized_with_verify_output": report.optimized_with_verify_output,
                "benchmarks_with_skip_flags": report.benchmarks_with_skip_flags,
            }
        print(json.dumps(output, indent=2))
        return 0
    else:
        issues = print_summary(reports)
        if args.strict and issues > 0:
            return min(issues, 255)  # Cap at 255 for shell compatibility
        return 0


if __name__ == "__main__":
    sys.exit(main())


