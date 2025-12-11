#!/usr/bin/env python3
"""Audit verification compliance by actually instantiating benchmarks.

This script properly detects inherited methods (unlike grep-based approaches)
by importing and inspecting each benchmark class.

Usage:
    python -m core.scripts.audit_verification_compliance [--chapter ch10] [--lab decode_optimization]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.benchmark.verification import InputSignature

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def load_benchmark_class(filepath: Path) -> Optional[Tuple[Any, str]]:
    """Load benchmark class from a file.
    
    Returns:
        Tuple of (benchmark_instance, class_name) or None if failed
    """
    try:
        spec = importlib.util.spec_from_file_location("benchmark_module", filepath)
        if spec is None or spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_module"] = module
        spec.loader.exec_module(module)
        
        # Look for get_benchmark factory function
        if hasattr(module, "get_benchmark"):
            benchmark = module.get_benchmark()
            return (benchmark, type(benchmark).__name__)
        
        return None
    except Exception as e:
        # Silently skip files that can't be loaded
        return None
    finally:
        # Clean up
        if "benchmark_module" in sys.modules:
            del sys.modules["benchmark_module"]


def check_compliance(benchmark: Any) -> Dict[str, bool]:
    """Check if a benchmark has required verification methods.
    
    Returns dict with compliance status for each required method.
    """
    compliance = {
        "get_verify_output": False,
        "get_input_signature": False,
        "get_verify_inputs": False,
        "get_output_tolerance": False,
        "validate_result": False,
        # Jitter exemptions are NOT allowed; any exemption is a failure.
        "jitter_exemption_reason": False,
        "register_workload_metadata_called": False,
    }
    
    # Check methods (including inherited)
    if hasattr(benchmark, "get_verify_output"):
        # Check if it's a real implementation, not just NotImplementedError
        try:
            method = getattr(benchmark, "get_verify_output")
            # Check if the method is defined in the class (not just BaseBenchmark)
            if hasattr(method, "__func__"):
                defining_class = method.__func__.__qualname__.split(".")[0]
                if defining_class != "BaseBenchmark":
                    compliance["get_verify_output"] = True
            # Also check CudaBinaryBenchmark
            if "CudaBinaryBenchmark" in str(type(benchmark).__mro__):
                compliance["get_verify_output"] = True
        except Exception:
            pass
    
    if hasattr(benchmark, "get_input_signature"):
        try:
            sig = benchmark.get_input_signature()
            if isinstance(sig, InputSignature):
                compliance["get_input_signature"] = not sig.validate(strict=True)
            elif isinstance(sig, dict):
                shapes = sig.get("shapes", {})
                dtypes = sig.get("dtypes", {})
                batch = sig.get("batch_size", None)
                params = sig.get("parameter_count", None)
                compliance["get_input_signature"] = bool(
                    shapes and dtypes and batch is not None and params is not None
                )
        except RuntimeError:
            # Accept RuntimeError (e.g., setup not called) as an implemented method
            compliance["get_input_signature"] = True
        except Exception:
            pass

    if hasattr(benchmark, "get_verify_inputs"):
        try:
            inp = benchmark.get_verify_inputs()
            if isinstance(inp, torch.Tensor):
                compliance["get_verify_inputs"] = True
            elif isinstance(inp, dict):
                compliance["get_verify_inputs"] = any(isinstance(v, torch.Tensor) for v in inp.values())
        except RuntimeError:
            compliance["get_verify_inputs"] = True
        except Exception:
            pass
    
    if hasattr(benchmark, "get_output_tolerance"):
        try:
            tol = benchmark.get_output_tolerance()
            compliance["get_output_tolerance"] = tol is not None
        except Exception:
            pass
    
    if hasattr(benchmark, "validate_result"):
        compliance["validate_result"] = True
    
    # Jitter exemptions are no longer permitted.
    if hasattr(benchmark, "jitter_exemption_reason"):
        reason = getattr(benchmark, "jitter_exemption_reason")
        if reason:
            compliance["jitter_exemption_reason"] = False
    elif hasattr(benchmark, "non_jitterable_reason"):
        reason = getattr(benchmark, "non_jitterable_reason")
        if reason:
            compliance["jitter_exemption_reason"] = False
    else:
        compliance["jitter_exemption_reason"] = True
    
    # Check if workload metadata was registered
    if hasattr(benchmark, "_workload_registered") and benchmark._workload_registered:
        compliance["register_workload_metadata_called"] = True
    elif hasattr(benchmark, "_workload") and benchmark._workload is not None:
        compliance["register_workload_metadata_called"] = True
    
    return compliance


def audit_directory(directory: Path) -> Dict[str, Dict[str, Any]]:
    """Audit all benchmark files in a directory.
    
    Returns:
        Dict mapping filepath to compliance info
    """
    results = {}
    
    for filepath in sorted(directory.glob("*.py")):
        if not (filepath.name.startswith("baseline_") or filepath.name.startswith("optimized_")):
            continue
        
        result = load_benchmark_class(filepath)
        if result is None:
            results[str(filepath)] = {
                "status": "error",
                "error": "Could not load benchmark",
                "class_name": None,
                "compliance": None,
            }
            continue
        
        benchmark, class_name = result
        compliance = check_compliance(benchmark)
        
        # Determine overall status
        critical_methods = ["get_verify_output", "get_input_signature", "jitter_exemption_reason"]
        is_compliant = all(compliance.get(m, False) for m in critical_methods)
        
        results[str(filepath)] = {
            "status": "compliant" if is_compliant else "needs_work",
            "class_name": class_name,
            "compliance": compliance,
        }
    
    return results


def print_summary(results: Dict[str, Dict[str, Any]], title: str) -> Tuple[int, int, int]:
    """Print summary of audit results.
    
    Returns:
        Tuple of (compliant_count, needs_work_count, error_count)
    """
    compliant = [f for f, r in results.items() if r["status"] == "compliant"]
    needs_work = [f for f, r in results.items() if r["status"] == "needs_work"]
    errors = [f for f, r in results.items() if r["status"] == "error"]
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"✅ Compliant: {len(compliant)}")
    print(f"⚠️  Needs work: {len(needs_work)}")
    print(f"❌ Errors: {len(errors)}")
    print(f"Total: {len(results)}")
    
    if needs_work:
        print(f"\n--- Files needing work ---")
        for filepath in needs_work[:10]:  # Show first 10
            r = results[filepath]
            missing = [k for k, v in r["compliance"].items() if not v]
            print(f"  {Path(filepath).name}: missing {missing}")
        if len(needs_work) > 10:
            print(f"  ... and {len(needs_work) - 10} more")
    
    return len(compliant), len(needs_work), len(errors)


def main():
    parser = argparse.ArgumentParser(description="Audit benchmark verification compliance")
    parser.add_argument("--chapter", type=str, help="Specific chapter to audit (e.g., ch10)")
    parser.add_argument("--lab", type=str, help="Specific lab to audit (e.g., decode_optimization)")
    parser.add_argument("--all", action="store_true", help="Audit all chapters and labs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    code_dir = repo_root
    
    total_compliant = 0
    total_needs_work = 0
    total_errors = 0
    
    # Audit chapters
    if args.chapter:
        chapters = [args.chapter]
    elif args.all or not args.lab:
        chapters = [f"ch{i:02d}" for i in range(1, 21)]
    else:
        chapters = []
    
    for chapter in chapters:
        chapter_dir = code_dir / chapter
        if not chapter_dir.exists():
            continue
        
        results = audit_directory(chapter_dir)
        if results:
            c, n, e = print_summary(results, f"{chapter.upper()}")
            total_compliant += c
            total_needs_work += n
            total_errors += e
    
    # Audit labs
    if args.lab:
        labs = [args.lab]
    elif args.all or not args.chapter:
        labs_dir = code_dir / "labs"
        if labs_dir.exists():
            labs = [d.name for d in labs_dir.iterdir() if d.is_dir()]
        else:
            labs = []
    else:
        labs = []
    
    for lab in sorted(labs):
        lab_dir = code_dir / "labs" / lab
        if not lab_dir.exists():
            continue
        
        results = audit_directory(lab_dir)
        if results:
            c, n, e = print_summary(results, f"LAB: {lab}")
            total_compliant += c
            total_needs_work += n
            total_errors += e
    
    # Grand total
    print(f"\n{'='*60}")
    print("GRAND TOTAL")
    print(f"{'='*60}")
    print(f"✅ Compliant: {total_compliant}")
    print(f"⚠️  Needs work: {total_needs_work}")
    print(f"❌ Errors: {total_errors}")
    print(f"Total: {total_compliant + total_needs_work + total_errors}")
    
    coverage = (total_compliant / max(1, total_compliant + total_needs_work)) * 100
    print(f"\n📊 Coverage: {coverage:.1f}%")


if __name__ == "__main__":
    main()
