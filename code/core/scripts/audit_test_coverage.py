#!/usr/bin/env python3
"""
Audit test coverage for all 94 validity issues / anti-cheat protections.

This script:
1. Extracts all documented protections from README.md
2. Extracts all test functions from verification test files
3. Maps tests to protections
4. Reports coverage gaps
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

# Categories and their protections (from README.md table)
PROTECTIONS = {
    # TIMING (7 issues)
    "Timing": [
        ("Unsynced Streams", "full_device_sync|StreamAuditor|stream_sync"),
        ("Incomplete Async Ops", "full_device_sync|async.*sync"),
        ("Event Timing Gaps", "cross_validate.*timing|timing_cross"),
        ("Timer Granularity", "adaptive_iterations|granularity"),
        ("Warmup Bleed", "isolate_warmup|warmup.*isolation"),
        ("Clock Drift", "monotonic"),
        ("Profiler Overhead", "profile.*overhead|profiler"),
    ],
    
    # OUTPUT (10 issues)
    "Output": [
        ("Constant Output", "jitter.*check|constant.*output"),
        ("Stale Cache", "fresh.*input|stale.*cache"),
        ("Approximation Drift", "tolerance|approximation"),
        ("Invalid Values NaN", "nan|validate_result"),
        ("Invalid Values Inf", "inf|validate_result"),
        ("Invalid Ground Truth", "golden.*output|ground.*truth"),
        ("Shape Mismatch", "shape.*mismatch|shape.*valid"),
        ("Dtype Mismatch", "dtype|tolerance.*spec"),
        ("Denormalized Values", "denormal|subnormal"),
        ("Uninitialized Memory", "uninit.*mem|memory.*init"),
    ],
    
    # WORKLOAD (11 issues)
    "Workload": [
        ("Precision Mismatch", "precision|input.*signature.*dtype"),
        ("Undeclared Shortcuts", "workload.*invariant|shortcut"),
        ("Early Exit", "config.*immutab|early.*exit"),
        ("Batch Shrinking", "input.*signature.*batch"),
        ("Sequence Truncation", "input.*signature.*seq|truncat"),
        ("Hidden Downsampling", "dimension|downsample"),
        ("Sparsity Mismatch", "sparsity"),
        ("Attention Mask Mismatch", "mask|attention"),
        ("KV Cache Size Mismatch", "kv.*cache|cache.*size"),
        ("Train/Test Overlap", "dataset|overlap|contaminat"),
    ],
    
    # LOCATION (7 issues)
    "Location": [
        ("CPU Spillover", "cpu|spillover|gpu.*kernel"),
        ("Setup Pre-computation", "setup.*precomputation|check_setup"),
        ("Graph Capture Cheat", "graph.*capture|GraphCaptureCheatDetector"),
        ("Warmup Computation", "isolate_warmup|warmup.*compute"),
        ("Background Thread", "process.*isolation|background"),
        ("Lazy Evaluation Skip", "force_tensor_evaluation|lazy"),
        ("JIT Compilation Timing", "compile.*cache|jit"),
    ],
    
    # MEMORY (7 issues)
    "Memory": [
        ("Pre-allocated Output", "memory.*allocation|pre.*alloc"),
        ("Input-Output Aliasing", "input.*output.*alias|aliasing"),
        ("Pinned Memory Timing", "pinned|transfer"),
        ("Memory Pool Reuse", "memory.*pool|reset.*cuda"),
        ("Fragmentation Effects", "fragment"),
        ("Page Fault Timing", "page.*fault"),
        ("Swap Interference", "swap"),
    ],
    
    # CUDA (10 issues)
    "CUDA": [
        ("Host Callback Escape", "host.*callback|cudaLaunchHostFunc"),
        ("Async Memcpy Incomplete", "async.*memcpy|memcpy.*sync"),
        ("Workspace Pre-compute", "workspace"),
        ("Persistent Kernel", "persistent"),
        ("Undeclared Multi-GPU", "validate_environment|multi.*gpu"),
        ("Context Switch Overhead", "context.*switch"),
        ("Driver Overhead", "driver"),
        ("Cooperative Launch Abuse", "cooperative"),
        ("Dynamic Parallelism Hidden", "dynamic.*parallel|cdp"),
        ("Unified Memory Faults", "unified.*memory|um.*fault"),
    ],
    
    # COMPILE (7 issues)
    "Compile": [
        ("Compilation Cache Hit", "compile.*cache|clear_compile"),
        ("Trace Reuse", "dynamo.*reset|trace"),
        ("Mode Inconsistency", "mode.*consist"),
        ("Inductor Asymmetry", "inductor"),
        ("Guard Failure Hidden", "guard|compile.*state"),
        ("Autotuning Variance", "autotun"),
        ("Symbolic Shape Exploit", "symbolic|shape"),
    ],
    
    # DISTRIBUTED (8 issues)
    "Distributed": [
        ("Rank Skipping", "rank.*skip|check_rank_execution"),
        ("Collective Short-circuit", "collective|nccl"),
        ("Topology Mismatch", "topology|verify_distributed"),
        ("Barrier Timing", "barrier"),
        ("Gradient Bucketing Mismatch", "bucket|gradient"),
        ("Async Gradient Timing", "async.*gradient|allreduce"),
        ("Pipeline Bubble Hiding", "pipeline|bubble"),
        ("Shard Size Mismatch", "shard|fsdp"),
    ],
    
    # ENVIRONMENT (12 issues)
    "Environment": [
        ("Device Mismatch", "validate_environment|device.*mismatch"),
        ("Frequency Boost", "lock_gpu_clocks|frequency"),
        ("Priority Elevation", "priority"),
        ("Memory Overcommit", "overcommit"),
        ("NUMA Inconsistency", "numa"),
        ("CPU Governor Mismatch", "governor"),
        ("Thermal Throttling", "thermal|capture_gpu_state|temperature"),
        ("Power Limit Difference", "power|tdp|capture_gpu_state"),
        ("Driver Version Mismatch", "driver.*version|manifest"),
        ("Library Version Mismatch", "library.*version|manifest"),
        ("Container Resource Limits", "container|cgroup"),
        ("Virtualization Overhead", "virtual|vm"),
    ],
    
    # STATISTICAL (8 issues)
    "Statistical": [
        ("Cherry-picking", "cherry|all.*iteration"),
        ("Outlier Injection", "outlier"),
        ("Variance Gaming", "variance"),
        ("Percentile Selection", "percentile"),
        ("Insufficient Samples", "adaptive.*iteration|sample.*size"),
        ("Cold Start Inclusion", "warmup.*enforce"),
        ("GC Interference", "gc_disabled|garbage"),
        ("Background Process Noise", "process.*isolation|background"),
    ],
    
    # EVALUATION (7 issues)
    "Evaluation": [
        ("Eval Code Exploitation", "contract|benchmark.*contract"),
        ("Timeout Manipulation", "config.*immutab|timeout"),
        ("Metric Definition Gaming", "metric.*defin"),
        ("Test Data Leakage", "contaminat|leakage"),
        ("Benchmark Overfitting", "fresh.*input|jitter|overfit"),
        ("Self-Modifying Tests", "config.*immutab|self.*modify"),
    ],
}


def extract_test_names(test_dir: Path) -> Dict[str, str]:
    """Extract all test function names and their docstrings."""
    tests = {}
    test_files = (
        list(test_dir.glob("test_verification*.py")) + 
        list(test_dir.glob("test_benchmark_verification.py")) +
        list(test_dir.glob("test_anti_cheat*.py"))
    )
    
    for test_file in test_files:
        content = test_file.read_text()
        # Find all test functions
        pattern = r'def (test_\w+)\(.*?\):\s*(?:"""([^"]*?)""")?'
        for match in re.finditer(pattern, content, re.DOTALL):
            name = match.group(1)
            doc = match.group(2) or ""
            tests[name] = doc.strip().replace('\n', ' ')
    
    return tests


def match_test_to_protection(test_name: str, test_doc: str, patterns: List[str]) -> bool:
    """Check if a test matches any of the protection patterns."""
    combined = f"{test_name} {test_doc}".lower()
    for pattern in patterns:
        if re.search(pattern.lower(), combined):
            return True
    return False


def audit_coverage(code_root: Path) -> Tuple[Dict[str, List], Dict[str, List], int, int]:
    """Audit test coverage for all protections."""
    test_dir = code_root / "tests"
    tests = extract_test_names(test_dir)
    
    covered = {}
    uncovered = {}
    total_protections = 0
    covered_count = 0
    
    for category, protections in PROTECTIONS.items():
        covered[category] = []
        uncovered[category] = []
        
        for protection_name, patterns in protections:
            total_protections += 1
            patterns_list = patterns.split("|")
            
            matching_tests = []
            for test_name, test_doc in tests.items():
                if match_test_to_protection(test_name, test_doc, patterns_list):
                    matching_tests.append(test_name)
            
            if matching_tests:
                covered[category].append((protection_name, matching_tests))
                covered_count += 1
            else:
                uncovered[category].append((protection_name, patterns_list))
    
    return covered, uncovered, covered_count, total_protections


def main():
    code_root = Path(__file__).parent.parent.parent
    
    print("=" * 80)
    print("ANTI-CHEAT TEST COVERAGE AUDIT")
    print("=" * 80)
    print()
    
    covered, uncovered, covered_count, total = audit_coverage(code_root)
    
    # Summary
    print(f"SUMMARY: {covered_count}/{total} protections have tests ({100*covered_count/total:.1f}%)")
    print()
    
    # Covered protections
    print("-" * 80)
    print("COVERED PROTECTIONS (with matching tests):")
    print("-" * 80)
    for category, items in covered.items():
        if items:
            print(f"\n{category} ({len(items)}/{len(PROTECTIONS[category])} covered):")
            for protection, tests in items:
                print(f"  ✅ {protection}")
                for t in tests[:3]:  # Show max 3 tests
                    print(f"      → {t}")
                if len(tests) > 3:
                    print(f"      → ... and {len(tests)-3} more")
    
    # Uncovered protections
    print()
    print("-" * 80)
    print("UNCOVERED PROTECTIONS (need tests):")
    print("-" * 80)
    has_gaps = False
    for category, items in uncovered.items():
        if items:
            has_gaps = True
            print(f"\n{category} ({len(items)} missing):")
            for protection, patterns in items:
                print(f"  ❌ {protection}")
                print(f"      Keywords: {', '.join(patterns)}")
    
    if not has_gaps:
        print("\n  (none - all protections have tests!)")
    
    # Generate test recommendations
    print()
    print("-" * 80)
    print("RECOMMENDED TESTS TO ADD:")
    print("-" * 80)
    
    test_count = 0
    for category, items in uncovered.items():
        for protection, patterns in items:
            test_count += 1
            func_name = protection.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
            print(f"""
def test_{func_name}_detection():
    \"\"\"Test that {protection} is detected and prevented.
    
    Category: {category}
    Protection: Verify the harness detects and handles {protection}.
    \"\"\"
    # TODO: Implement test for {protection}
    pass
""")
    
    print()
    print("=" * 80)
    print(f"TOTAL: {covered_count}/{total} covered | {total - covered_count} tests needed")
    print("=" * 80)
    
    return covered_count, total


if __name__ == "__main__":
    covered, total = main()
    sys.exit(0 if covered >= total * 0.9 else 1)  # Fail if <90% covered

