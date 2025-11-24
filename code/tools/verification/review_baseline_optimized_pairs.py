#!/usr/bin/env python3
"""Review all baseline/optimized pairs for questionable practices.

Checks for:
- sleep() calls in benchmark code
- Artificial delays
- Mismatched workloads
- Unfair comparisons
- Missing synchronizations
- Other questionable practices
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.python.discovery import discover_benchmark_pairs


def _get_attr_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def _get_call_attr_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _eval_numeric_expr(node: ast.AST, attr_values: Dict[str, float], local_values: Dict[str, float]) -> Optional[float]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        return None
    if isinstance(node, ast.Name):
        return local_values.get(node.id)
    if isinstance(node, ast.Attribute):
        attr_name = _get_attr_name(node)
        if attr_name is not None:
            return attr_values.get(attr_name)
        return None
    if isinstance(node, ast.UnaryOp):
        operand = _eval_numeric_expr(node.operand, attr_values, local_values)
        if operand is None:
            return None
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        return None
    if isinstance(node, ast.BinOp):
        left = _eval_numeric_expr(node.left, attr_values, local_values)
        right = _eval_numeric_expr(node.right, attr_values, local_values)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right if right != 0 else None
        if isinstance(node.op, ast.FloorDiv):
            return float(int(left) // int(right)) if right != 0 else None
        if isinstance(node.op, ast.Mod):
            return left % right if right != 0 else None
        if isinstance(node.op, ast.Pow):
            return left ** right
        return None
    if isinstance(node, ast.Call):
        func_name = _get_call_attr_name(node.func)
        if func_name in {"float", "int"} and len(node.args) == 1:
            arg_val = _eval_numeric_expr(node.args[0], attr_values, local_values)
            if arg_val is None:
                return None
            return float(arg_val) if func_name == "float" else float(int(arg_val))
        if func_name in {"max", "min"} and node.args:
            evaluated_args = [
                _eval_numeric_expr(arg, attr_values, local_values)
                for arg in node.args
            ]
            if any(val is None for val in evaluated_args):
                return None
            return float(max(evaluated_args) if func_name == "max" else min(evaluated_args))
        return None
    if isinstance(node, ast.IfExp):
        # try both branches; only keep result if both resolve to same value
        true_val = _eval_numeric_expr(node.body, attr_values, local_values)
        false_val = _eval_numeric_expr(node.orelse, attr_values, local_values)
        if true_val is not None and false_val is not None and true_val == false_val:
            return true_val
        return None
    return None


def _maybe_extract_metadata_from_call(
    call: ast.Call,
    attr_values: Dict[str, float],
    local_values: Dict[str, float],
) -> Optional[Dict[str, float]]:
    if isinstance(call.func, ast.Name) and call.func.id == "WorkloadMetadata":
        keywords = call.keywords
    elif isinstance(call.func, ast.Attribute) and call.func.attr == "register_workload_metadata":
        keywords = call.keywords
    else:
        return None
    metadata: Dict[str, float] = {}
    for kw in keywords:
        if kw.arg is None:
            continue
        value = _eval_numeric_expr(kw.value, attr_values, local_values)
        if value is not None:
            metadata[kw.arg] = value
    return metadata or None


class _MethodAnalyzer:
    def __init__(self, attr_values: Dict[str, float]):
        self.attr_values = dict(attr_values)
        self.local_values: Dict[str, float] = {}
        self.metadata: List[Dict[str, float]] = []

    def process(self, func_def: ast.FunctionDef) -> None:
        self._process_block(func_def.body)

    def _process_block(self, statements: List[ast.stmt]) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                self._handle_assign(stmt)
            elif isinstance(stmt, ast.AnnAssign):
                self._handle_ann_assign(stmt)
            elif isinstance(stmt, ast.Expr):
                self._handle_expr(stmt.value)
            elif isinstance(stmt, ast.If):
                self._process_block(stmt.body)
                self._process_block(stmt.orelse)
            elif isinstance(stmt, ast.With):
                self._process_block(stmt.body)
            elif isinstance(stmt, ast.For):
                self._process_block(stmt.body)
                self._process_block(stmt.orelse)
            elif isinstance(stmt, ast.While):
                self._process_block(stmt.body)
                self._process_block(stmt.orelse)
            elif isinstance(stmt, ast.Try):
                self._process_block(stmt.body)
                for handler in stmt.handlers:
                    self._process_block(handler.body)
                self._process_block(stmt.orelse)
                self._process_block(stmt.finalbody)

    def _handle_assign(self, node: ast.Assign) -> None:
        self._handle_expr(node.value)
        value = _eval_numeric_expr(node.value, self.attr_values, self.local_values)
        if value is None:
            return
        for target in node.targets:
            self._assign_target(target, value)

    def _handle_ann_assign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            return
        self._handle_expr(node.value)
        value = _eval_numeric_expr(node.value, self.attr_values, self.local_values)
        if value is None:
            return
        self._assign_target(node.target, value)

    def _assign_target(self, target: ast.AST, value: float) -> None:
        if isinstance(target, ast.Attribute):
            attr_name = _get_attr_name(target)
            if attr_name is not None:
                self.attr_values[attr_name] = value
        elif isinstance(target, ast.Name):
            self.local_values[target.id] = value

    def _handle_expr(self, expr: ast.AST) -> None:
        if isinstance(expr, ast.Call):
            metadata = _maybe_extract_metadata_from_call(expr, self.attr_values, self.local_values)
            if metadata:
                self.metadata.append(metadata)


class _ClassAnalyzer:
    def __init__(self):
        self.attr_values: Dict[str, float] = {}
        self.metadata: List[Dict[str, float]] = []

    def process(self, node: ast.ClassDef) -> None:
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name in {"__init__", "setup"}:
                analyzer = _MethodAnalyzer(self.attr_values)
                analyzer.process(item)
                self.attr_values.update(analyzer.attr_values)
                self.metadata.extend(analyzer.metadata)


def extract_workload_metadata_from_source(content: str) -> Optional[Dict[str, float]]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    extractor = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_analyzer = _ClassAnalyzer()
            class_analyzer.process(node)
            extractor.extend(class_analyzer.metadata)
    return extractor[0] if extractor else None


class CodeReviewer:
    """Review code for questionable practices."""
    
    def __init__(self):
        self.issues: List[Dict[str, str]] = []
    
    def check_file(self, file_path: Path, pair_type: str) -> List[Dict[str, str]]:
        """Check a single file for issues."""
        file_issues = []
        
        try:
            content = file_path.read_text()
        except Exception as e:
            file_issues.append({
                "file": str(file_path),
                "type": "error",
                "severity": "high",
                "message": f"Could not read file: {e}"
            })
            return file_issues
        
        # Check for sleep calls
        if re.search(r'\btime\.sleep\s*\(', content):
            # Check if it's in benchmark_fn or setup/teardown
            if self._is_in_benchmark_code(content, 'time.sleep'):
                file_issues.append({
                    "file": str(file_path),
                    "type": "sleep_call",
                    "severity": "critical",
                    "message": "time.sleep() found in benchmark code - this artificially inflates timing"
                })
        
        # Check for artificial delays
        delay_patterns = [
            (r'sleep\s*\(\s*[\d.]+\s*\)', "sleep() call"),
            (r'time\.sleep\s*\(', "time.sleep() call"),
            (r'asyncio\.sleep\s*\(', "asyncio.sleep() call"),
        ]
        for pattern, desc in delay_patterns:
            if re.search(pattern, content):
                if self._is_in_benchmark_code(content, pattern):
                    file_issues.append({
                        "file": str(file_path),
                        "type": "artificial_delay",
                        "severity": "critical",
                        "message": f"{desc} found in benchmark code"
                    })
        
        # Check for missing synchronizations in optimized but present in baseline
        # This is harder - we'll do pair-wise comparison
        
        # Check for suspicious patterns
        suspicious_patterns = [
            (r'#.*fake|#.*fake|#.*dummy|#.*mock', "Suspicious comment suggesting fake/dummy code"),
            (r'pass\s*#.*benchmark', "Empty benchmark function"),
        ]
        for pattern, desc in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                file_issues.append({
                    "file": str(file_path),
                    "type": "suspicious_pattern",
                    "severity": "medium",
                    "message": desc
                })
        
        return file_issues
    
    def _is_in_benchmark_code(self, content: str, pattern: str) -> bool:
        """Check if pattern appears in benchmark code (setup/benchmark_fn/teardown)."""
        # Simple heuristic: check if it's in a method that's likely benchmark code
        # Look for def setup, def benchmark_fn, def teardown
        lines = content.split('\n')
        in_benchmark_method = False
        method_depth = 0
        
        for i, line in enumerate(lines):
            # Track method definitions
            if re.match(r'\s*def\s+(setup|benchmark_fn|teardown)', line):
                in_benchmark_method = True
                method_depth = len(line) - len(line.lstrip())
            elif in_benchmark_method:
                # Check if we've left the method (dedented)
                current_depth = len(line) - len(line.lstrip()) if line.strip() else method_depth
                if current_depth <= method_depth and line.strip() and not line.strip().startswith('#'):
                    in_benchmark_method = False
                
                # Check if pattern is in this method
                if re.search(pattern, line):
                    return True
        
        return False
    
    def compare_pair(self, baseline_path: Path, optimized_paths: List[Path]) -> List[Dict[str, str]]:
        """Compare a baseline/optimized pair for fairness."""
        pair_issues = []
        
        try:
            baseline_content = baseline_path.read_text()
        except Exception as e:
            pair_issues.append({
                "file": f"{baseline_path} (pair)",
                "type": "error",
                "severity": "high",
                "message": f"Could not read baseline: {e}"
            })
            return pair_issues
        
        for opt_path in optimized_paths:
            try:
                opt_content = opt_path.read_text()
            except Exception as e:
                pair_issues.append({
                    "file": f"{opt_path} (pair)",
                    "type": "error",
                    "severity": "high",
                    "message": f"Could not read optimized: {e}"
                })
                continue
            
            # Check for workload mismatches
            baseline_workload = self._extract_workload_info(baseline_content)
            opt_workload = self._extract_workload_info(opt_content)
            
            if baseline_workload and opt_workload:
                if baseline_workload != opt_workload:
                    # Check if difference is reasonable (within 10%)
                    if not self._workloads_similar(baseline_workload, opt_workload):
                        pair_issues.append({
                            "file": f"{baseline_path} vs {opt_path}",
                            "type": "workload_mismatch",
                            "severity": "medium",
                            "message": f"Workload mismatch: baseline={baseline_workload}, optimized={opt_workload}"
                        })
            
            # Check for synchronization mismatches
            baseline_syncs = self._count_synchronizations(baseline_content)
            opt_syncs = self._count_synchronizations(opt_content)
            
            # Optimized should have same or fewer syncs (not more, which would slow it down unfairly)
            if opt_syncs > baseline_syncs * 1.5:  # Allow some variance
                pair_issues.append({
                    "file": f"{baseline_path} vs {opt_path}",
                    "type": "sync_mismatch",
                    "severity": "low",
                    "message": f"Optimized has more synchronizations ({opt_syncs} vs {baseline_syncs}) - may be unfair"
                })
            
            # Check if optimized is doing less work (unfair)
            baseline_ops = self._estimate_operations(baseline_content)
            opt_ops = self._estimate_operations(opt_content)
            
            if opt_ops < baseline_ops * 0.8:  # Optimized doing significantly less work
                pair_issues.append({
                    "file": f"{baseline_path} vs {opt_path}",
                    "type": "work_reduction",
                    "severity": "medium",
                    "message": f"Optimized appears to do less work - may be unfair comparison"
                })
        
        return pair_issues
    
    def _extract_workload_info(self, content: str) -> Optional[Dict[str, float]]:
        """Extract workload information using metadata when available."""
        metadata = extract_workload_metadata_from_source(content)
        if metadata:
            return metadata
        return self._extract_workload_info_from_regex(content)

    def _extract_workload_info_from_regex(self, content: str) -> Optional[Dict[str, float]]:
        workload = {}
        patterns = [
            (r'self\.N\s*=\s*(\d+)', 'N'),
            (r'batch_size\s*=\s*(\d+)', 'batch_size'),
            (r'size\s*=\s*(\d+)', 'size'),
            (r'hidden_size\s*=\s*(\d+)', 'hidden_size'),
        ]
        for pattern, key in patterns:
            match = re.search(pattern, content)
            if match:
                workload[key] = float(match.group(1))
        return workload if workload else None
    
    def _workloads_similar(self, w1: Dict[str, float], w2: Dict[str, float]) -> bool:
        """Check if workloads are similar (within 10%) using overlapping keys only."""
        comparable_keys = {
            "requests_per_iteration",
            "tokens_per_iteration",
            "samples_per_iteration",
            "bytes_per_iteration",
            "custom_units_per_iteration",
            "batch_size",
            "N",
            "size",
            "hidden_size",
        }
        overlap = [k for k in comparable_keys if k in w1 and k in w2]
        if not overlap:
            return True
        for key in overlap:
            base = w1[key]
            opt = w2[key]
            denom = max(abs(base), abs(opt), 1e-9)
            if denom == 0:
                continue
            diff = abs(base - opt) / denom
            if diff > 0.1:
                return False
        return True
    
    def _count_synchronizations(self, content: str) -> int:
        """Count synchronization calls."""
        patterns = [
            r'torch\.cuda\.synchronize',
            r'\.synchronize\s*\(',
            r'\.wait\s*\(',
        ]
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, content))
        return count
    
    def _estimate_operations(self, content: str) -> int:
        """Rough estimate of operations in benchmark_fn."""
        # Count tensor operations, function calls in benchmark_fn
        benchmark_fn_match = re.search(r'def\s+benchmark_fn.*?(?=def\s|\Z)', content, re.DOTALL)
        if not benchmark_fn_match:
            return 0
        
        benchmark_code = benchmark_fn_match.group(0)
        
        # Count operations
        op_count = 0
        op_patterns = [
            r'\.\w+\s*\(',  # Method calls
            r'\+\s*|\-\s*|\*\s*|/\s*',  # Arithmetic ops
            r'torch\.',  # Torch operations
        ]
        
        for pattern in op_patterns:
            op_count += len(re.findall(pattern, benchmark_code))
        
        return op_count


def main():
    """Main review function."""
    print("Discovering benchmark pairs...")
    pairs = discover_benchmark_pairs(REPO_ROOT, "all")
    print(f"Found {len(pairs)} pairs to review\n")
    
    reviewer = CodeReviewer()
    all_issues = []
    
    for baseline_path, optimized_paths, example_name in pairs:
        print(f"Reviewing: {example_name} ({baseline_path.name})")
        
        # Check baseline
        baseline_issues = reviewer.check_file(baseline_path, "baseline")
        all_issues.extend(baseline_issues)
        
        # Check optimized files
        for opt_path in optimized_paths:
            opt_issues = reviewer.check_file(opt_path, "optimized")
            all_issues.extend(opt_issues)
        
        # Compare pair
        pair_issues = reviewer.compare_pair(baseline_path, optimized_paths)
        all_issues.extend(pair_issues)
    
    # Report issues
    print("\n" + "="*80)
    print("REVIEW SUMMARY")
    print("="*80)
    
    if not all_issues:
        print("✓ No issues found!")
        return 0
    
    # Group by severity
    critical = [i for i in all_issues if i['severity'] == 'critical']
    high = [i for i in all_issues if i['severity'] == 'high']
    medium = [i for i in all_issues if i['severity'] == 'medium']
    low = [i for i in all_issues if i['severity'] == 'low']
    
    print(f"\nCritical issues: {len(critical)}")
    for issue in critical:
        print(f"  [{issue['type']}] {issue['file']}: {issue['message']}")
    
    print(f"\nHigh severity issues: {len(high)}")
    for issue in high:
        print(f"  [{issue['type']}] {issue['file']}: {issue['message']}")
    
    print(f"\nMedium severity issues: {len(medium)}")
    for issue in medium[:20]:  # Limit output
        print(f"  [{issue['type']}] {issue['file']}: {issue['message']}")
    if len(medium) > 20:
        print(f"  ... and {len(medium) - 20} more")
    
    print(f"\nLow severity issues: {len(low)}")
    for issue in low[:10]:  # Limit output
        print(f"  [{issue['type']}] {issue['file']}: {issue['message']}")
    if len(low) > 10:
        print(f"  ... and {len(low) - 10} more")
    
    return 1 if critical or high else 0


if __name__ == "__main__":
    sys.exit(main())

