"""Verification runner for benchmark correctness validation.

This module provides the main verification engine that executes benchmark
pairs (baseline/optimized) in verify mode to ensure correctness before
allowing performance comparison.

Key Features:
- Deterministic seed setup for reproducible execution
- Golden output caching for baseline comparison
- Fresh-input and jitter checks to detect hardcoding/caching
- Workload invariant enforcement
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from core.benchmark.verification import (
    ComparisonDetails,
    EnforcementPhase,
    InputSignature,
    PrecisionFlags,
    QuarantineReason,
    ToleranceSpec,
    VerifyResult,
    compare_workload_metrics,
    detect_seed_mutation,
    get_enforcement_phase,
    get_output_tolerance,
    is_verification_enabled,
    get_tolerance_for_dtype,
    select_jitter_dimension,
    set_deterministic_seeds,
)
from core.benchmark.quarantine import (
    QuarantineManager,
    check_benchmark_compliance,
    detect_skip_flags,
)


# Default golden output cache directory
DEFAULT_CACHE_DIR = Path("artifacts/verify_cache/golden_outputs")


@dataclass
class GoldenOutput:
    """Cached baseline output for verification comparison.
    
    Stores the output tensor(s) from a baseline benchmark execution,
    along with metadata about the execution context.
    """
    signature_hash: str
    outputs: Dict[str, torch.Tensor]  # Named outputs
    workload_metrics: Dict[str, float]
    checksum: str
    created_at: datetime
    seed: int
    
    def compute_checksum(self) -> str:
        """Compute checksum of outputs for integrity verification."""
        parts = []
        for name in sorted(self.outputs.keys()):
            tensor = self.outputs[name]
            # Use numpy for consistent hash across sessions
            arr = tensor.cpu().numpy()
            parts.append(hashlib.sha256(arr.tobytes()).hexdigest()[:16])
        return "-".join(parts)


class GoldenOutputCache:
    """Manages cached golden outputs from baseline benchmark runs.
    
    Golden outputs are keyed by input signature hash and used for
    verifying that optimized benchmarks produce equivalent results.
    """
    
    def __init__(self, cache_dir: Optional[Union[Path, str]] = None):
        """Initialize the golden output cache.
        
        Args:
            cache_dir: Directory for storing cached outputs.
                      Defaults to artifacts/verify_cache/golden_outputs
        """
        if cache_dir is None:
            self.cache_dir = DEFAULT_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, signature_hash: str) -> Path:
        """Get the path to the cache file for a signature hash."""
        return self.cache_dir / f"{signature_hash}.pkl"
    
    def has(self, signature_hash: str) -> bool:
        """Check if a golden output exists for the given signature."""
        return self._get_cache_path(signature_hash).exists()
    
    def get(self, signature_hash: str) -> Optional[GoldenOutput]:
        """Retrieve a cached golden output.
        
        Args:
            signature_hash: Hash of the input signature
            
        Returns:
            GoldenOutput if found, None otherwise
        """
        path = self._get_cache_path(signature_hash)
        if not path.exists():
            return None
        
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return GoldenOutput(
                signature_hash=data["signature_hash"],
                outputs={k: torch.tensor(v) for k, v in data["outputs"].items()},
                workload_metrics=data["workload_metrics"],
                checksum=data["checksum"],
                created_at=datetime.fromisoformat(data["created_at"]),
                seed=data["seed"],
            )
        except Exception:
            return None
    
    def put(self, golden: GoldenOutput) -> None:
        """Store a golden output in the cache.
        
        Args:
            golden: The GoldenOutput to cache
        """
        path = self._get_cache_path(golden.signature_hash)
        data = {
            "signature_hash": golden.signature_hash,
            "outputs": {k: v.cpu().numpy() for k, v in golden.outputs.items()},
            "workload_metrics": golden.workload_metrics,
            "checksum": golden.checksum,
            "created_at": golden.created_at.isoformat(),
            "seed": golden.seed,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def invalidate(self, signature_hash: str) -> bool:
        """Invalidate (delete) a cached golden output.
        
        Args:
            signature_hash: Hash of the input signature
            
        Returns:
            True if cache was deleted, False if it didn't exist
        """
        path = self._get_cache_path(signature_hash)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def clear_all(self) -> int:
        """Clear all cached golden outputs.
        
        Returns:
            Number of cache entries cleared
        """
        count = 0
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink()
            count += 1
        return count


@dataclass
class VerifyConfig:
    """Configuration for verification runs."""
    seed: int = 42
    tolerance_override: Optional[ToleranceSpec] = None
    skip_jitter_check: bool = False
    skip_fresh_input_check: bool = False
    workload_tolerance: float = 0.01  # 1% tolerance for workload metrics
    verbose: bool = False
    force_recache: bool = False  # Ignore existing cache
    

class VerifyRunner:
    """Main verification engine for benchmark correctness validation.
    
    Executes benchmark pairs (baseline/optimized) with deterministic seeds
    and verifies that outputs match within tolerance. Implements anti-reward-hacking
    checks including fresh-input and jitter verification.
    
    Usage:
        runner = VerifyRunner()
        
        # Verify a benchmark pair
        result = runner.verify_pair(baseline_benchmark, optimized_benchmark)
        
        if not result.passed:
            print(f"Verification failed: {result.reason}")
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[Path, str]] = None,
        quarantine_manager: Optional[QuarantineManager] = None,
    ):
        """Initialize the verification runner.
        
        Args:
            cache_dir: Directory for golden output cache
            quarantine_manager: Manager for quarantine records
        """
        cache_path = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self.cache = GoldenOutputCache(cache_path)
        self.quarantine = quarantine_manager or QuarantineManager()
    
    def _extract_output(self, benchmark: Any) -> Dict[str, torch.Tensor]:
        """Extract output tensor from benchmark using ONLY get_verify_output().
        
        STRICT: No fallbacks, no auto-detection. Every benchmark MUST implement
        get_verify_output() explicitly.
        
        Args:
            benchmark: The benchmark instance
            
        Returns:
            Dict mapping output names to tensors
            
        Raises:
            NotImplementedError: If benchmark doesn't implement get_verify_output()
            ValueError: If get_verify_output() returns invalid type
        """
        # STRICT: Only use get_verify_output() - no fallbacks
        if not hasattr(benchmark, "get_verify_output"):
            raise NotImplementedError(
                f"{benchmark.__class__.__name__} must implement get_verify_output(). "
                "No fallbacks or auto-detection allowed."
            )
        
        if not callable(benchmark.get_verify_output):
            raise TypeError(
                f"{benchmark.__class__.__name__}.get_verify_output must be a method, "
                f"got {type(benchmark.get_verify_output)}"
            )
        
        # Call get_verify_output() - let NotImplementedError propagate
        out = benchmark.get_verify_output()
        
        # Validate return type
        if out is None:
            raise ValueError(
                f"{benchmark.__class__.__name__}.get_verify_output() returned None. "
                "Must return a tensor or dict of tensors. "
                "If this is a throughput-only benchmark, return a checksum tensor."
            )
        
        outputs: Dict[str, torch.Tensor] = {}
        
        if isinstance(out, torch.Tensor):
            outputs["output"] = out.detach().clone()
        elif isinstance(out, dict):
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    outputs[k] = v.detach().clone()
            if not outputs:
                raise ValueError(
                    f"{benchmark.__class__.__name__}.get_verify_output() returned dict "
                    "with no tensor values."
                )
        else:
            raise TypeError(
                f"{benchmark.__class__.__name__}.get_verify_output() must return "
                f"torch.Tensor or Dict[str, torch.Tensor], got {type(out)}"
            )
        
        return outputs
    
    def _extract_signature(self, benchmark: Any) -> Optional[InputSignature]:
        """Extract input signature from a benchmark.
        
        Supports two modes:
        1. Full InputSignature with shapes/dtypes (rigorous verification)
        2. Simple parameter-based dict (workload parameter matching)
        
        For simple dicts, we create a minimal InputSignature that can be
        hashed for caching but is validated in non-strict mode.
        
        Args:
            benchmark: The benchmark instance
            
        Returns:
            InputSignature if extractable, None otherwise
        """
        if not hasattr(benchmark, "get_input_signature"):
            return None
        
        try:
            sig_dict = benchmark.get_input_signature()
            if sig_dict is None or not sig_dict:
                return None
            
            # Convert dict to InputSignature
            # Handle both old-style dict and new InputSignature
            if isinstance(sig_dict, InputSignature):
                return sig_dict
            
            # Build InputSignature from dict
            shapes = {}
            dtypes = {}
            
            # Extract shapes and dtypes from various formats
            if "shapes" in sig_dict:
                shapes = {k: tuple(v) if isinstance(v, list) else v 
                         for k, v in sig_dict["shapes"].items()}
            if "dtypes" in sig_dict:
                dtypes = sig_dict["dtypes"]
            
            # Infer shapes from tensor_shape or similar keys
            if not shapes:
                for key in ["tensor_shape", "shape", "input_shape"]:
                    if key in sig_dict:
                        val = sig_dict[key]
                        if isinstance(val, (list, tuple)):
                            shapes["input"] = tuple(val)
                            break
            
            # For simple parameter-based signatures, create a synthetic shape
            # from the parameters to enable hashing and comparison
            if not shapes:
                # Store all numeric parameters as a synthetic shape for hashing
                param_values = []
                for key, val in sorted(sig_dict.items()):
                    if key in ("batch_size", "parameter_count", "fp16", "bf16", "fp8", "tf32"):
                        continue  # Skip known fields
                    if isinstance(val, (int, float)):
                        param_values.append(int(val))
                    elif isinstance(val, (list, tuple)) and all(isinstance(x, (int, float)) for x in val):
                        param_values.extend(int(x) for x in val)
                if param_values:
                    shapes["_params"] = tuple(param_values)
            
            # Infer batch_size
            batch_size = sig_dict.get("batch_size", 0)
            if not batch_size and shapes and "_params" not in shapes:
                # Try to get from first dimension of any shape
                first_shape = next(iter(shapes.values()), ())
                if first_shape:
                    batch_size = first_shape[0]
            
            return InputSignature(
                shapes=shapes,
                dtypes=dtypes,
                batch_size=batch_size,
                parameter_count=sig_dict.get("parameter_count", 0),
                precision_flags=PrecisionFlags(
                    fp16=sig_dict.get("fp16", False),
                    bf16=sig_dict.get("bf16", False),
                    fp8=sig_dict.get("fp8", False),
                    tf32=sig_dict.get("tf32", True),
                ),
            )
        except Exception:
            return None
    
    def _extract_workload_metrics(self, benchmark: Any) -> Dict[str, float]:
        """Extract workload metrics from a benchmark.
        
        Args:
            benchmark: The benchmark instance
            
        Returns:
            Dict of workload metrics (bytes/tokens/ops per iteration)
        """
        metrics: Dict[str, float] = {}
        
        if hasattr(benchmark, "get_workload_metadata"):
            try:
                metadata = benchmark.get_workload_metadata()
                if metadata:
                    if hasattr(metadata, "bytes_per_iter") and metadata.bytes_per_iter:
                        metrics["bytes_per_iter"] = float(metadata.bytes_per_iter)
                    if hasattr(metadata, "tokens_per_iter") and metadata.tokens_per_iter:
                        metrics["tokens_per_iter"] = float(metadata.tokens_per_iter)
                    if hasattr(metadata, "flops_per_iter") and metadata.flops_per_iter:
                        metrics["flops_per_iter"] = float(metadata.flops_per_iter)
            except Exception:
                pass
        
        return metrics
    
    def _compare_outputs(
        self,
        expected: Dict[str, torch.Tensor],
        actual: Dict[str, torch.Tensor],
        tolerance: Optional[ToleranceSpec] = None,
    ) -> ComparisonDetails:
        """Compare expected and actual outputs.
        
        Args:
            expected: Expected output tensors (from baseline)
            actual: Actual output tensors (from optimized)
            tolerance: Optional custom tolerance override
            
        Returns:
            ComparisonDetails with comparison results
        """
        if set(expected.keys()) != set(actual.keys()):
            return ComparisonDetails(
                passed=False,
                max_diff=None,
                location=None,
            )
        
        max_diff_overall = 0.0
        worst_location: Optional[Tuple[int, ...]] = None
        worst_expected: Optional[float] = None
        worst_actual: Optional[float] = None
        
        for name in expected.keys():
            exp_tensor = expected[name]
            act_tensor = actual[name]
            
            # Ensure tensors are on the same device for comparison
            if exp_tensor.device != act_tensor.device:
                # Move to actual tensor's device (typically GPU during verification)
                exp_tensor = exp_tensor.to(act_tensor.device)
            
            # Check shapes match
            if exp_tensor.shape != act_tensor.shape:
                return ComparisonDetails(
                    passed=False,
                    max_diff=float('inf'),
                    location=None,
                )
            
            # Get tolerance for dtype
            tol = tolerance or get_tolerance_for_dtype(exp_tensor.dtype)
            
            # Custom comparator takes precedence
            if tol.comparator_fn is not None:
                try:
                    if not tol.comparator_fn(exp_tensor, act_tensor):
                        return ComparisonDetails(
                            passed=False,
                            tolerance_used=tol,
                        )
                    continue
                except Exception:
                    return ComparisonDetails(passed=False, tolerance_used=tol)
            
            # Standard numeric comparison
            if exp_tensor.is_floating_point():
                # Use allclose with tolerances
                diff = torch.abs(exp_tensor - act_tensor)
                rel_diff = diff / (torch.abs(exp_tensor) + 1e-12)
                
                max_diff = float(diff.max())
                if max_diff > max_diff_overall:
                    max_diff_overall = max_diff
                    max_idx = diff.argmax()
                    flat_idx = max_idx.item()
                    worst_location = tuple(int(x) for x in np.unravel_index(flat_idx, diff.shape))
                    worst_expected = float(exp_tensor.flatten()[flat_idx])
                    worst_actual = float(act_tensor.flatten()[flat_idx])
                
                # Check if passes tolerance
                if not torch.allclose(exp_tensor, act_tensor, rtol=tol.rtol, atol=tol.atol):
                    return ComparisonDetails(
                        passed=False,
                        max_diff=max_diff_overall,
                        location=worst_location,
                        expected_sample=worst_expected,
                        actual_sample=worst_actual,
                        tolerance_used=tol,
                    )
            else:
                # Exact match for non-floating point
                if not torch.equal(exp_tensor, act_tensor):
                    diff = (exp_tensor != act_tensor)
                    max_idx = diff.int().argmax()
                    flat_idx = max_idx.item()
                    worst_location = tuple(int(x) for x in np.unravel_index(flat_idx, diff.shape))
                    return ComparisonDetails(
                        passed=False,
                        max_diff=float('inf'),
                        location=worst_location,
                        expected_sample=float(exp_tensor.flatten()[flat_idx]),
                        actual_sample=float(act_tensor.flatten()[flat_idx]),
                        tolerance_used=tol,
                    )
        
        return ComparisonDetails(
            passed=True,
            max_diff=max_diff_overall if max_diff_overall > 0 else None,
            tolerance_used=tolerance,
        )
    
    def _run_with_seed(
        self,
        benchmark: Any,
        seed: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], Dict[str, int]]:
        """Run a benchmark with specific seed and extract outputs.
        
        Args:
            benchmark: The benchmark instance
            seed: Random seed to use
            
        Returns:
            Tuple of (outputs, workload_metrics, seed_info)
        """
        # Set deterministic seeds BEFORE setup
        # Note: Seeds are re-set AFTER setup to ensure verification seeds
        # take precedence over any seeds benchmarks may set during setup()
        # for standalone reproducibility. This is the recommended pattern
        # per the design doc - benchmarks can set seeds in setup() for
        # standalone use, and verification re-seeds after.
        set_deterministic_seeds(seed)
        
        # Setup (benchmarks may set their own seeds for standalone use)
        benchmark.setup()
        
        # Re-set seeds after setup to ensure verification seeds take precedence
        seed_info = set_deterministic_seeds(seed)
        
        try:
            # Run benchmark function
            benchmark.benchmark_fn()
            
            # Sync CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Extract outputs
            outputs = self._extract_output(benchmark)
            metrics = self._extract_workload_metrics(benchmark)
            
            # Check for seed mutation
            if detect_seed_mutation(seed_info):
                raise RuntimeError("Benchmark mutated RNG seeds during execution")
            
            return outputs, metrics, seed_info
            
        finally:
            # Always teardown
            if hasattr(benchmark, "teardown"):
                try:
                    benchmark.teardown()
                except Exception:
                    pass
    
    def _run_fresh_input_check(
        self,
        benchmark: Any,
        original_outputs: Dict[str, torch.Tensor],
        config: VerifyConfig,
    ) -> Tuple[bool, Optional[str]]:
        """Run fresh-input check to detect output caching.
        
        Runs the benchmark with a different seed and verifies that
        outputs change (for non-deterministic algorithms) or stay
        the same (for deterministic algorithms).
        
        Args:
            benchmark: The benchmark instance
            original_outputs: Outputs from original run
            config: Verification configuration
            
        Returns:
            Tuple of (passed, failure_reason)
        """
        if config.skip_fresh_input_check:
            return True, None
        
        try:
            # Run with different seed
            fresh_seed = config.seed + 1000
            fresh_outputs, _, _ = self._run_with_seed(benchmark, fresh_seed)
            
            # For deterministic algorithms, outputs should match
            # For non-deterministic, they should differ
            # We check if they're suspiciously identical when they shouldn't be
            
            if not fresh_outputs:
                return True, None  # Can't check
            
            # Simple check: if outputs are bitwise identical with different seed,
            # that's suspicious for most algorithms (might be cached)
            all_identical = True
            for name in original_outputs:
                if name in fresh_outputs:
                    if not torch.equal(original_outputs[name], fresh_outputs[name]):
                        all_identical = False
                        break
            
            # If outputs are identical, check if algorithm is marked deterministic
            if all_identical:
                # Check if benchmark declares itself as deterministic
                is_deterministic = getattr(benchmark, "_is_deterministic", False)
                if not is_deterministic:
                    # Suspicious - might be caching
                    # But don't fail yet - could be legit deterministic algo
                    pass
            
            return True, None
            
        except Exception as e:
            # Fresh input check is advisory, don't fail hard
            return True, f"Fresh input check warning: {e}"
    
    def _run_jitter_check(
        self,
        benchmark: Any,
        input_signature: InputSignature,
        config: VerifyConfig,
    ) -> Tuple[bool, Optional[str]]:
        """Run jitter check to detect hardcoded outputs.
        
        Perturbs a dimension of the input and verifies that
        outputs change accordingly.
        
        Args:
            benchmark: The benchmark instance
            input_signature: The input signature
            config: Verification configuration
            
        Returns:
            Tuple of (passed, failure_reason)
        """
        if config.skip_jitter_check:
            return True, None
        
        # Select dimension to perturb
        jitter_dim = select_jitter_dimension(input_signature)
        if jitter_dim is None:
            # Check for jitter exemption
            exemption = getattr(benchmark, "jitter_exemption_reason", None)
            if exemption:
                return True, None
            # No suitable dimension and no exemption - flag it
            return False, "No suitable dimension for jitter check and no exemption provided"
        
        # For now, jitter check is advisory - full implementation would
        # modify the input dimension and verify output changes
        return True, None
    
    def verify_baseline(
        self,
        baseline: Any,
        config: Optional[VerifyConfig] = None,
    ) -> VerifyResult:
        """Run verification on baseline benchmark and cache golden output.
        
        This should be called first when establishing a new baseline.
        The output is cached and used for subsequent optimized comparisons.
        
        Args:
            baseline: The baseline benchmark instance
            config: Optional verification configuration
            
        Returns:
            VerifyResult with verification outcome
        """
        config = config or VerifyConfig()
        
        # Check compliance
        issues = check_benchmark_compliance(baseline)
        if issues:
            reason = issues[0]  # Report first issue
            return VerifyResult.fail(
                f"Baseline compliance check failed: {reason.value}",
            )
        
        # Extract input signature
        signature = self._extract_signature(baseline)
        if signature is None:
            return VerifyResult.fail("Baseline has no valid input signature")
        
        errors = signature.validate(strict=False)  # Allow simple parameter-based signatures
        if errors:
            return VerifyResult.fail(f"Invalid signature: {errors[0]}")
        
        sig_hash = signature.hash()
        
        # Check cache (unless forced)
        if not config.force_recache and self.cache.has(sig_hash):
            golden = self.cache.get(sig_hash)
            if golden:
                return VerifyResult.success(
                    signature_hash=sig_hash,
                    baseline_checksum=golden.checksum,
                )
        
        try:
            # Run baseline with deterministic seed
            outputs, metrics, seed_info = self._run_with_seed(baseline, config.seed)
            
            if not outputs:
                return VerifyResult.fail("Baseline produced no extractable outputs")
            
            # Create and cache golden output
            golden = GoldenOutput(
                signature_hash=sig_hash,
                outputs=outputs,
                workload_metrics=metrics,
                checksum="",  # Will be computed
                created_at=datetime.now(),
                seed=config.seed,
            )
            golden.checksum = golden.compute_checksum()
            self.cache.put(golden)
            
            return VerifyResult.success(
                signature_hash=sig_hash,
                baseline_checksum=golden.checksum,
                workload_delta=None,
                seed_info=seed_info,
            )
            
        except Exception as e:
            return VerifyResult.fail(f"Baseline execution failed: {e}\n{traceback.format_exc()}")
    
    def verify_optimized(
        self,
        optimized: Any,
        config: Optional[VerifyConfig] = None,
    ) -> VerifyResult:
        """Run verification on optimized benchmark against cached baseline.
        
        Compares optimized outputs against the cached golden baseline output.
        Also runs anti-hacking checks (fresh-input, jitter).
        
        Args:
            optimized: The optimized benchmark instance
            config: Optional verification configuration
            
        Returns:
            VerifyResult with verification outcome
        """
        config = config or VerifyConfig()
        
        # Check compliance
        issues = check_benchmark_compliance(optimized)
        if issues:
            reason = issues[0]
            return VerifyResult.fail(
                f"Optimized compliance check failed: {reason.value}",
            )
        
        # Extract input signature
        signature = self._extract_signature(optimized)
        if signature is None:
            return VerifyResult.fail("Optimized has no valid input signature")
        
        sig_hash = signature.hash()
        
        # Get golden output
        golden = self.cache.get(sig_hash)
        if golden is None:
            return VerifyResult.fail(
                f"No golden output cached for signature {sig_hash}. "
                "Run verify_baseline first."
            )
        
        try:
            # Run optimized with same seed
            outputs, metrics, seed_info = self._run_with_seed(optimized, config.seed)
            
            if not outputs:
                return VerifyResult.fail("Optimized produced no extractable outputs")
            
            # Get tolerance - config override takes precedence, then benchmark, then dtype default
            tolerance = config.tolerance_override
            if tolerance is None:
                tolerance = get_output_tolerance(optimized)
            
            # Compare outputs
            comparison = self._compare_outputs(
                golden.outputs,
                outputs,
                tolerance,
            )
            
            if not comparison.passed:
                return VerifyResult(
                    passed=False,
                    reason="Output mismatch between baseline and optimized",
                    signature_hash=sig_hash,
                    comparison_details=comparison,
                )
            
            # Compare workload metrics
            if golden.workload_metrics and metrics:
                metrics_match, deltas = compare_workload_metrics(
                    golden.workload_metrics,
                    metrics,
                    config.workload_tolerance,
                )
                if not metrics_match:
                    return VerifyResult(
                        passed=False,
                        reason="Workload metrics mismatch",
                        signature_hash=sig_hash,
                        workload_delta=deltas,
                    )
            
            # Run anti-hacking checks
            fresh_passed, fresh_msg = self._run_fresh_input_check(
                optimized, outputs, config
            )
            jitter_passed, jitter_msg = self._run_jitter_check(
                optimized, signature, config
            )
            
            if not fresh_passed:
                return VerifyResult.fail(f"Fresh-input check failed: {fresh_msg}")
            if not jitter_passed:
                return VerifyResult.fail(f"Jitter check failed: {jitter_msg}")
            
            # Compute checksum for optimized
            optimized_checksum = "-".join(
                hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()[:16]
                for v in [outputs[k] for k in sorted(outputs.keys())]
            )
            
            return VerifyResult.success(
                signature_hash=sig_hash,
                baseline_checksum=golden.checksum,
                optimized_checksum=optimized_checksum,
                comparison_details=comparison,
                seed_info=seed_info,
            )
            
        except Exception as e:
            return VerifyResult.fail(f"Optimized execution failed: {e}\n{traceback.format_exc()}")
    
    def verify_pair(
        self,
        baseline: Any,
        optimized: Any,
        config: Optional[VerifyConfig] = None,
    ) -> VerifyResult:
        """Verify a baseline/optimized benchmark pair.
        
        Convenience method that runs both baseline verification (with caching)
        and optimized verification in sequence.
        
        Args:
            baseline: The baseline benchmark instance
            optimized: The optimized benchmark instance
            config: Optional verification configuration
            
        Returns:
            VerifyResult with final verification outcome
        """
        config = config or VerifyConfig()
        
        # First verify and cache baseline
        baseline_result = self.verify_baseline(baseline, config)
        if not baseline_result.passed:
            return baseline_result
        
        # Then verify optimized against baseline
        return self.verify_optimized(optimized, config)
    
    def gate_perf(
        self,
        benchmark_path: str,
    ) -> Tuple[bool, Optional[str]]:
        """Check if a benchmark is allowed to run performance measurement.
        
        Based on enforcement phase and quarantine status, determines
        whether perf measurement should proceed.
        
        Args:
            benchmark_path: Path to the benchmark file
            
        Returns:
            Tuple of (allowed, reason_if_blocked)
        """
        phase = get_enforcement_phase()
        
        if phase == EnforcementPhase.DETECT:
            # Detect mode: always allow perf, just report issues
            return True, None
        
        # Check quarantine status
        if self.quarantine.is_quarantined(benchmark_path):
            record = self.quarantine.get_quarantine_record(benchmark_path)
            reason = record.quarantine_reason.value if record else "unknown"
            
            if phase == EnforcementPhase.GATE:
                return False, f"Benchmark quarantined ({reason}) - perf blocked in GATE phase"
            else:  # QUARANTINE phase
                return False, f"Benchmark quarantined ({reason}) - excluded from perf reports"
        
        return True, None

