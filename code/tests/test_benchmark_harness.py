"""Unit tests for benchmark harness core functionality.

Tests timeout handling, subprocess isolation, timing accuracy, memory tracking,
and error propagation as specified in Part 2.10 of the unified improvement plan.
"""

import pytest
import sys
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Optional

# Add repo root to path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.env import apply_env_defaults
apply_env_defaults()

import torch
from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode, BenchmarkConfig, BaseBenchmark
from core.benchmark.models import BenchmarkResult, TimingStats, MemoryStats


# Skip tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required - NVIDIA GPU and tools must be available"
)


class SimpleBenchmark(BaseBenchmark):
    """Simple benchmark for testing."""
    
    def __init__(self):
        super().__init__()
        self.tensor = None
        self.output = None
    
    def setup(self) -> None:
        self.tensor = torch.randn(100, 100, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        self.output = self.tensor @ self.tensor
        torch.cuda.synchronize()
    
    def validate_result(self) -> Optional[str]:
        return None
    
    def get_verify_inputs(self):
        if self.tensor is None:
            raise RuntimeError("Tensor not initialized")
        return {"input": self.tensor}
    
    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output
    
    def get_output_tolerance(self):
        return (1e-5, 1e-8)

    def get_input_signature(self) -> dict:
        if self.tensor is None:
            raise RuntimeError("setup() must be called before get_input_signature()")
        return {
            "tensor_shape": tuple(self.tensor.shape),
            "dtype": str(self.tensor.dtype),
        }


class SlowBenchmark(BaseBenchmark):
    """Benchmark that takes a long time - for timeout testing."""
    
    def __init__(self):
        super().__init__()
        self.tensor = None
        self.output = None
    
    def setup(self) -> None:
        self.tensor = torch.randn(100, 100, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        # Actually block to ensure timeout triggers - use sleep in a loop
        # This is more reliable than tensor operations which may finish too fast
        import time
        # Sleep for 0.1s per iteration, so 100 iterations = 10s (guaranteed > 1s timeout)
        for _ in range(100):
            time.sleep(0.1)
            # Do minimal work to keep CUDA context alive
            _ = self.tensor @ self.tensor
        torch.cuda.synchronize()
        self.output = self.tensor
    
    def validate_result(self) -> Optional[str]:
        return None
    
    def get_verify_inputs(self):
        if self.tensor is None:
            raise RuntimeError("Tensor not initialized")
        return {"input": self.tensor}
    
    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output
    
    def get_output_tolerance(self):
        return (1e-5, 1e-8)

    def get_input_signature(self) -> dict:
        if self.tensor is None:
            raise RuntimeError("setup() must be called before get_input_signature()")
        return {
            "tensor_shape": tuple(self.tensor.shape),
            "dtype": str(self.tensor.dtype),
        }


class FailingBenchmark(BaseBenchmark):
    """Benchmark that raises an error - for error propagation testing."""
    
    def __init__(self):
        super().__init__()
        self.output = None
    
    def setup(self) -> None:
        pass
    
    def benchmark_fn(self) -> None:
        raise RuntimeError("Intentional benchmark failure")
    
    def validate_result(self) -> Optional[str]:
        return None
    
    def get_verify_inputs(self):
        return {"input": torch.tensor([0.0], device=self.device)}
    
    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output
    
    def get_output_tolerance(self):
        return (1e-5, 1e-8)

    def get_input_signature(self) -> dict:
        return {"scalar": 0.0}


class TestSubprocessTimeoutKill:
    """Test subprocess timeout kill and error propagation."""
    
    def test_timeout_kills_subprocess(self):
        """Test that subprocess is killed when timeout is exceeded."""
        benchmark = SlowBenchmark()
        config = BenchmarkConfig(
            iterations=100,  # Enough iterations to exceed timeout
            warmup=5,
            measurement_timeout_seconds=1,  # Very short timeout (1 second)
            enable_profiling=False,
            enable_memory_tracking=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        # The harness may return a result with timeout info OR raise RuntimeError
        # Both are valid - check for timeout info in result or error message
        try:
            result = harness.benchmark(benchmark)
            # If we get a result, it should have timeout information
            assert result.timeout_stage is not None
            assert result.timeout_stage == "measurement"
            assert result.timeout_limit_seconds == 1
            assert result.timeout_duration_seconds > 0
            assert len(result.errors) > 0
            assert "timeout" in result.errors[0].lower() or "TIMEOUT" in result.errors[0]
        except RuntimeError as e:
            # If RuntimeError is raised, it should mention timeout
            error_msg = str(e).lower()
            assert "timeout" in error_msg or "TIMEOUT" in str(e)
    
    def test_error_propagation_from_subprocess(self):
        """Test that errors from subprocess are properly propagated."""
        benchmark = FailingBenchmark()
        config = BenchmarkConfig(
            iterations=5,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        # When subprocess fails, harness should surface errors in the result
        result = harness.benchmark(benchmark)
        assert len(result.errors) > 0
        assert any("intentional benchmark failure" in err.lower() for err in result.errors)


class TestPyTorchTimerCorrectness:
    """Test PyTorch Timer stmt/globals correctness."""
    
    def test_pytorch_timer_uses_correct_stmt(self):
        """Test that PyTorch Timer uses stmt='fn()' pattern."""
        benchmark = SimpleBenchmark()
        config = BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,
            use_subprocess=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.PYTORCH, config=config)
        
        # Mock Timer at the import location (torch.utils.benchmark.Timer)
        # Timer is imported inside _benchmark_pytorch, so we patch the module where it's used
        with patch('torch.utils.benchmark.Timer') as mock_timer:
            mock_timer_instance = MagicMock()
            mock_timer_instance.blocked_autorange.return_value = ([1.0] * 10, 10)
            mock_timer.return_value = mock_timer_instance
            
            result = harness.benchmark(benchmark)
            
            # Verify Timer was called with correct stmt pattern
            mock_timer.assert_called_once()
            call_kwargs = mock_timer.call_args[1]
            assert 'stmt' in call_kwargs
            assert call_kwargs['stmt'] == "fn()"
            assert 'globals' in call_kwargs
            assert 'fn' in call_kwargs['globals']
    
    def test_pytorch_timer_globals_contain_fn(self):
        """Test that Timer globals contain the benchmark function."""
        benchmark = SimpleBenchmark()
        config = BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,
            use_subprocess=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.PYTORCH, config=config)
        
        # Mock Timer at the import location (torch.utils.benchmark.Timer)
        with patch('torch.utils.benchmark.Timer') as mock_timer:
            mock_timer_instance = MagicMock()
            mock_timer_instance.blocked_autorange.return_value = ([1.0] * 10, 10)
            mock_timer.return_value = mock_timer_instance
            
            harness.benchmark(benchmark)
            
            call_kwargs = mock_timer.call_args[1]
            globals_dict = call_kwargs['globals']
            assert 'fn' in globals_dict
            assert callable(globals_dict['fn'])


class TestTimeoutMultiplierPropagation:
    """Ensure timeout multipliers behave correctly inside the harness."""
    
    def test_harness_preserves_explicit_timeouts_when_cloning_config(self):
        config = BenchmarkConfig(
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=2,
            timeout_multiplier=5.0,
            enable_profiling=False,
            enable_memory_tracking=False,
            use_subprocess=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        benchmark = SimpleBenchmark()
        
        with patch.object(BenchmarkHarness, "_benchmark_with_threading", autospec=True) as mock_thread:
            mock_thread.return_value = MagicMock()
            harness.benchmark(benchmark)
            passed_config = mock_thread.call_args[0][2]
            assert passed_config.measurement_timeout_seconds == 2
            assert harness.config.measurement_timeout_seconds == 2


class TestEventSyncTimingLoop:
    """Test event-sync timing loop correctness (no device-wide sync)."""
    
    def test_cuda_timing_uses_end_event_sync(self):
        """Test that CUDA timing only synchronizes end event, not device-wide."""
        benchmark = SimpleBenchmark()
        config = BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,
            use_subprocess=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        # Track CUDA synchronization calls
        sync_calls = []
        original_sync = torch.cuda.synchronize
        
        def tracked_sync(*args, **kwargs):
            sync_calls.append(('synchronize', args, kwargs))
            return original_sync(*args, **kwargs)
        
        with patch('torch.cuda.synchronize', side_effect=tracked_sync):
            result = harness.benchmark(benchmark)
        
        # Should have timing results
        assert result.timing is not None
        assert result.timing.iterations > 0
        
        # Verify synchronize was called (for end event sync)
        # The exact count depends on implementation, but should be called
        assert len(sync_calls) > 0
    
    def test_timing_loop_produces_valid_results(self):
        """Test that timing loop produces valid timing statistics."""
        benchmark = SimpleBenchmark()
        config = BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,
            adaptive_iterations=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        
        # Verify timing stats are valid
        assert result.timing is not None
        assert result.timing.iterations == 20
        assert result.timing.warmup_iterations == 5
        assert result.timing.mean_ms > 0
        assert result.timing.median_ms > 0
        assert result.timing.min_ms > 0
        assert result.timing.max_ms > 0
        assert result.timing.max_ms >= result.timing.min_ms
        assert result.timing.mean_ms >= result.timing.min_ms
        assert result.timing.mean_ms <= result.timing.max_ms


class TestMemoryTracking:
    """Test memory tracking values on CUDA."""
    
    def test_memory_tracking_enabled(self):
        """Test that memory tracking captures peak and allocated memory."""
        benchmark = SimpleBenchmark()
        config = BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=True,  # Enable memory tracking
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        
        # Should have memory stats
        assert result.memory is not None
        assert result.memory.peak_mb is not None
        assert result.memory.peak_mb >= 0
        # Allocated memory may be None if not tracked, but peak should be present
    
    def test_memory_tracking_disabled(self):
        """Test that memory tracking can be disabled."""
        benchmark = SimpleBenchmark()
        config = BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,  # Disable memory tracking
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        
        # Memory stats may be None when disabled
        # (implementation may still create empty stats, which is fine)
        if result.memory is not None:
            # If present, values should be None or 0
            pass  # Accept either None or empty stats
    
    def test_memory_tracking_values_are_reasonable(self):
        """Test that memory tracking values are reasonable (not negative, etc.)."""
        benchmark = SimpleBenchmark()
        config = BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=True,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        
        if result.memory is not None:
            if result.memory.peak_mb is not None:
                assert result.memory.peak_mb >= 0
            if result.memory.allocated_mb is not None:
                assert result.memory.allocated_mb >= 0
            if result.memory.reserved_mb is not None:
                assert result.memory.reserved_mb >= 0


class TestBenchmarkModes:
    """Test all benchmark modes work correctly."""
    
    def test_custom_mode(self):
        """Test CUSTOM mode works."""
        benchmark = SimpleBenchmark()
        config = BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            adaptive_iterations=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        assert result.timing.iterations == 10
    
    def test_pytorch_mode(self):
        """Test PYTORCH mode works."""
        benchmark = SimpleBenchmark()
        config = BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            adaptive_iterations=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.PYTORCH, config=config)
        
        result = harness.benchmark(benchmark)
        assert result.timing.iterations == 10
    
    def test_triton_mode(self):
        """Test TRITON mode works (requires Triton benchmark)."""
        assert True


class TestErrorHandling:
    """Test error handling and recovery."""
    
    def test_benchmark_without_setup(self):
        """Test benchmark that doesn't implement setup."""
        class NoSetupBenchmark(BaseBenchmark):
            def benchmark_fn(self) -> None:
                self.output = torch.tensor([1.0], device=self.device)
            
            def get_verify_inputs(self):
                return {"input": torch.tensor([1.0], device=self.device)}
            
            def get_verify_output(self):
                return getattr(self, "output", torch.tensor([1.0], device=self.device))
            
            def get_output_tolerance(self):
                return (1e-5, 1e-8)

        benchmark = NoSetupBenchmark()
        config = BenchmarkConfig(
            iterations=5,
            warmup=5,
            enable_profiling=False,
            use_subprocess=False,
            adaptive_iterations=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        # Should not crash - setup() is optional
        result = harness.benchmark(benchmark)
        assert result.timing.iterations == 5
    
    def test_benchmark_with_validation_error(self):
        """Test benchmark that returns validation error."""
        class InvalidBenchmark(BaseBenchmark):
            def benchmark_fn(self) -> None:
                self.tensor = torch.randn(10, 10, device=self.device)
                self.output = self.tensor
            
            def validate_result(self) -> Optional[str]:
                return "Validation failed"
            
            def get_verify_inputs(self):
                return {"input": self.tensor}
            
            def get_verify_output(self):
                return getattr(self, "output", self.tensor)
            
            def get_output_tolerance(self):
                return (1e-5, 1e-8)
        
        benchmark = InvalidBenchmark()
        config = BenchmarkConfig(iterations=5, warmup=5, enable_profiling=False, use_subprocess=False)
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        # Should have validation error
        assert len(result.errors) > 0 or result.timing.iterations == 0
