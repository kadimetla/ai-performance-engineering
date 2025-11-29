"""Custom exception hierarchy for benchmark execution.

Provides specific exception types for different failure modes, enabling
better error handling and recovery strategies.
"""

from __future__ import annotations

from typing import Any


class BenchmarkError(Exception):
    """Base exception for all benchmark-related errors."""
    pass


class BenchmarkTimeoutError(BenchmarkError):
    """Raised when a benchmark exceeds its timeout limit.
    
    Attributes:
        stage: Stage that timed out ('setup', 'warmup', 'measurement', 'profiling')
        timeout_seconds: Configured timeout limit
        elapsed_seconds: Time elapsed before timeout
    """
    
    def __init__(
        self,
        message: str,
        stage: str,
        timeout_seconds: float,
        elapsed_seconds: float,
    ):
        super().__init__(message)
        self.stage = stage
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds


class BenchmarkExecutionError(BenchmarkError):
    """Raised when benchmark execution fails.
    
    Attributes:
        benchmark_name: Name of the benchmark that failed
        stage: Stage where failure occurred
        original_error: The original exception that caused the failure
    """
    
    def __init__(
        self,
        message: str,
        benchmark_name: str,
        stage: str,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.benchmark_name = benchmark_name
        self.stage = stage
        self.original_error = original_error


class BenchmarkValidationError(BenchmarkError):
    """Raised when benchmark validation fails.
    
    Attributes:
        benchmark_name: Name of the benchmark that failed validation
        validation_message: Message from validate_result()
    """
    
    def __init__(
        self,
        message: str,
        benchmark_name: str,
        validation_message: str,
    ):
        super().__init__(message)
        self.benchmark_name = benchmark_name
        self.validation_message = validation_message


class BenchmarkDiscoveryError(BenchmarkError):
    """Raised when benchmark discovery fails.
    
    Attributes:
        path: Path that failed to load
        reason: Reason for failure
    """
    
    def __init__(
        self,
        message: str,
        path: str,
        reason: str,
    ):
        super().__init__(message)
        self.path = path
        self.reason = reason


class ProfilingError(BenchmarkError):
    """Raised when profiling fails.
    
    Attributes:
        profiler: Name of profiler that failed ('nsys', 'ncu', 'torch')
        reason: Reason for failure
    """
    
    def __init__(
        self,
        message: str,
        profiler: str,
        reason: str,
    ):
        super().__init__(message)
        self.profiler = profiler
        self.reason = reason


class ConfigurationError(BenchmarkError):
    """Raised when benchmark configuration is invalid.
    
    Attributes:
        config_key: Configuration key that is invalid
        config_value: Invalid value
        reason: Reason for invalidity
    """
    
    def __init__(
        self,
        message: str,
        config_key: str,
        config_value: Any,
        reason: str,
    ):
        super().__init__(message)
        self.config_key = config_key
        self.config_value = config_value
        self.reason = reason

