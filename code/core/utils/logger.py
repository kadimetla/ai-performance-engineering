"""Structured logging infrastructure with Rich console and JSON file handlers.

Provides beautiful TTY output for interactive use and JSON logging for machine parsing.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional, Any, Union

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console: Any = None  # type: ignore[no-redef]
    RichHandler: Any = None  # type: ignore[no-redef]
    Progress: Any = None  # type: ignore[no-redef]
    SpinnerColumn: Any = None  # type: ignore[no-redef]
    TextColumn: Any = None  # type: ignore[no-redef]


# Global console instance
_console: Optional[Console] = None


def get_console() -> Optional[Console]:
    """Get global Rich console instance."""
    global _console
    if _console is None and RICH_AVAILABLE:
        _console = Console()
    return _console


def is_tty() -> bool:
    """Check if stdout is a TTY (interactive terminal)."""
    return sys.stdout.isatty()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = "text",  # "text" or "json"
    use_rich: Optional[bool] = None
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        log_format: Format for file logging ("text" or "json")
        use_rich: Whether to use Rich for console output (auto-detects TTY if None)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Determine if we should use Rich
    if use_rich is None:
        use_rich = RICH_AVAILABLE and is_tty()
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler: Union[RichHandler, logging.Handler]
    if use_rich and RICH_AVAILABLE:
        console_handler = RichHandler(
            console=get_console(),
            show_time=False,
            show_level=False,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)
    else:
        # Plain console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        if isinstance(log_file, str):
            log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if log_format == "json":
            # JSON file handler for machine parsing
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    log_entry = {
                        "timestamp": self.formatTime(record, self.datefmt),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                    }
                    if record.exc_info:
                        log_entry["exception"] = self.formatException(record.exc_info)
                    return json.dumps(log_entry)
            
            file_handler.setFormatter(JSONFormatter())
        else:
            # Plain text file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
        
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_benchmark_start(logger: logging.Logger, benchmark_name: str, chapter: Optional[str] = None) -> None:
    """Log benchmark start with context.
    
    Args:
        logger: Logger instance
        benchmark_name: Name of the benchmark
        chapter: Optional chapter identifier
    """
    context = f"[{chapter}] " if chapter else ""
    logger.info(f"🚀 Starting benchmark: {context}{benchmark_name}")


def log_benchmark_complete(logger: logging.Logger, benchmark_name: str, mean_ms: float, chapter: Optional[str] = None) -> None:
    """Log benchmark completion with results.
    
    Args:
        logger: Logger instance
        benchmark_name: Name of the benchmark
        mean_ms: Mean execution time in milliseconds
        chapter: Optional chapter identifier
    """
    context = f"[{chapter}] " if chapter else ""
    logger.info(f"✅ Completed: {context}{benchmark_name} - {mean_ms:.3f} ms")


def log_benchmark_error(logger: logging.Logger, benchmark_name: str, error: str, chapter: Optional[str] = None) -> None:
    """Log benchmark error.
    
    Args:
        logger: Logger instance
        benchmark_name: Name of the benchmark
        error: Error message
        chapter: Optional chapter identifier
    """
    context = f"[{chapter}] " if chapter else ""
    logger.error(f"❌ Failed: {context}{benchmark_name} - {error}")


def log_profiling_start(logger: logging.Logger, profiler: str, benchmark_name: str) -> None:
    """Log profiling start.
    
    Args:
        logger: Logger instance
        profiler: Profiler name ('nsys', 'ncu', 'torch')
        benchmark_name: Name of the benchmark
    """
    logger.debug(f"📊 Starting {profiler} profiling for {benchmark_name}")


def log_profiling_complete(logger: logging.Logger, profiler: str, benchmark_name: str, artifact_path: Optional[str] = None) -> None:
    """Log profiling completion.
    
    Args:
        logger: Logger instance
        profiler: Profiler name ('nsys', 'ncu', 'torch')
        benchmark_name: Name of the benchmark
        artifact_path: Optional path to profiling artifact
    """
    if artifact_path:
        logger.debug(f"📊 Completed {profiler} profiling for {benchmark_name}: {artifact_path}")
    else:
        logger.debug(f"📊 Completed {profiler} profiling for {benchmark_name}")


def log_profiling_skipped(logger: logging.Logger, profiler: str, reason: str) -> None:
    """Log profiling skip.
    
    Args:
        logger: Logger instance
        profiler: Profiler name ('nsys', 'ncu', 'torch')
        reason: Reason for skipping
    """
    logger.warning(f"⏭️  Skipping {profiler} profiling: {reason}")
