"""Global pytest configuration.

We explicitly disable auto-loading of external pytest plugins to prevent
environment-provided plugins from interfering with test discovery and capture
in this repository's harness.
"""

import os
import logging
import warnings
import signal
import pytest

# Guard against site-wide plugins that can change stdout handling (causing
# Illegal seek/OSError on teardown in CI and local shells).
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
# Disable parallel tokenizer workers to avoid fork-time warnings from tokenizers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Keep linear warnings visible by default; callers can still override PYTHONWARNINGS.
os.environ.setdefault("PYTHONWARNINGS", "default")
# Torch debug logging is enabled in many environments via TORCH_LOGS.
# When those atexit hooks fire after pytest has closed capture streams,
# the logging writes can raise ValueError on a closed file. Disable noisy
# torch logging for test runs up front.
os.environ.pop("TORCH_LOGS", None)
try:  # PyTorch 2.1+
    import torch._logging as torch_logging  # type: ignore
    torch_logging.set_logs(dynamo=False, fx=False, inductor=False, aot=False)
except Exception:
    pass


# Torch occasionally logs at process teardown via atexit hooks (e.g., Dynamo
# compile stats). If pytest has already closed its capture streams, those
# log writes can raise ValueError("I/O operation on closed file"). Install
# inert handlers on the noisy loggers so teardown logging cannot fail.
for _logger_name in [
    "torch._dynamo",
    "torch._dynamo.utils",
    "torch.fx.experimental.symbolic_shapes",
]:
    _logger = logging.getLogger(_logger_name)
    _logger.handlers.clear()
    _logger.addHandler(logging.NullHandler())
    _logger.propagate = False

# Noise filters for known benign warnings during tests.
warnings.filterwarnings(
    "ignore",
    message="Attempting to run cuBLAS, but there was no current CUDA context",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Attempting to run cuBLAS.*",
    category=UserWarning,
    module="torch.nn.modules.linear",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"labs\\.nanochat_fullstack\\.nanochat\\.kernels\\.stubs",
)
warnings.filterwarnings(
    "ignore",
    message=".*cuBLAS.*primary context.*",
    category=UserWarning,
)

# -----------------------------------------------------------------------------
# Simple built-in timeout support (pytest-timeout is disabled by plugin block)
# -----------------------------------------------------------------------------
def _parse_timeout(config) -> float:
    try:
        return float(config.getini("timeout"))
    except Exception:
        return 0.0


def pytest_configure(config):
    # Accept the ini option from pytest.ini even when pytest-timeout isn't loaded.
    config._global_timeout = _parse_timeout(config)


def pytest_addoption(parser):
    parser.addini("timeout", "Global timeout (seconds)", default="0")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    timeout = getattr(item.config, "_global_timeout", 0)
    if not timeout or timeout <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Test exceeded global timeout of {timeout} seconds")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
