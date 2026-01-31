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


def _install_fastapi_stub() -> None:
    import asyncio
    import types
    import sys
    import urllib.parse
    from contextlib import contextmanager

    class Request:
        def __init__(self, *, method="GET", query_params=None, json_body=None):
            self.method = method
            self.query_params = query_params or {}
            self._json_body = json_body

        async def json(self):
            if self._json_body is None:
                raise ValueError("No JSON body")
            return self._json_body

        async def is_disconnected(self):
            return False

    class JSONResponse:
        def __init__(self, content, status_code=200, **_):
            self.content = content
            self.status_code = status_code

    class StreamingResponse:
        def __init__(self, content, status_code=200, **_):
            self.content = content
            self.status_code = status_code

    class CORSMiddleware:
        def __init__(self, *_args, **_kwargs):
            return None

    class _StubRoute:
        def __init__(self, path, methods, endpoint):
            self.path = path
            self.methods = methods
            self.endpoint = endpoint

    class FastAPI:
        def __init__(self, *_args, **_kwargs):
            self.routes = []

        def add_middleware(self, *_args, **_kwargs):
            return None

        def get(self, path):
            def decorator(fn):
                self.routes.append(_StubRoute(path, {"GET"}, fn))
                return fn

            return decorator

        def post(self, path):
            def decorator(fn):
                self.routes.append(_StubRoute(path, {"POST"}, fn))
                return fn

            return decorator

    class _Response:
        def __init__(self, status_code, payload=None, text_chunks=None):
            self.status_code = status_code
            self._payload = payload
            self._text_chunks = text_chunks or []

        def json(self):
            return self._payload

        def iter_text(self):
            return iter(self._text_chunks)

    class TestClient:
        def __init__(self, app):
            self._app = app

        def _coerce_param(self, value, param):
            from typing import get_args, get_origin
            if value is None:
                return value
            annotation = param.annotation if param.annotation is not param.empty else None
            target_type = None
            if annotation:
                if isinstance(annotation, str):
                    lowered = annotation.lower()
                    if "float" in lowered:
                        target_type = float
                    elif "int" in lowered:
                        target_type = int
                    elif "bool" in lowered:
                        target_type = bool
                    elif "str" in lowered:
                        target_type = str
                else:
                    origin = get_origin(annotation)
                    if origin is None:
                        target_type = annotation
                    else:
                        for arg in get_args(annotation):
                            if arg in (int, float, bool, str):
                                target_type = arg
                                break
            if target_type is None and param.default is not param.empty:
                if isinstance(param.default, bool):
                    target_type = bool
                elif isinstance(param.default, int):
                    target_type = int
                elif isinstance(param.default, float):
                    target_type = float
            try:
                if target_type is bool:
                    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
                if target_type is int:
                    return int(value)
                if target_type is float:
                    return float(value)
            except Exception:
                return value
            return value

        def _find_route(self, path, method):
            for route in getattr(self._app, "routes", []):
                if route.path == path and method in route.methods:
                    return route
            return None

        def _request(self, method, url):
            parsed = urllib.parse.urlparse(url)
            path = parsed.path
            query_params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
            route = self._find_route(path, method)
            if route is None:
                return _Response(404, payload={"error": "Not Found"})

            request = Request(method=method, query_params=query_params)
            import inspect
            signature = inspect.signature(route.endpoint)
            kwargs = {}
            for name, param in signature.parameters.items():
                if name == "request":
                    continue
                if name in query_params:
                    kwargs[name] = self._coerce_param(query_params[name], param)

            async def _call_endpoint():
                return await route.endpoint(request, **kwargs)

            result = asyncio.run(_call_endpoint())

            if isinstance(result, StreamingResponse):
                async def _consume_async_gen(gen):
                    chunks = []
                    async for chunk in gen:
                        chunks.append(str(chunk))
                    return chunks

                chunks = asyncio.run(_consume_async_gen(result.content))
                return _Response(result.status_code, text_chunks=chunks)

            if isinstance(result, JSONResponse):
                return _Response(result.status_code, payload=result.content)

            return _Response(200, payload=result)

        def get(self, url):
            return self._request("GET", url)

        @contextmanager
        def stream(self, method, url):
            response = self._request(method, url)
            yield response

    TestClient.__test__ = False

    fastapi_module = types.ModuleType("fastapi")
    fastapi_module.FastAPI = FastAPI
    fastapi_module.Request = Request
    fastapi_module.__dict__["__all__"] = ["FastAPI", "Request"]

    middleware_module = types.ModuleType("fastapi.middleware")
    cors_module = types.ModuleType("fastapi.middleware.cors")
    cors_module.CORSMiddleware = CORSMiddleware

    responses_module = types.ModuleType("fastapi.responses")
    responses_module.JSONResponse = JSONResponse
    responses_module.StreamingResponse = StreamingResponse

    testclient_module = types.ModuleType("fastapi.testclient")
    testclient_module.TestClient = TestClient

    sys.modules.setdefault("fastapi", fastapi_module)
    sys.modules.setdefault("fastapi.middleware", middleware_module)
    sys.modules.setdefault("fastapi.middleware.cors", cors_module)
    sys.modules.setdefault("fastapi.responses", responses_module)
    sys.modules.setdefault("fastapi.testclient", testclient_module)


try:
    import fastapi  # noqa: F401
except Exception:
    _install_fastapi_stub()
