#!/usr/bin/env python3
"""FastAPI backend for the dashboard."""

from __future__ import annotations

import asyncio
import json
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

import typer
try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    _FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for minimal test environments
    _FASTAPI_AVAILABLE = False

    class Request:  # type: ignore[override]
        pass

    class JSONResponse:  # type: ignore[override]
        def __init__(self, content: Any, **_: Any) -> None:
            self.content = content

    class StreamingResponse:  # type: ignore[override]
        def __init__(self, content: Any, **_: Any) -> None:
            self.content = content

    class CORSMiddleware:  # type: ignore[override]
        pass

    class _StubRoute:
        def __init__(self, path: str, methods: set[str]) -> None:
            self.path = path
            self.methods = methods

    class FastAPI:  # type: ignore[override]
        def __init__(self, *_: Any, **__: Any) -> None:
            self.routes: list[_StubRoute] = []

        def add_middleware(self, *_: Any, **__: Any) -> None:
            return None

        def get(self, path: str):
            def decorator(fn):
                self.routes.append(_StubRoute(path, {"GET"}))
                return fn

            return decorator

        def post(self, path: str):
            def decorator(fn):
                self.routes.append(_StubRoute(path, {"POST"}))
                return fn

            return decorator

from core.api.registry import ApiRoute, get_routes
from core.api.response import build_response


fastapi_app = FastAPI(title="AISP Dashboard API", version="1.0")
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _configure_engine(data_file: Path | None) -> None:
    if data_file is None:
        return
    path = Path(data_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dashboard data file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Dashboard data file must be a file: {path}")
    path = path.resolve()

    from core.analysis.performance_analyzer import PerformanceAnalyzer, load_benchmark_data
    import core.engine as engine
    from core.perf_core import get_core

    handler = get_core(data_file=path, refresh=True)
    engine._handler_instance = handler
    engine._analyzer_instance = PerformanceAnalyzer(lambda: load_benchmark_data(path, handler.bench_roots))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _collect_params(request: Request, body: Dict[str, Any] | None) -> Dict[str, Any]:
    params = dict(request.query_params)
    if body:
        params.update(body)
    return params


def _make_endpoint(route: ApiRoute):
    async def _endpoint(request: Request) -> JSONResponse:
        started = time.time()
        body: Dict[str, Any] | None = None
        if request.method in {"POST", "PUT"}:
            try:
                payload = await request.json()
                if isinstance(payload, dict):
                    body = payload
                else:
                    body = {"body": payload}
            except Exception:
                body = None
        params = _collect_params(request, body)
        include_context = _parse_bool(params.get("include_context"))
        context_level = str(params.get("context_level", "summary"))

        had_exception = False
        try:
            result = route.handler(params)
        except Exception as exc:
            had_exception = True
            result = {"error": str(exc)}
        duration_ms = int((time.time() - started) * 1000)
        payload = build_response(
            route.name,
            params,
            result,
            duration_ms,
            had_exception=had_exception,
            include_context=include_context,
            context_level=context_level,
        )
        return JSONResponse(payload)

    return _endpoint


def _register_routes() -> None:
    for route in get_routes():
        if route.method.upper() == "GET":
            fastapi_app.get(route.path)(_make_endpoint(route))
        elif route.method.upper() == "POST":
            fastapi_app.post(route.path)(_make_endpoint(route))
        else:
            raise RuntimeError(f"Unsupported HTTP method in API registry: {route.method}")


_register_routes()


@fastapi_app.get("/api/gpu/stream")
async def gpu_stream(
    request: Request,
    interval: float = 5.0,
    max_events: Optional[int] = None,
) -> StreamingResponse:
    if interval <= 0:
        raise ValueError("interval must be > 0")
    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be > 0")

    async def _event_stream():
        from core.engine import get_engine

        count = 0
        while True:
            if await request.is_disconnected():
                break
            try:
                payload = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "gpu": get_engine().gpu.info(),
                }
                yield f"event: gpu\ndata: {json.dumps(payload)}\n\n"
            except Exception as exc:
                error_payload = {"error": str(exc)}
                yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"

            count += 1
            if max_events is not None and count >= max_events:
                break
            await asyncio.sleep(interval)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(_event_stream(), media_type="text/event-stream", headers=headers)


cli = typer.Typer(help="AISP dashboard API server", no_args_is_help=True)


@cli.callback()
def cli_main() -> None:
    """AISP dashboard API server."""
    return None


def serve_dashboard(
    port: int = 6970,
    data_file: Path | None = None,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    log_level: str = "info",
) -> None:
    """Start the dashboard API server."""
    _configure_engine(data_file)
    if open_browser:
        browser_host = host
        if host in {"0.0.0.0", "::"}:
            browser_host = "127.0.0.1"
        url = f"http://{browser_host}:{port}"
        if not webbrowser.open(url):
            raise RuntimeError(f"Failed to open browser at {url}")
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run the dashboard API server") from exc
    uvicorn.run("dashboard.api.server:fastapi_app", host=host, port=port, log_level=log_level)


@cli.command("serve")
def cli_serve(
    port: int = typer.Option(6970, "--port", "-p", help="Port to run the server on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    log_level: str = typer.Option("info", "--log-level", help="Uvicorn log level"),
    data_file: Optional[Path] = typer.Option(None, "--data", "-d", help="Path to benchmark_test_results.json"),
    open_browser: bool = typer.Option(False, "--open-browser", help="Open browser to the backend URL"),
) -> None:
    """Start the dashboard API server."""
    serve_dashboard(
        port=port,
        host=host,
        log_level=log_level,
        data_file=data_file,
        open_browser=open_browser,
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
