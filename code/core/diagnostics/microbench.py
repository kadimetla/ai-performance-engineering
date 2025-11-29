"""Lightweight microbenchmarks for CLI/dashboard exposure."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional
import typer


def _now() -> float:
    return time.perf_counter()


def disk_io_test(file_size_mb: int = 256, block_size_kb: int = 1024, tmp_dir: str | None = None) -> Dict[str, Any]:
    """Simple sequential disk write/read benchmark.

    Args:
        file_size_mb: Size of file to write/read.
        block_size_kb: Block size used for write/read.
        tmp_dir: Optional directory for the test file.
    """
    tmp_path = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
    tmp_path.mkdir(parents=True, exist_ok=True)
    file_path = tmp_path / "microbench_io.bin"

    total_bytes = file_size_mb * 1024 * 1024
    block_bytes = block_size_kb * 1024
    data = os.urandom(block_bytes)

    # Write
    start = _now()
    with open(file_path, "wb") as f:
        written = 0
        while written < total_bytes:
            f.write(data)
            written += len(data)
    write_time = _now() - start

    # Read
    start = _now()
    with open(file_path, "rb") as f:
        while f.read(block_bytes):
            pass
    read_time = _now() - start

    try:
        file_path.unlink()
    except Exception:
        pass

    return {
        "file_size_mb": file_size_mb,
        "block_size_kb": block_size_kb,
        "write_seconds": write_time,
        "write_gbps": (total_bytes / write_time) / 1e9 if write_time > 0 else None,
        "read_seconds": read_time,
        "read_gbps": (total_bytes / read_time) / 1e9 if read_time > 0 else None,
        "path": str(tmp_path),
    }


def pcie_bandwidth_test(size_mb: int = 256, iters: int = 10) -> Dict[str, Any]:
    """Measure H2D and D2H bandwidth using torch CUDA if available."""
    try:
        import torch
    except ImportError as e:
        return {"error": f"torch not available: {e}"}

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    size_bytes = size_mb * 1024 * 1024
    tensor_cpu = torch.empty(size_bytes // 4, dtype=torch.float32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device=device)

    torch.cuda.synchronize()
    # H2D
    start = _now()
    for _ in range(iters):
        tensor_gpu.copy_(tensor_cpu, non_blocking=True)
    torch.cuda.synchronize()
    h2d_time = (_now() - start) / iters

    # D2H
    start = _now()
    for _ in range(iters):
        tensor_cpu.copy_(tensor_gpu, non_blocking=True)
    torch.cuda.synchronize()
    d2h_time = (_now() - start) / iters

    return {
        "size_mb": size_mb,
        "iters": iters,
        "h2d_gbps": (size_bytes / h2d_time) / 1e9 if h2d_time > 0 else None,
        "d2h_gbps": (size_bytes / d2h_time) / 1e9 if d2h_time > 0 else None,
    }


def mem_hierarchy_test(size_mb: int = 256, stride: int = 128) -> Dict[str, Any]:
    """Crude stride-based bandwidth test on GPU memory."""
    try:
        import torch
    except ImportError as e:
        return {"error": f"torch not available: {e}"}
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    n = (size_mb * 1024 * 1024) // 4
    x = torch.arange(n, device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    start = _now()
    # stride access
    y = x[::stride].clone()
    torch.cuda.synchronize()
    elapsed = _now() - start
    bytes_moved = y.numel() * 4
    return {
        "size_mb": size_mb,
        "stride": stride,
        "bandwidth_gbps": (bytes_moved / elapsed) / 1e9 if elapsed > 0 else None,
        "elements": y.numel(),
    }


def tensor_core_bench(size: int = 4096, precision: str = "fp16") -> Dict[str, Any]:
    """Matmul throughput benchmark to stress tensor cores."""
    try:
        import torch
    except ImportError as e:
        return {"error": f"torch not available: {e}"}
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    precision_lower = precision.lower()
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "tf32": torch.float32,
        "fp32": torch.float32,
    }
    placeholder_used = False

    if precision_lower == "fp8":
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["fp8"] = getattr(torch, "float8_e4m3fn")
        else:
            dtype_map["fp8"] = torch.float16
            placeholder_used = True
    elif precision_lower == "int8":
        return {"error": "INT8 matmul not supported in this minimal microbench; use fp16/bf16/tf32"}

    dtype = dtype_map.get(precision_lower)
    if dtype is None:
        return {"error": f"unsupported precision: {precision}"}

    device = torch.device("cuda")
    a = torch.randn((size, size), device=device, dtype=dtype)
    b = torch.randn((size, size), device=device, dtype=dtype)
    torch.cuda.synchronize()
    start = _now()
    c = a @ b
    torch.cuda.synchronize()
    elapsed = _now() - start
    flops = 2 * (size ** 3)
    tflops = (flops / elapsed) / 1e12 if elapsed > 0 else None
    return {
        "size": size,
        "precision": precision,
        "tflops": tflops,
        "elapsed_seconds": elapsed,
        "output_shape": list(c.shape),
        "placeholder_used": placeholder_used,
    }


def sfu_bench(size: int = 64 * 1024 * 1024) -> Dict[str, Any]:
    """SFU-heavy benchmark using sin/cos operations."""
    try:
        import torch
    except ImportError as e:
        return {"error": f"torch not available: {e}"}
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    x = torch.linspace(0, 10, steps=size, device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    start = _now()
    y = torch.sin(x) + torch.cos(x)
    torch.cuda.synchronize()
    elapsed = _now() - start
    ops = size * 4  # approx operations per element
    gops = (ops / elapsed) / 1e9 if elapsed > 0 else None
    return {
        "elements": size,
        "elapsed_seconds": elapsed,
        "gops": gops,
        "result_sample": float(y[0].item()) if y.numel() > 0 else None,
    }


def network_loopback_test(size_mb: int = 64, port: int = 50007) -> Dict[str, Any]:
    """Simple loopback TCP throughput test (localhost)."""
    total_bytes = size_mb * 1024 * 1024
    payload = b"x" * 65536

    def server():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        conn, _ = srv.accept()
        received = 0
        while received < total_bytes:
            data = conn.recv(len(payload))
            if not data:
                break
            received += len(data)
        conn.close()
        srv.close()

    import threading
    t = threading.Thread(target=server, daemon=True)
    t.start()
    time.sleep(0.1)

    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    start = _now()
    cli.connect(("127.0.0.1", port))
    sent = 0
    while sent < total_bytes:
        cli.sendall(payload)
        sent += len(payload)
    cli.shutdown(socket.SHUT_WR)
    cli.close()
    t.join()
    elapsed = _now() - start
    return {
        "size_mb": size_mb,
        "elapsed_seconds": elapsed,
        "throughput_gbps": (total_bytes / elapsed) / 1e9 if elapsed > 0 else None,
        "notes": "Loopback TCP only; use iperf for real NIC tests",
    }


def _print(res: Dict[str, Any], json_out: bool) -> None:
    if json_out:
        typer.echo(json.dumps(res, indent=2, default=str))
    else:
        typer.echo(res)

# CLI/aisp wrappers (SimpleNamespace args)
def disk(args: Any) -> int:
    res = disk_io_test(
        file_size_mb=getattr(args, "file_size_mb", getattr(args, "size_mb", 256)),
        block_size_kb=getattr(args, "block_size_kb", getattr(args, "block_kb", 1024)),
        tmp_dir=getattr(args, "tmp_dir", None),
    )
    _print(res, getattr(args, "json", False))
    return 0


def pcie(args: Any) -> int:
    res = pcie_bandwidth_test(
        size_mb=getattr(args, "size_mb", 256),
        iters=getattr(args, "iters", 10),
    )
    _print(res, getattr(args, "json", False))
    return 0


def mem_hierarchy(args: Any) -> int:
    res = mem_hierarchy_test(
        size_mb=getattr(args, "size_mb", 256),
        stride=getattr(args, "stride", 128),
    )
    _print(res, getattr(args, "json", False))
    return 0


def tensor_core(args: Any) -> int:
    res = tensor_core_bench(
        size=getattr(args, "size", 4096),
        precision=getattr(args, "precision", "fp16"),
    )
    _print(res, getattr(args, "json", False))
    return 0


def sfu(args: Any) -> int:
    res = sfu_bench(size=getattr(args, "elements", 64 * 1024 * 1024))
    _print(res, getattr(args, "json", False))
    return 0


def loopback(args: Any) -> int:
    res = network_loopback_test(
        size_mb=getattr(args, "size_mb", 256),
        port=getattr(args, "port", 5789),
    )
    _print(res, getattr(args, "json", False))
    return 0


app = typer.Typer(help="Lightweight microbenchmarks (disk, PCIe, memory, tensor core, SFU, loopback).")


@app.command("disk")
def cli_disk(
    file_size_mb: int = typer.Option(256, "--file-size-mb", "-s", help="Size of the temp file to write/read (MB)"),
    block_size_kb: int = typer.Option(1024, "--block-size-kb", "-b", help="Block size for IO (KB)"),
    tmp_dir: Optional[str] = typer.Option(None, "--tmp-dir", "-t", help="Optional directory for the temp file"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    _print(disk_io_test(file_size_mb=file_size_mb, block_size_kb=block_size_kb, tmp_dir=tmp_dir), json_out)


@app.command("pcie")
def cli_pcie(
    size_mb: int = typer.Option(256, "--size-mb", "-s", help="Transfer size (MB)"),
    iters: int = typer.Option(10, "--iters", "-i", help="Number of repetitions"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    _print(pcie_bandwidth_test(size_mb=size_mb, iters=iters), json_out)


@app.command("mem")
def cli_mem(
    size_mb: int = typer.Option(256, "--size-mb", "-s", help="Tensor size (MB)"),
    stride: int = typer.Option(128, "--stride", help="Stride for access pattern"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    _print(mem_hierarchy_test(size_mb=size_mb, stride=stride), json_out)


@app.command("tensor")
def cli_tensor(
    size: int = typer.Option(4096, "--size", "-s", help="Matrix size (N x N)"),
    precision: str = typer.Option("fp16", "--precision", "-p", help="Precision: fp16/bf16/tf32/fp32/fp8"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    _print(tensor_core_bench(size=size, precision=precision), json_out)


@app.command("sfu")
def cli_sfu(
    elements: int = typer.Option(64 * 1024 * 1024, "--elements", "-n", help="Number of elements"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    _print(sfu_bench(size=elements), json_out)


@app.command("loopback")
def cli_loopback(
    size_mb: int = typer.Option(64, "--size-mb", "-s", help="Transfer size in MB"),
    port: int = typer.Option(50007, "--port", "-p", help="Port to use"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    _print(network_loopback_test(size_mb=size_mb, port=port), json_out)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
