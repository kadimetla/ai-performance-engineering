#!/usr/bin/env python3
"""CUDA 13 green-context demo using cuda-python driver APIs."""

from __future__ import annotations

import argparse
import ctypes
import time
from typing import Tuple

import numpy as np
from cuda.bindings import driver, nvrtc


KERNEL_CODE = r"""
extern "C" __global__ void scale(float* data, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= alpha;
}
"""


def _cuda_check(err: driver.CUresult, msg: str) -> None:
    if err != driver.CUresult.CUDA_SUCCESS:
        _, err_str = driver.cuGetErrorString(err)
        raise RuntimeError(f"{msg}: {err_str.decode()}")


def _nvrtc_check(err: nvrtc.nvrtcResult, msg: str) -> None:
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err_str = nvrtc.nvrtcGetErrorString(err)
        raise RuntimeError(f"{msg}: {err_str.decode()}")


def _compile_ptx(device: driver.CUdevice) -> bytes:
    err, prog = nvrtc.nvrtcCreateProgram(KERNEL_CODE.encode(), b"green_context.cu", 0, None, None)
    _nvrtc_check(err, "nvrtcCreateProgram failed")
    err, major, minor = driver.cuDeviceComputeCapability(device)
    _cuda_check(err, "cuDeviceComputeCapability failed")
    arch = f"--gpu-architecture=sm_{major}{minor}".encode()
    err, = nvrtc.nvrtcCompileProgram(prog, 1, [arch])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        _nvrtc_check(err, f"nvrtcCompileProgram failed: {log.decode().strip()}")
    err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    _nvrtc_check(err, "nvrtcGetPTXSize failed")
    ptx = b" " * ptx_size
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
    _nvrtc_check(err, "nvrtcGetPTX failed")
    err, = nvrtc.nvrtcDestroyProgram(prog)
    _nvrtc_check(err, "nvrtcDestroyProgram failed")
    return ptx


def _run_kernel(
    *,
    ctx: driver.CUcontext,
    stream,
    ptx: bytes,
    elements: int,
    iterations: int,
    block_size: int,
    alpha: float,
) -> Tuple[float, float]:
    _cuda_check(driver.cuCtxSetCurrent(ctx)[0], "cuCtxSetCurrent failed")
    err, module = driver.cuModuleLoadData(ptx)
    _cuda_check(err, "cuModuleLoadData failed")
    err, func = driver.cuModuleGetFunction(module, b"scale")
    _cuda_check(err, "cuModuleGetFunction failed")

    host = np.random.rand(elements).astype(np.float32)
    err, dptr = driver.cuMemAlloc(host.nbytes)
    _cuda_check(err, "cuMemAlloc failed")
    err, = driver.cuMemcpyHtoD(dptr, host.ctypes.data, host.nbytes)
    _cuda_check(err, "cuMemcpyHtoD failed")

    arg0 = ctypes.c_void_p(int(dptr))
    arg1 = ctypes.c_int(elements)
    arg2 = ctypes.c_float(alpha)
    params = (ctypes.c_void_p * 3)(
        ctypes.addressof(arg0),
        ctypes.addressof(arg1),
        ctypes.addressof(arg2),
    )

    grid = (elements + block_size - 1) // block_size
    for _ in range(5):
        err, = driver.cuLaunchKernel(
            func,
            grid,
            1,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            params,
            0,
        )
        _cuda_check(err, "cuLaunchKernel warmup failed")
    if stream != 0:
        _cuda_check(driver.cuStreamSynchronize(stream)[0], "cuStreamSynchronize failed")
    else:
        _cuda_check(driver.cuCtxSynchronize()[0], "cuCtxSynchronize failed")

    start = time.perf_counter()
    for _ in range(iterations):
        err, = driver.cuLaunchKernel(
            func,
            grid,
            1,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            params,
            0,
        )
        _cuda_check(err, "cuLaunchKernel failed")
    if stream != 0:
        _cuda_check(driver.cuStreamSynchronize(stream)[0], "cuStreamSynchronize failed")
    else:
        _cuda_check(driver.cuCtxSynchronize()[0], "cuCtxSynchronize failed")
    elapsed = time.perf_counter() - start

    out = np.empty_like(host)
    err, = driver.cuMemcpyDtoH(out.ctypes.data, dptr, host.nbytes)
    _cuda_check(err, "cuMemcpyDtoH failed")

    _cuda_check(driver.cuMemFree(dptr)[0], "cuMemFree failed")
    _cuda_check(driver.cuModuleUnload(module)[0], "cuModuleUnload failed")
    return elapsed, float(out[0])


def _create_green_context(
    device: driver.CUdevice,
    sm_fraction: float,
) -> Tuple[driver.CUgreenCtx, driver.CUcontext, driver.CUstream, int, int]:
    err, sm_res = driver.cuDeviceGetDevResource(device, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
    _cuda_check(err, "cuDeviceGetDevResource failed")
    total_sms = int(sm_res.sm.smCount)
    min_partition = int(sm_res.sm.minSmPartitionSize)
    alignment = int(sm_res.sm.smCoscheduledAlignment) or min_partition
    desired = max(min_partition, int(total_sms * sm_fraction))
    desired = max(min_partition, (desired // alignment) * alignment)
    desired = min(desired, total_sms)

    err, groups, _, _ = driver.cuDevSmResourceSplitByCount(1, sm_res, 0, desired)
    _cuda_check(err, "cuDevSmResourceSplitByCount failed")
    err, desc = driver.cuDevResourceGenerateDesc(groups, len(groups))
    _cuda_check(err, "cuDevResourceGenerateDesc failed")
    flags = driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
    err, green = driver.cuGreenCtxCreate(desc, device, flags)
    _cuda_check(err, "cuGreenCtxCreate failed")
    err, ctx = driver.cuCtxFromGreenCtx(green)
    _cuda_check(err, "cuCtxFromGreenCtx failed")
    err, stream = driver.cuGreenCtxStreamCreate(
        green,
        driver.CUstream_flags.CU_STREAM_NON_BLOCKING,
        0,
    )
    _cuda_check(err, "cuGreenCtxStreamCreate failed")
    return green, ctx, stream, desired, total_sms


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA 13 green context demo.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--elements", type=int, default=1_048_576, help="Vector length.")
    parser.add_argument("--iterations", type=int, default=100, help="Timed kernel iterations.")
    parser.add_argument("--block-size", type=int, default=256, help="Kernel block size.")
    parser.add_argument("--alpha", type=float, default=1.01, help="Scale factor.")
    parser.add_argument(
        "--sm-fraction",
        type=float,
        default=0.5,
        help="Fraction of SMs to reserve for the green context.",
    )
    args = parser.parse_args()
    if args.sm_fraction <= 0.0 or args.sm_fraction > 1.0:
        raise ValueError("--sm-fraction must be within (0, 1].")

    err, = driver.cuInit(0)
    _cuda_check(err, "cuInit failed")
    err, dev = driver.cuDeviceGet(args.device)
    _cuda_check(err, "cuDeviceGet failed")
    ptx = _compile_ptx(dev)

    err, ctx = driver.cuCtxCreate(None, 0, dev)
    _cuda_check(err, "cuCtxCreate failed")
    try:
        base_time, base_sample = _run_kernel(
            ctx=ctx,
            stream=0,
            ptx=ptx,
            elements=args.elements,
            iterations=args.iterations,
            block_size=args.block_size,
            alpha=args.alpha,
        )
    finally:
        _cuda_check(driver.cuCtxDestroy(ctx)[0], "cuCtxDestroy failed")

    green_ctx = None
    green_stream = None
    try:
        green_ctx, green_ctx_handle, green_stream, green_sms, total_sms = _create_green_context(
            dev,
            args.sm_fraction,
        )
        green_time, green_sample = _run_kernel(
            ctx=green_ctx_handle,
            stream=green_stream,
            ptx=ptx,
            elements=args.elements,
            iterations=args.iterations,
            block_size=args.block_size,
            alpha=args.alpha,
        )
    finally:
        if green_ctx is not None:
            _cuda_check(driver.cuGreenCtxDestroy(green_ctx)[0], "cuGreenCtxDestroy failed")

    ratio = green_time / base_time if base_time > 0 else float("inf")
    print("Green Context Demo")
    print(f"  Device: {args.device}")
    print(f"  SM allocation: {green_sms}/{total_sms} ({args.sm_fraction:.2f} fraction)")
    print(f"  Default context time: {base_time:.4f} s (sample {base_sample:.4f})")
    print(f"  Green context time:   {green_time:.4f} s (sample {green_sample:.4f})")
    print(f"  Slowdown vs default:  {ratio:.2f}x")


if __name__ == "__main__":
    main()
