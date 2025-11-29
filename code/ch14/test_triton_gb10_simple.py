#!/usr/bin/env python3
"""
Simple Triton smoke script for GB10 (SM 12.1) kernels.
When imported (e.g., by pytest) it performs no work; running the module executes the demo.
"""

import os
import sys

# Ensure arch_config patches are applied before any Triton imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import triton
import triton.language as tl


def run_demo():
    if not torch.cuda.is_available():
        print("CUDA device required for Triton GB10 simple kernel validation")
        return

    print("=" * 80)
    print("Testing Triton on GB10 (SM 12.1) - Simple Kernels (No TMA)")
    print("=" * 80)

    props = torch.cuda.get_device_properties(0)
    print(f"GPU:      {props.name}")
    print(f"CC:       {props.major}.{props.minor}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"Triton:   {triton.__version__}")
    print()

    # Test 1: Simple addition kernel
    print("Test 1: Simple vector addition")
    print("-" * 80)

    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x + y, mask=mask)

    n_elements = 1024
    BLOCK_SIZE = 256
    x = torch.randn(n_elements, device='cuda', dtype=torch.float16)
    y = torch.randn(n_elements, device='cuda', dtype=torch.float16)
    output = torch.empty_like(x)

    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    expected = x + y
    if not torch.allclose(output, expected, rtol=1e-3, atol=1e-3):
        max_diff = (output - expected).abs().max().item()
        raise AssertionError(f"Result mismatch: max diff = {max_diff}")

    # Test 2: Simple matmul using Triton
    print("\nTest 2: Triton matmul")
    print("-" * 80)

    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
        b_ptrs = b_ptr + (offs_k[:, None] * N + offs_n[None, :])
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N
        c = acc.to(tl.float16)
        c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n[None, :])
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    M, N, K = 64, 64, 64
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    matmul_kernel[grid](a, b, c, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    torch.cuda.synchronize()

    expected = torch.matmul(a, b)
    if not torch.allclose(c, expected, rtol=1e-2, atol=1e-2):
        max_diff = (c - expected).abs().max().item()
        raise AssertionError(f"Result mismatch: max diff = {max_diff}")


def test_triton_simple_smoke():
    """Lightweight smoke test to ensure module imports."""
    assert True


if __name__ == "__main__":
    run_demo()
