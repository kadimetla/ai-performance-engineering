import torch
import triton
import triton.language as tl
import triton.testing

# Check for FP8 support
try:
    FP8_E4M3_DTYPE = torch.float8_e4m3fn
    FP8_E5M2_DTYPE = torch.float8_e5m2
    FP8_AVAILABLE = True
except AttributeError:
    FP8_AVAILABLE = False
    FP8_E4M3_DTYPE = torch.float16
    FP8_E5M2_DTYPE = torch.float16


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def tiled_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Tiled GEMM with tensor descriptors and autotuning."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    B_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_tiles = (K + BLOCK_K - 1) // BLOCK_K
    if K_tiles == 0:
        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)
        return

    k0 = 0
    if (m0 + BLOCK_M <= M) and (k0 + BLOCK_K <= K):
        a_cur = A_desc.load([m0, k0])
    else:
        col_ids = k0 + offs_k
        row_offsets = offs_m[:, None] + tl.zeros((BLOCK_M, BLOCK_K), dtype=offs_m.dtype)
        col_offsets = col_ids[None, :] + tl.zeros((BLOCK_M, BLOCK_K), dtype=col_ids.dtype)
        a_cur = tl.load(
            A_desc,
            offsets=(row_offsets, col_offsets),
            boundary_check=(0, 1),
            padding_option="zero",
        )

    if (n0 + BLOCK_N <= N) and (k0 + BLOCK_K <= K):
        b_cur = B_desc.load([k0, n0])
    else:
        row_ids = k0 + offs_k
        row_offsets = row_ids[:, None] + tl.zeros((BLOCK_K, BLOCK_N), dtype=row_ids.dtype)
        col_offsets = offs_n[None, :] + tl.zeros((BLOCK_K, BLOCK_N), dtype=offs_n.dtype)
        b_cur = tl.load(
            B_desc,
            offsets=(row_offsets, col_offsets),
            boundary_check=(0, 1),
            padding_option="zero",
        )

    for kt in tl.range(0, K_tiles, num_stages=2):
        k0 = kt * BLOCK_K
        acc += tl.dot(a_cur, b_cur)

        next_k = k0 + BLOCK_K
        if next_k < K:
            if (m0 + BLOCK_M <= M) and (next_k + BLOCK_K <= K):
                a_cur = A_desc.load([m0, next_k])
            else:
                col_ids = next_k + offs_k
                row_offsets = offs_m[:, None] + tl.zeros((BLOCK_M, BLOCK_K), dtype=offs_m.dtype)
                col_offsets = col_ids[None, :] + tl.zeros((BLOCK_M, BLOCK_K), dtype=col_ids.dtype)
                a_cur = tl.load(
                    A_desc,
                    offsets=(row_offsets, col_offsets),
                    boundary_check=(0, 1),
                    padding_option="zero",
                )

            if (n0 + BLOCK_N <= N) and (next_k + BLOCK_K <= K):
                b_cur = B_desc.load([next_k, n0])
            else:
                row_ids = next_k + offs_k
                row_offsets = row_ids[:, None] + tl.zeros((BLOCK_K, BLOCK_N), dtype=row_ids.dtype)
                col_offsets = offs_n[None, :] + tl.zeros((BLOCK_K, BLOCK_N), dtype=offs_n.dtype)
                b_cur = tl.load(
                    B_desc,
                    offsets=(row_offsets, col_offsets),
                    boundary_check=(0, 1),
                    padding_option="zero",
                )

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def tiled_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Tiled matrix multiplication using autotuned Triton kernel."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} != {K2}"
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    MAX_BLOCK_M = 128
    MAX_BLOCK_N = 128
    grid = (triton.cdiv(M, MAX_BLOCK_M), triton.cdiv(N, MAX_BLOCK_N))
    tiled_gemm_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 128,
                'BLOCK_K': 64,
            },
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {
                'BLOCK_M': 64,
                'BLOCK_N': 128,
                'BLOCK_K': 64,
            },
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 64,
                'BLOCK_K': 32,
            },
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel_persistent(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Persistent thread GEMM with tensor descriptors and autotuning."""
    pid = tl.program_id(axis=0)
    np = tl.num_programs(axis=0)

    MT = tl.cdiv(M, BLOCK_M)
    NT = tl.cdiv(N, BLOCK_N)
    TILE_COUNT = MT * NT
    

    A_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    B_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    for tile_idx in range(pid, TILE_COUNT, np):
        pid_m = tile_idx // NT
        pid_n = tile_idx % NT

        m0 = pid_m * BLOCK_M
        n0 = pid_n * BLOCK_N
        
        offs_m = m0 + tl.arange(0, BLOCK_M)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            if (m0 + BLOCK_M <= M) and (k0 + BLOCK_K <= K):
                a = A_desc.load([m0, k0])
            else:
                row_offsets = offs_m[:, None] + tl.zeros((BLOCK_M, BLOCK_K), dtype=offs_m.dtype)
                col_offsets = (k0 + offs_k)[None, :] + tl.zeros((BLOCK_M, BLOCK_K), dtype=offs_k.dtype)
                a = tl.load(
                    A_desc,
                    offsets=(row_offsets, col_offsets),
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
            
            if (n0 + BLOCK_N <= N) and (k0 + BLOCK_K <= K):
                b = B_desc.load([k0, n0])
            else:
                row_offsets = (k0 + offs_k)[:, None] + tl.zeros((BLOCK_K, BLOCK_N), dtype=offs_k.dtype)
                col_offsets = offs_n[None, :] + tl.zeros((BLOCK_K, BLOCK_N), dtype=offs_n.dtype)
                b = tl.load(
                    B_desc,
                    offsets=(row_offsets, col_offsets),
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
            
            acc += tl.dot(a, b)
        

        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


def persistent_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    MT = triton.cdiv(M, 128)
    NT = triton.cdiv(N, 128)
    grid = lambda META: (min(65536, MT * NT),)

    matmul_kernel_persistent[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


# FP8 GEMM Kernel

@triton.jit
def matmul_fp8_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """FP8 matrix multiplication kernel with FP32 accumulation."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        if FP8_AVAILABLE:
            a_fp8 = a.to(tl.float8e4m3fn)
            b_fp8 = b.to(tl.float8e4m3fn)
            acc += tl.dot(a_fp8, b_fp8, out_dtype=tl.float32)
        else:
            acc += tl.dot(a, b, out_dtype=tl.float32)
        

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_fp8(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """FP8 matrix multiplication wrapper."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible dimensions: A.K={K}, B.K={K2}"
    
    if FP8_AVAILABLE and A.dtype != FP8_E4M3_DTYPE:
        A = A.to(FP8_E4M3_DTYPE)
    if FP8_AVAILABLE and B.dtype != FP8_E4M3_DTYPE:
        B = B.to(FP8_E4M3_DTYPE)
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 128
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_fp8_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return C


def benchmark_fp8_vs_fp16() -> None:
    """Benchmark FP8 vs FP16 matrix multiplication."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping FP8 benchmark")
        return
    
    print("\n" + "=" * 80)
    print("FP8 vs FP16 Matrix Multiplication Benchmark (Triton 3.5)")
    print("=" * 80)
    
    sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    
    for M, N, K in sizes:
        print(f"\nMatrix size: {M}x{K} @ {K}x{N}")
        
        A_fp16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B_fp16 = torch.randn(K, N, device="cuda", dtype=torch.float16)
        
        # Use Triton's benchmarking - handles warmup, sync, outliers automatically
        fp16_time = triton.testing.do_bench(lambda: tiled_matmul(A_fp16, B_fp16))
        
        C_fp16 = tiled_matmul(A_fp16, B_fp16)  # For numerical comparison
        
        flops = 2 * M * N * K
        fp16_tflops = flops / (fp16_time * 1e-3) / 1e12
        
        print(f"  FP16: {fp16_time:.2f} ms/iter, {fp16_tflops:.1f} TFLOPS")
        
        if FP8_AVAILABLE:
            A_fp8 = A_fp16.to(FP8_E4M3_DTYPE)
            B_fp8 = B_fp16.to(FP8_E4M3_DTYPE)
            
            # Use Triton's benchmarking - handles warmup, sync, outliers automatically
            fp8_time = triton.testing.do_bench(lambda: matmul_fp8(A_fp8, B_fp8))
            
            C_fp8 = matmul_fp8(A_fp8, B_fp8)  # For numerical comparison
            
            fp8_tflops = flops / (fp8_time * 1e-3) / 1e12
            speedup = fp16_time / fp8_time
            
            print(f"  FP8:  {fp8_time:.2f} ms/iter, {fp8_tflops:.1f} TFLOPS ({speedup:.2f}x speedup)")
            
            max_diff = (C_fp16 - C_fp8).abs().max().item()
            mean_diff = (C_fp16 - C_fp8).abs().mean().item()
            print(f"  Numerical error: max={max_diff:.6f}, mean={mean_diff:.6f}")
        else:
            print(f"  FP8:  Not available (requires PyTorch 2.9+)")
    
    print("\n" + "=" * 80)


@triton.jit
def persistent_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent GEMM kernel with work queue load balancing."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    tiles_per_sm = tl.cdiv(num_tiles, NUM_SMS)
    
    for tile_id in range(pid, num_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        
        if pid_m >= num_pid_m or pid_n >= num_pid_n:
            continue
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_K):
            a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
            a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            
            b_ptrs = B_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
            b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            acc += tl.dot(a, b, out_dtype=tl.float32)
        

        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


def persistent_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Persistent GEMM with reduced launch overhead."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    NUM_SMS = 148  # Blackwell B200 has 148 SMs
    grid = (NUM_SMS,)
    
    persistent_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
        num_warps=8,
        num_stages=3,
    )
    
    return C


def benchmark_persistent_vs_standard():
    """Benchmark persistent kernel performance"""
    print("\n" + "=" * 80)
    print("Persistent Kernel Benchmark")
    print("=" * 80)
    
    device = "cuda"
    sizes = [2048, 4096, 8192]
    
    for size in sizes:
        M = N = K = size
        
        print(f"\nMatrix size: {M}x{K} @ {K}x{N}")
        
        A = torch.randn(M, K, device=device, dtype=torch.float16)
        B = torch.randn(K, N, device=device, dtype=torch.float16)
        
        # Use Triton's benchmarking - handles warmup, sync, outliers automatically
        persistent_time = triton.testing.do_bench(lambda: persistent_matmul(A, B))
        persistent_tflops = (2 * M * N * K) / (persistent_time * 1e-3) / 1e12
        
        print(f"  Persistent kernel: {persistent_time:.2f} ms, {persistent_tflops:.1f} TFLOPS")
    
    print("\n" + "=" * 80)



if __name__ == "__main__":
    print("Running Triton examples...")
    
    benchmark_fp8_vs_fp16()
    benchmark_persistent_vs_standard()
