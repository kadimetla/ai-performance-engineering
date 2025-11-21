# Lab - FlashAttention Gluon

## Summary
Contrast a naive unfused attention (matmul + softmax + matmul) with a fused, warp-specialized FlashAttention kernel. The optimized path prefers a Gluon/Triton kernel; if Gluon is unavailable, it falls back to the flash-attn fused kernel (warp-specialized on Blackwell).

## Workloads
- `labs.flashattention_gluon.baseline_flashattention_gluon`: unfused attention, math softmax path.
- `labs.flashattention_gluon.optimized_flashattention_gluon`: fused Gluon/flash-attn kernel.

## Running
```bash
cd ai-performance-engineering
python tools/cli/benchmark_cli.py list-targets --chapter labs/flashattention_gluon
python tools/cli/benchmark_cli.py run --targets labs/flashattention_gluon:baseline_flashattention_gluon --profile minimal
python tools/cli/benchmark_cli.py run --targets labs/flashattention_gluon:optimized_flashattention_gluon --profile minimal
```

## Requirements
- NVIDIA GPU with CUDA.
- Gluon or flash-attn installed (setup.sh installs flash-attn; Gluon install is attempted there as well).

## What to inspect
- NVTX ranges `flashattention_baseline_unfused` vs `flashattention_optimized_<provider>`.
- Provider metric indicates whether Gluon or flash-attn was used.
- Expect fused path to show fewer kernels and higher throughput due to warp specialization and on-chip softmax.
