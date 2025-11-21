# KV-Cache Compression Lab (FP8 → NVFP4)

Benchmarks a KV-cache heavy attention block using Transformer Engine 2.9 recipes. The baseline runs MXFP8 block scaling; the optimized variant calibrates in FP8 and then switches to NVFP4 block scaling when FP4 tensor cores are available.

## Targets
- `labs/kv_cache_compression:kv_cache` – Baseline MXFP8 path
- `labs/kv_cache_compression:kv_cache_nvfp4` – Optimized NVFP4 path (falls back to MXFP8 if FP4 kernels are unavailable)

## Run
```bash
python tools/cli/benchmark_cli.py list-targets --chapter labs/kv_cache_compression
python tools/cli/benchmark_cli.py run --targets labs/kv_cache_compression:kv_cache_nvfp4 --profile minimal
```

## Notes
- Requires Blackwell-class FP4 support for NVFP4; otherwise the NVFP4 benchmark reports a graceful fallback to MXFP8.
- Tensor dimensions are multiples of 16 to satisfy FP4/FP8 GEMM kernel constraints.
- No cross-chapter imports; everything is self-contained under `labs/kv_cache_compression/`.
