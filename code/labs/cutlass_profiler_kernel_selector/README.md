# CUTLASS Profiler Kernel Selector Lab

Purpose: pick the fastest stock CUTLASS GEMM kernels for transformer-like shapes, then sanity-check custom paths (Triton, DeepEP, custom PTX) against that baseline. Use it when you change GPUs/CUDA, upgrade CUTLASS, or validate custom kernels.

## Prereqs
- Run `./setup.sh` (builds `build/cutlass_profiler/tools/profiler/cutlass_profiler` and exports `CUTLASS_PROFILER_BIN`; see `artifacts/cutlass_profiler_env.sh`). Override with your own binary via `CUTLASS_PROFILER_BIN`.
- CUDA GPU + Tensor Cores; Triton optional for comparison.

## Shapes we sweep
Transformer-ish GEMMs (prefill + decode): `M,N ∈ {4096..16384}`, `K ∈ {4096,8192}`. Edit `labs/cutlass_profiler_kernel_selector/shapes.py` to adjust.

## 1) Generate CUTLASS baselines
```bash
python labs/cutlass_profiler_kernel_selector/run_cutlass_profiler_sweep.py \
  --output-dir artifacts/cutlass_profiler
# optional: --shapes decode_mlp_m4096_n4096_k8192 kv_proj_m4096_n4096_k4096
```
Creates per-shape CSV/logs plus `artifacts/cutlass_profiler/cutlass_profiler_results.json` with best kernel, runtime, and TFLOP/s.

## 2) (Optional) Triton baseline for the same shapes
```bash
python labs/cutlass_profiler_kernel_selector/run_triton_matmul_baseline.py \
  --output-dir artifacts/cutlass_profiler \
  --warmup 5 --iters 10
```
Uses a local Triton matmul kernel (no `triton.ops` dependency); writes `artifacts/cutlass_profiler/triton_matmul_results.json`.

## 3) Compare against CUTLASS
```bash
python labs/cutlass_profiler_kernel_selector/compare_against_baselines.py \
  --include-default-triton \
  --providers artifacts/cutlass_profiler/deepep_results.json
```
- Reads `cutlass_profiler_results.json` plus any competitor JSONs and reports TFLOP/s + speedup vs CUTLASS.
- Drop in your own custom results (DeepEP, custom PTX, etc.) using the same schema:
```json
{
  "provider": "deepep",
  "results": [
    {"name": "decode_mlp_m4096_n4096_k8192", "tflops": 500.0, "runtime_ms": 0.55, "kernel": "my_deepep_kernel"}
  ]
}
```

## Files
- `run_cutlass_profiler_sweep.py` – drives `cutlass_profiler` for each shape and captures the best kernel per shape.
- `run_triton_matmul_baseline.py` – custom Triton matmul kernel for quick comparisons.
- `compare_against_baselines.py` – collates CUTLASS vs Triton/DeepEP/custom results and emits a comparison table + JSON.
- `shapes.py` – centralized list of transformer GEMM shapes.

Use this lab whenever you port to a new GPU/CUDA, or before/after Triton/DeepEP kernel changes, to ensure your custom wins beat the fastest stock CUTLASS kernels.
