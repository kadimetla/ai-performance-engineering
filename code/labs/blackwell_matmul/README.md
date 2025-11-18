# Blackwell Matmul Suite

This suite replays the four-part Modular.org series on matrix multiplication
for Blackwell GPUs by pairing a PyTorch 2.10 benchmark harness with CUDA 13
kernels. Each lesson ships as a `baseline_*.py` / `optimized_*.py` module so you
can run them directly through `tools/cli/benchmark_cli.py` or import them into
notebooks. The CUDA extension is intentionally lightweight so it can compile on
GB100/GB200 devkits as well as Hopper-class fallbacks.

## Why a dedicated suite?

The companion `labs/fullstack_cluster/` directory tells a story about MoE end-to-end systems.
`labs/blackwell_matmul/` zooms into a single GEMM and shows how Grace-Blackwell features
unlock real speedups:

| Blog part | Module | What it demonstrates |
| --- | --- | --- |
| Part 1 – Introduction | `baseline_blackwell_matmul.py` | A naïve CUDA kernel and a PyTorch harness to establish the reference roofline. |
| Part 2 – Hardware features | `optimized_blackwell_matmul_pseudo.py` | Warp-specialized loads with `cuda::pipeline` standing in for per-stage TMA copies and explicit tensor-core accumulation buffers (runs everywhere). |
| Part 2b – Hardware TMA | `optimized_blackwell_matmul_tma.py` | Real TMA path that fails fast when the device reports TMA unsupported. |
| Part 3 – 85% of SOTA | `optimized_blackwell_matmul_pipeline.py` | Multi-tile accumulation using asynchronous stages and PyTorch 2.10 `torch.compile(mode="max-autotune")` at the Python layer. |
| Part 4 – Breaking SOTA | `optimized_blackwell_matmul_cluster.py` | Thread-block clusters broadcast tiles over DSMEM so that only one CTA issues the heavyweight loads per cluster.

Each optimized variant is still callable from Python and keeps FP16 inputs with
FP32 accumulation so you can compare outputs numerically.

## Layout

```
labs/blackwell_matmul/
 ├── __init__.py                      # public API (baseline + optimized callables)
 ├── blackwell_benchmarks.py          # BaseBenchmark wrapper for CLI + notebooks
 ├── baseline_blackwell_matmul.py     # PyTorch harness for the introductory kernel
 ├── optimized_blackwell_matmul_tma.py        # Part 2 port (pipeline loads)
 ├── optimized_blackwell_matmul_pipeline.py   # Part 3 port (multi-stage accumulators)
 ├── optimized_blackwell_matmul_cluster.py    # Part 4 port (cluster DSMEM broadcast)
 ├── grace_blackwell_extension.py     # torch.utils.cpp_extension loader
 ├── grace_blackwell_kernels.cu       # CUDA 13 kernels + pybind11 bindings
 └── run_blackwell_matmul.py         # Standalone CLI runner for quick demos
```

## Building and running

The first invocation builds the extension automatically:

```bash
python - <<'PY'
from labs.blackwell_matmul import baseline_blackwell_matmul
import torch
x = torch.randn(128, 128, device='cuda', dtype=torch.float16)
y = torch.randn(128, 128, device='cuda', dtype=torch.float16)
baseline_blackwell_matmul(x, y)
PY
```

Benchmark *all* variants through the standard harness (timings + Nsight profiling artifacts when `--profile` is set):

```bash
python tools/cli/benchmark_cli.py run \
  --targets labs/blackwell_matmul:blackwell_matmul \
  --targets labs/blackwell_matmul:blackwell_matmul_tma \
  --targets labs/blackwell_matmul:blackwell_matmul_pipeline \
  --targets labs/blackwell_matmul:blackwell_matmul_cluster \
  --profile
```

or with the dedicated runner:

```bash
python labs/blackwell_matmul/run_blackwell_matmul.py --variant cluster --size 4096
```

Set `CUDA_VISIBLE_DEVICES` if you need to target Grace-only hosts. Cluster runs
require `cudaDevAttrClusterLaunch=1` and will fail fast on GPUs without DSMEM
support. Use the pipeline variant when clusters are unavailable.
TMA-backed copies only become available on Blackwell SM100/SM103 (B100/B200/B300);
on earlier devices the TMA example uses the shared-memory pipeline emulation so
the same scripts stay runnable everywhere.
