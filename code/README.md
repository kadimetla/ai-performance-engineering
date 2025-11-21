# AI Systems Performance Engineering

## Summary
Reference implementation of high-performance PyTorch, CUDA, and Triton workloads for NVIDIA Blackwell platforms.
The repository packages 20 focused chapters, advanced labs, and the shared benchmarking harness so you can profile baselines, apply optimizations, and capture artifacts that prove performance gains.

## Learning Goals
- Understand how the chapters, labs, and shared tooling fit together.
- Stand up a reproducible environment for PyTorch 2.10-dev + CUDA 13 workloads on Blackwell GPUs.
- Run the benchmark harness directly or through the Typer CLI for automated artifact capture.
- Validate peak hardware characteristics before grading optimizations against stored expectations.

## Directory Layout
| Path | Description |
| --- | --- |
| `ch1` - `ch20` | One directory per chapter with baseline/optimized benchmarks, workload configs, and `compare.py` harness entrypoints. |
| `labs/` | Deep-dive labs for matmul, routing, FlexAttention, MoE, persistent decode, distributed training, and more. |
| `common/python/` | Shared benchmark harness, logging, workload metadata, and profiling utilities used by every chapter. |
| `tools/cli/benchmark_cli.py` | Typer-based CLI for running, profiling, and verifying targets with reproducible artifacts. |
| `docs/` + `scripts/` | Operational guides, profiling workflows, and setup/reset helpers (`setup.sh`, `cleanup.py`, `reset-gpu.sh`). |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_latest.txt
python tools/cli/benchmark_cli.py list-targets --chapter ch1
python tools/cli/benchmark_cli.py run --targets ch1 --profile minimal
# Speculative decoding config sweep (draft/verify, n-gram, small-chunk variants)
python tools/cli/benchmark_cli.py run --targets labs/speculative_decode:spec_config_sweep --profile none
# Async input pipeline sweep (pin_memory/non_blocking/num_workers/copy-stream)
python tools/benchmarking/async_input_pipeline_sweep.py --copy-stream
# NVLink-C2C KV offload contrast (pageable vs pinned/async)
python tools/cli/benchmark_cli.py run --targets labs/persistent_decode:nvlink_offload --profile none
```
- `setup.sh` installs system prerequisites (drivers, CUDA, Nsight) and should be rerun after driver upgrades.
- Use `python tools/testing/run_all_benchmarks.py --targets ch*` for automated regression suites.
- `python tools/analysis/analyze_expectations.py --artifacts-dir artifacts` compares new runs to stored thresholds.
- For fast local/CI validation, benchmarks default to a lightweight path (`AIPERF_FAST_BENCH=1`). Set `AIPERF_FAST_BENCH=0` to run full-sized Chapter 20 examples.

## Utilities via benchmark_cli
- Run handy calculators from one entrypoint:
  - KV cache sizing: `python tools/cli/benchmark_cli.py utils --tool kv-cache -- --layers 80 --hidden 8192 --tokens 4096 --batch 8 --dtype fp8 --gpu-mem-gb 192 --kv-overhead-frac 0.10 --reserve-activations-gb 40`
  - Cost per token: `python tools/cli/benchmark_cli.py utils --tool cost-per-token -- --avg-power 800 --throughput 1500 --electricity-cost 0.16 --pue 1.5`
  - Precision diff: `python tools/cli/benchmark_cli.py utils --tool compare-precision -- --help`
  - CUTLASS probe: `python tools/cli/benchmark_cli.py utils --tool detect-cutlass`
  - Hardware dump: `python tools/cli/benchmark_cli.py utils --tool dump-hw`
  - Hardware probe: `python tools/cli/benchmark_cli.py utils --tool probe-hw`
- For direct use, the underlying scripts live in `tools/utilities/`.

## Validation Checklist
- `pytest tests/integration` succeeds to confirm harness discovery and CLI plumbing.
- `python tools/benchmarking/benchmark_peak.py` reports TFLOP/s, bandwidth, and NVLink numbers close to the published ceilings.
- `python tools/cli/benchmark_cli.py verify --targets all` completes without regressions before refreshing expectation files.

## Notes
- `tools/profiling/profile.sh` and `ncu_template.ini` capture Nsight traces with consistent metric sets.
- `benchmark_profiles/` and `artifacts/` hold run outputs; clean them via `python cleanup.py` when rotating hardware.
- `docs/perf_intake_and_triage.md` outlines the standard intake bundle for performance investigations.

## TMEM-aware roofline workflow
- Requires a Blackwell GPU with TMA/cluster enabled (B200/B300/GB200). Grace-Blackwell GB10 (sm_121) will skip the TMA/DSMEM matmul lab and produce no Nsight kernels.
- Run through the CLI with profiling plus the metadata hook so kernels emit arithmetic intensity and achieved TFLOP/s alongside Nsight CSVs:
  `python tools/cli/benchmark_cli.py run --targets labs/blackwell_matmul:blackwell_matmul_cluster --profile roofline --artifacts-dir artifacts --target-extra-arg 'labs/blackwell_matmul:blackwell_matmul_cluster="--roofline-meta artifacts/blackwell_matmul_meta.csv"'`
- After the run, feed the Nsight CSV (`artifacts/<run_id>/profiles/.../metrics.csv`) and the metadata CSV into the dual-roofline plotter:
  `python tools/analysis/dual_roofline_plot.py --ncu-csv <metrics.csv> --kernel-meta artifacts/blackwell_matmul_meta.csv --output artifacts/blackwell_matmul_dual_roofline.png`
- The plot labels whether SM compute or TMEM throughput is binding; use it for chapter figures and proof-of-benefit reports.
