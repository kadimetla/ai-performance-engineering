# cuDNN SDPA Bench Lab

Measure cuDNN fused scaled-dot-product attention versus Flash/math backends with explicit CLI control of the SDPA backend.

## What it runs
- `baseline_flash_sdp.py`, `optimized_flash_sdp.py`: identical attention microbenchmarks; backend is selected via CLI (`--backend {auto,cudnn,flash,math}`) passed through `--target-extra-arg`.
- `expectations_gb10.json`: current regression thresholds captured on GB10 (Hopper-class expectations will differ).

## How to run
- List targets:  
  `python tools/cli/benchmark_cli.py list-targets --chapter labs/cudnn_sdpa_bench`
- Run with cuDNN backend and capture Nsight traces:  
  `python tools/cli/benchmark_cli.py run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend cudnn"`
- Compare Flash backend:  
  `python tools/cli/benchmark_cli.py run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend flash"`
- Math backend sanity:  
  `python tools/cli/benchmark_cli.py run --targets labs/cudnn_sdpa_bench:flash_sdp --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend math"`

## Notes
- Backend selection is CLI-only; environment variables are intentionally ignored.
- Profiling outputs land under `benchmark_profiles/labs/cudnn_sdpa_bench/<run_id>` and the harness artifacts under `artifacts/<run_id>/...`.
