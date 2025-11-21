# Lab - Dynamic Prefill/Decode Router

## Summary
Simulates and benchmarks dynamic routing policies for large-scale inference: split GPUs into prefill/decode pools, monitor TTFT/TPOT, honor KV locality, and migrate traffic only when the score gap warrants it.

## Learning Goals
- Compare naive round-robin routing with telemetry-driven policies that stabilize TTFT.
- Prototype migration budgets, KV-locality boosts, and per-pool thresholds.
- Drive the router against synthetic workloads or real vLLM engines.
- Export detailed metrics (TTFT, TPOT, queue depth) for visualization.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_router.py`, `optimized_router.py`, `driver.py` | Core router logic plus a synthetic simulator for deterministic comparisons. |
| `baseline_dynamic_router.py`, `optimized_dynamic_router.py` | Harness-facing benchmarks derived from the simulator. |
| `baseline_dynamic_router_vllm.py`, `optimized_dynamic_router_vllm.py`, `vllm_runner.py` | Integrations for running the routing policy against vLLM instances. |
| `baseline_dual_pool_vllm.py`, `optimized_dual_pool_vllm.py` | Shared-pool vs dual-pool TTFT benchmarks that reuse `vllm_runner.py`. |
| `baseline_eval_stack.py`, `optimized_eval_stack.py`, `eval_stack.py` | Cheap quality + latency + MoE telemetry stack that emits the six “thin checks” scorecard; requires vLLM + `gpt-oss-20b/original` on GPU (fails fast if missing). |
| `scorecard.py` | CLI to render a ship/no-ship table from emitted JSONLs (text + optional matplotlib PNG). |
| `topology.py`, `topology_probe.py` | NUMA/GPU topology helpers plus a harness target that writes `artifacts/topology/topology.json` for routing hints. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python tools/cli/benchmark_cli.py list-targets --chapter labs/dynamic_router
python tools/cli/benchmark_cli.py run --targets labs/dynamic_router --profile minimal
```
- Targets follow the `labs/dynamic_router:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/dynamic_router:<workload>="--flag value"` to sweep schedule knobs.
- Run the cheap eval stack directly: `python tools/cli/benchmark_cli.py run --targets labs/dynamic_router:eval_stack --profile none` (runs both baseline and optimized variants and writes scorecards under `artifacts/dynamic_router/cheap_eval/`).
- The cheap eval stack will error if vLLM, CUDA, or the model path are missing—no synthetic fallback remains. Override the model path in code if you need a different checkpoint.
- Render a CLI table (and optional plot) from past runs: `python labs/dynamic_router/scorecard.py artifacts/dynamic_router/cheap_eval/baseline_* artifacts/dynamic_router/cheap_eval/optimized_* --plot artifacts/dynamic_router/cheap_eval/scorecard.png`.

### Cheap eval stack (6 checks)
- Mirrors the blog post: mini quality slice (MMLU-mini/GSM8K-lite/TruthfulQA-lite/domain), TTFT p50/p95 split by warm/cold, decode p50/p95 for 128/512/2048-token outputs, MoE token-drop %, expert imbalance (CV), router entropy/margin, plus throughput/goodput in a single scorecard.
- Artifacts: `quality.jsonl`, `latency.jsonl`, `moe_router.jsonl`, `moe_traffic.jsonl`, `tps_goodput.json`, and `scorecard.json` under `artifacts/dynamic_router/cheap_eval/<mode_timestamp>/`. Scorecard includes reference thresholds (+10–15% headroom on TTFT/decode p95 vs baseline, ≤0.25 CV, ≤0.5% drops) so you can eyeball ship/no-ship.
- Baseline vs optimized variants only change priors (quality bias, TTFT/decode base, MoE spillover odds) so you can plug in real engine hooks without touching the run layout.
- Replay real telemetry via flags (no env vars): `python tools/cli/benchmark_cli.py run --targets labs/dynamic_router:eval_stack --profile none --target-extra-arg labs/dynamic_router:eval_stack="--metrics-dir /runs/<model>@<ckpt>/<date> --baseline-scorecard /path/to/baseline/scorecard.json --allow-missing-metrics"`. Add `--no-vllm` to force synthetic quality or `--model-path ... --max-gen-tokens ...` when using vLLM in the quality slice. The same `target-extra-arg` applies to both baseline and optimized eval_stack runs because they share the `labs/dynamic_router:eval_stack` label.
- File format expectations when replaying telemetry: `quality.jsonl` rows with `task`, `prediction`, `expected`, `correct`; `latency.jsonl` rows with `ttft_ms`, `decode_ms`, `output_tokens`, `case` (warm|cold); `moe_router.jsonl` rows with `entropy`, `margin`, `drops` (plus optional `total_tokens`); `moe_traffic.jsonl` rows with `expert_hist` (list) and optional `imbalance_cv`. `tps_goodput.json` is optional—when absent we recompute throughput/goodput from latency + drop rate.

### vLLM Dual-Pool TTFT Lab
- Baseline: shared prefill/decode pool. Optimized: dedicated prefill and decode pools.
- Dependencies: `pip install vllm` plus a local model path; at least two visible GPUs. Runs will be skipped if vLLM is absent or only one GPU is present.
- Setup helper: `setup.sh` builds vLLM from source (default tag `${VLLM_VERSION_TAG:-v0.11.2}`) against the local CUDA/PyTorch stack; set `VLLM_VERSION_TAG` to override when running setup.
- Requirements: provide `--model /path/to/model`, and pass distinct GPU ids via `--prefill-gpus` / `--decode-gpus` for dual mode (defaults to `0` and `1`).
- Run: `python tools/cli/benchmark_cli.py run --targets labs/dynamic_router:dual_pool_vllm --target-extra-arg labs/dynamic_router:dual_pool_vllm="--model /path/to/model --prefill-gpus 0 --decode-gpus 1"`
- Optional knobs (flags): `--long-prompt-tokens` (default 4096), `--short-prompt-tokens` (128), `--prefill-burst` (6), `--decode-requests` (48), `--continue-requests` (48), `--max-tokens` (32), `--prefill-ctx-thresh` (2048).
- Metrics: overall and per-pool TTFT p50/p95, mean queue depth per pool, and per-GPU TPOT samples.

## Validation Checklist
- `python labs/dynamic_router/driver.py --mode baseline` vs `--mode optimized` shows lower TTFT variance and higher TPOT for the optimized policy.
- `python labs/dynamic_router/driver.py --mode optimized --scenario flagship_vs_mid --arrival-rate 1.2 --log-json artifacts/dynamic_router/flagship_vs_mid.json` followed by `python labs/dynamic_router/plot_right_sized.py artifacts/dynamic_router/*.json` renders TTFT p95 and goodput-per-dollar comparisons for right-sized mixes.
- `python tools/cli/benchmark_cli.py run --targets labs/dynamic_router --profile minimal` records artifacts comparing baseline/optimized harness runs. For quicker smoke tests, set `--profile none` and pass `--target-extra-arg labs/dynamic_router:dynamic_router="--ticks 120 --arrival-rate 1.0"` to cut runtime/log volume.
- `python tools/cli/benchmark_cli.py run --targets labs/dynamic_router:dynamic_router_vllm --target-extra-arg labs/dynamic_router:dynamic_router_vllm="--model /path/to/model --decode-gpus 0,1"` succeeds on hosts with at least two GPUs and a local model copy.
- `python tools/cli/benchmark_cli.py run --targets labs/dynamic_router:dual_pool_vllm --target-extra-arg labs/dynamic_router:dual_pool_vllm="--model /path/to/model --prefill-gpus 0 --decode-gpus 1"` contrasts shared versus dual pools and emits per-pool TTFT and queue depth.
- `python tools/cli/benchmark_cli.py run --targets labs/dynamic_router:eval_stack --profile none` emits the cheap six-check scorecard (quality, TTFT/latency, MoE health, throughput/goodput) into `artifacts/dynamic_router/cheap_eval/`.
- `python labs/dynamic_router/scorecard.py <run_dir> [<run_dir>...] --plot score.png` prints a ship/no-ship table and saves a PNG when matplotlib is available.
- `python tools/cli/benchmark_cli.py run --targets labs/dynamic_router:topology_probe` persists GPU↔NUMA mappings to `artifacts/topology/topology.json` for downstream router hints.

## Notes
- `driver.py` accepts knobs such as `--scenario`, `--arrival-rate`, `--burst-factor`, `--decode-cost-penalty`, and `--log-json` to stress right-sized GPU mixes and emit plotting-ready summaries.
- vLLM integration now takes flags (`--model`, `--prefill-gpus`, `--decode-gpus`, etc.) rather than environment variables.
- Router scoring now factors KV locality, queue depth urgency, and an optional decode cost penalty; feed real mappings via `topology_probe.py` when running on Grace/dual-socket systems.
- `eval_stack.py` is mock-friendly: swap the synthetic generators for real engine hooks (TTFT, decode latency, router entropy, expert histogram, drop rate) without changing the run folder layout.
