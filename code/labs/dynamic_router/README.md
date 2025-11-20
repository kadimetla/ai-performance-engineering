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

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python tools/cli/benchmark_cli.py list-targets --chapter labs/dynamic_router
python tools/cli/benchmark_cli.py run --targets labs/dynamic_router --profile minimal
```
- Targets follow the `labs/dynamic_router:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/dynamic_router:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python labs/dynamic_router/driver.py --mode baseline` vs `--mode optimized` shows lower TTFT variance and higher TPOT for the optimized policy.
- `python tools/cli/benchmark_cli.py run --targets labs/dynamic_router --profile minimal` records artifacts comparing baseline/optimized harness runs.
- `VLLM_MODEL=/path/to/model python tools/cli/benchmark_cli.py run --targets labs/dynamic_router:dynamic_router_vllm` succeeds on hosts with at least two GPUs and a local model copy.

## Notes
- `driver.py` accepts knobs such as `--prefill-gpus`, `--decode-gpus`, and `--migration-budget` to stress different regimes.
- vLLM integration requires the `VLLM_MODEL` env var plus the associated tokenizer/model weights present on disk.
