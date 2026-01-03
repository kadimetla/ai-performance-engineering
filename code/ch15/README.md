# Chapter 15 - Disaggregated Inference & KV Management

## Summary
Addresses large-scale inference concerns: disaggregated compute/storage, KV-cache pooling over NVLink, continuous batching, and mixture-of-experts serving patterns.

## Learning Goals
- Benchmark monolithic vs disaggregated inference paths and quantify fabric costs.
- Design KV-cache managers that gracefully span local and remote HBM pools.
- Implement continuous batching and queueing so decode throughput stays high.
- Serve MoE models efficiently by pairing routing with optimized communication.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_inference_monolithic.py`, `optimized_inference_monolithic.py` | Single-box inference loops that establish the baseline before disaggregation. |
| `disaggregated_inference_multigpu.py` | Disaggregated inference demo that layers speculative decoding on top of prefill/decode pools. |
| `baseline_disaggregated_inference.py`, `optimized_disaggregated_inference.py`, `baseline_disaggregated_inference_multigpu.py`, `optimized_disaggregated_inference_multigpu.py`, `baseline_prefill_decode_disagg.py`, `optimized_prefill_decode_disagg.py`, `baseline_prefill_decode_disagg_multigpu.py`, `optimized_prefill_decode_disagg_multigpu.py`, `disaggregated_inference_single_common.py` | Disaggregated pipelines modeling remote prefills, decode overlap, and NVLink pooling (single- and multi-GPU), plus shared single-GPU helpers. |
| `baseline_kv_cache_management.py`, `optimized_kv_cache_management.py`, `optimized_kv_cache_management_math.py`, `baseline_kv_cache_nvlink_pool.py`, `optimized_kv_cache_nvlink_pool.py`, `baseline_kv_cache_nvlink_pool_multigpu.py`, `optimized_kv_cache_nvlink_pool_multigpu.py` | KV-cache orchestration utilities with local-only, math-only, and NVLink-pooled variants. |
| `baseline_continuous_batching.py`, `optimized_continuous_batching.py` | Single-GPU continuous batching scheduler for TTFT-aware queueing. |
| `baseline_continuous_batching_multigpu.py`, `optimized_continuous_batching_multigpu.py` | Multi-GPU continuous batching scheduler for scaled queueing throughput. |
| `baseline_moe_inference.py`, `optimized_moe_inference.py` | Inference-specific MoE workloads that pair router load with communication control. |
| `baseline_moe_overlap.py`, `optimized_moe_overlap_shared_expert.py`, `baseline_wide_ep.py`, `optimized_wide_ep.py`, `baseline_moe_routing_simple.py`, `optimized_moe_routing_simple_topology_aware.py` | MoE expert-parallel microbenchmarks illustrating overlap, packing/unpacking, and topology-aware routing dispatch. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `Makefile` | Harness entry and dependencies for inference-focused validation. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python ch15/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch15
python -m cli.aisp bench run --targets ch15 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python -m cli.aisp bench run --targets ch15:disaggregated_inference_multigpu --profile minimal --ncu-replay-mode kernel` shows reduced fabric stalls compared to the baseline while maintaining accuracy parity (kernel replay avoids NCU application-replay stalls on this workload).
- `python optimized_kv_cache_management.py --validate` confirms eviction + promotion policies keep decode latency within the budget.
- `python compare.py --examples continuous_batching` (single GPU) and `python compare.py --examples continuous_batching_multigpu` (multi-GPU) show optimized scheduling increases tokens/sec vs naive queue draining.

## Notes
- `disaggregated_inference_multigpu.py` can run purely in simulation mode; set `--simulate-network` when hardware isn't wired for NVLink pooling.
- Use `torchrun --nproc_per_node <num_gpus>` to run the disaggregated pipeline on the desired GPU count (defaults to all visible GPUs, even count).
- `Makefile` wraps the MPI/UCX targets needed for the multi-node decode experiments.
