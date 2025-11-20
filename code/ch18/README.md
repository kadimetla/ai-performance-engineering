# Chapter 18 - Advanced Attention & Decoding

## Summary
Collects modern decoder techniques-FlexAttention, FlexDecoding, speculative and paged attention workflows-implemented in both PyTorch and CUDA/Triton so you can iterate quickly while validating kernels on real hardware.

## Learning Goals
- Prototype FlexAttention/FlexDecoding workloads with custom masks, score mods, and KV-cache integration.
- Evaluate speculative decoding pipelines that trade extra compute for lower latency.
- Test tensor-core optimized attention kernels tailored for Blackwell tmem limits.
- Validate integration points with serving frameworks (vLLM) using the provided runners.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flexdecoding.py`, `optimized_flexdecoding.py`, `optimized_flexdecoding_math.py`, `baseline_speculative_decoding.py`, `optimized_speculative_decoding.py`, `optimized_speculative_decoding_math.py` | Decoding workloads comparing eager vs math-only vs optimized pipeline variants. |
| `baseline_tensor_cores.py`, `optimized_tensor_cores.py`, `flashmla_kernel.cu`, `warp_specialized_triton.py` | Tensor-core attention kernels plus Triton equivalents for rapid validation. |
| `flex_attention_native.py`, `flex_attention_enhanced.py`, `flex_attention_large_model.py`, `kv_cache_integration_example.py` | FlexAttention examples ranging from toy sizes to large models with KV-cache reuse. |
| `run_vllm_decoder.py`, `configs/`, `paged_attn_common.py`, `workload_config.py` | Serving integrations and config presets for pushing workloads through vLLM or custom harnesses. |
| `compare.py`, `expectations_gb10.json`, `test_flex_attention.py` | Harness entry, regression thresholds, and pytest coverage for FlexAttention APIs. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch18
python compare.py --profile none
python tools/cli/benchmark_cli.py list-targets --chapter ch18
python tools/cli/benchmark_cli.py run --targets ch18 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python optimized_flexdecoding.py --profiling` reports significantly fewer kernels and lower latency than the baseline while matching decoded tokens.
- `python run_vllm_decoder.py --config configs/flex_attention.yaml` completes with accuracy parity vs the native FlexAttention path.
- `python test_flex_attention.py` passes locally, confirming mask/score-mod helpers are wired correctly.

## Notes
- `flex_attention` scripts accept env vars like `BLOCK_SIZE`, `DOC_SPAN`, and `SEQ_LEN` so you can sweep shapes without editing code.
- `flashmla_kernel.cu` includes the Blackwell-specific tensor memory guard to keep compilation healthy on SM121 hardware.
