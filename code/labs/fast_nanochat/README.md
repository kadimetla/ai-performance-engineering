# Fast NanoChat Lab

Baseline vs optimized decode microbenchmark inspired by [karpathy/nanochat](https://github.com/karpathy/nanochat). It measures a lightweight LLM-style decode loop and applies common throughput optimizations:

- Baseline: eager execution, pageable host prompts, default stream only.
- Pinned: pinned host prompts, default stream.
- Streams: pinned + copy/compute streams.
- Compile: streams + `torch.compile` for decode.
- Graph: compile + decode graph.
- Graph (full): compile + full prefill+decode graph.
- FP8: full graph + Transformer Engine FP8.
- FP4: full graph + Transformer Engine FP4 (when NVFP4 is available).
- Warp specialized: Triton fused decode MLP with warp specialization/TMA-style block pointers and persistent prefill state.
- 8xGPU: multi-GPU launch to stress NVLink-C2C on an 8x B200 node.

Run via the unified CLI:
```bash
python tools/cli/benchmark_cli.py run --targets labs/fast_nanochat --profile none
```

Or run a specific target:
- `labs/fast_nanochat:baseline_fast_nanochat`
- `labs/fast_nanochat:optimized_fast_nanochat_pinned`
- `labs/fast_nanochat:optimized_fast_nanochat_streams`
- `labs/fast_nanochat:optimized_fast_nanochat_compile`
- `labs/fast_nanochat:optimized_fast_nanochat_graph`
- `labs/fast_nanochat:optimized_fast_nanochat_graph_full`
- `labs/fast_nanochat:optimized_fast_nanochat_fp8`
- `labs/fast_nanochat:optimized_fast_nanochat_fp4`
- `labs/fast_nanochat:optimized_fast_nanochat_warp_specialized`
- `labs/fast_nanochat:optimized_fast_nanochat_8xgpu` (requires torchrun with 8 GPUs)

All variants now emit TTFT (ms), TPOT mean (ms), decode time, total time, and tokens/sec in `custom_metrics` for easy comparison via the benchmark CLI output.
