# Lab - Decode Optimization

## Summary
Decode-focused microbenchmarks that demonstrate how common serving optimizations (pinned memory, streams, compile/graphs, FP8/FP4, warp specialization, multi-GPU) affect TTFT, TPOT, and throughput.

Uses a simplified MLP-based model (not a transformer) to isolate optimization effects without attention complexity.

## Learning Goals
- Contrast eager vs pinned/streamed vs compiled/graph decode paths on the same workload.
- Measure FP8/FP4 tensor-core benefits relative to FP16/BF16 baselines.
- Validate Triton warp-specialized decode kernels against Python math and harness expectations.
- Observe NVLink-C2C behavior by scaling the decode loop to 8×B200.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_decode.py` | Eager decode baseline with pageable prompts on a single stream. |
| `baseline_decode_fp8.py`, `baseline_decode_fp4.py` | BF16 baselines matching the FP8/FP4 prefill-focused workloads. |
| `baseline_decode_warp_specialized.py` | Eager baseline matching the warp-specialized Triton workload. |
| `baseline_decode_double_buffer_tma.py` | Baseline CUDA kernel for the TMA double-buffered decode comparison. |
| `optimized_decode_pinned.py`, `optimized_decode_streams.py` | Pinned-host and dual-stream variants that remove host bottlenecks. |
| `optimized_decode_compile.py`, `optimized_decode_graph.py`, `optimized_decode_graph_full.py` | `torch.compile` and CUDA Graph variants (decode-only and full prefill+decode). |
| `optimized_decode_fp8.py`, `optimized_decode_fp4.py` | Transformer Engine FP8/FP4 decode paths (FP4 falls back gracefully when unsupported). |
| `optimized_decode_warp_specialized.py`, `triton_fused_decode.py` | Triton fused decode MLP with warp specialization/TMA-style pointers. |
| `decode_8xgpu_demo.py` | Multi-GPU demo/tool to stress NVLink-C2C bandwidth on 8×B200 (torchrun required). |
| `optimized_decode_double_buffer_tma.py` | CUDA double-buffered decode using TMA-style loads. |
| `decode_common.py`, `expectations_b200.json`, `__init__.py` | Shared helpers, regression thresholds, and harness target exports. |

## Running the Benchmarks
Use the benchmark harness for repeatable runs and artifact capture.
```bash
cd ai-performance-engineering
python -m cli.aisp bench list-targets --chapter labs/decode_optimization
python -m cli.aisp bench run --targets labs/decode_optimization --profile none
```
- Target labels: `baseline_decode`, `optimized_decode_pinned`, `..._streams`, `..._compile`, `..._graph`, `..._graph_full`, `..._fp8`, `..._fp4`, `..._warp_specialized`.
- Pass extra flags via `--target-extra-arg labs/decode_optimization:<target>="--flag value"`.

To run the 8×GPU NVLink-C2C demo, use the demos runner:
```bash
python -m cli.aisp demos labs-decode-8xgpu --nproc-per-node 8 -- --iters 4 --warmup 1
```

## Validation Checklist
- Baseline vs pinned/streams shows improved TTFT and TPOT with lower host wait time.
- Compile/graph variants emit fewer kernels and higher tokens/sec than the baseline in harness output.
- FP8/FP4 runs use a prefill-focused workload (decode_tokens=0) to surface tensor-core benefits; outputs remain within tolerance.
- Warp-specialized Triton kernel is validated against a workload-matched eager baseline; expectation file stays green.
- 8×GPU demo exercises NVLink-C2C without graph capture failures when launched via `torchrun`.

## Notes
- All targets emit TTFT, TPOT mean, decode time, total time, and tokens/sec in `custom_metrics` for easy diffing.
- FP4 path requires NVFP4 (Blackwell); otherwise it reports the fallback in logs.
- Multi-GPU demo expects eight visible GPUs; reduce `--world-size` if you customize the script for smaller boxes.
- This lab uses a simplified MLP model (no attention) to isolate serving optimization effects.
