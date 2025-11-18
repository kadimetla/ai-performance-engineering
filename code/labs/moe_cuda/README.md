# MoE CUDA Tuning Suite

This suite stitches together the mixture-of-experts (MoE) techniques described in `docs/moe_gpu_cuda_tuning_showcase.md` and the corresponding chapters in `book/ch*.md`.  Each `baseline_*.py` / `optimized_*.py` pair is a runnable benchmark that plugs directly into the shared harness (`common/python/benchmark_harness.py`).  Run them with:

```bash
python labs/moe_cuda/baseline_router.py
python labs/moe_cuda/optimized_router.py
```

Each pair maps to real-world GPU/CUDA tuning breakthroughs:

1. **Router Density vs. Top-K Expert Dispatch**
   - Baseline activates *all* experts per token, mimicking the naive MoE design discussed early in Chapter 15.
   - Optimized version uses a top-2 router with capacity-factor enforcement and `torch.compile`, mirroring DeepSeek-V3 and Google GLaM’s sparse dispatch strategies.
   - **Progression step:** `optimized_router_vectorized.py` vectorizes expert dispatch (scatter/index_add), lowers batch size for GB10, and captures the forward pass in a CUDA graph to shave launch overhead.

2. **KV Cache Transfer Overlap**
   - Baseline copies the prefetched KV cache back to the decode service sequentially.
   - Optimized version pipelines compute and NVLink-style transfers using CUDA streams and events (Chapters 11, 15, and 17).
   - **Progression step:** `optimized_kv_transfer_graphs.py` deepens the pipeline, keeps GEMMs in bf16 with `torch.compile`, and captures the overlap loop in a CUDA graph to trim launch overhead on GB10.

3. **Decode Attention Kernel**
   - Baseline issues many small matmul + softmax kernels per head.
   - Optimized version fuses the decode step using `torch.nn.functional.scaled_dot_product_attention`, `torch.compile`, and BF16 autocast for FlashMLA-style efficiency (Chapters 18–20).
   - **Progression step:** `optimized_decode_attention_math.py` forces the non-Flash SDP backend as a separate math-only path so GB10/SM121 can run the same shapes even when Flash kernels are unavailable.

4. **Custom CUDA Decode Kernels**
   - `baseline_decode_kernel.cu` loads tiles directly from HBM using cooperative threads (Chapter 10 fallback path).
   - `optimized_decode_kernel.cu` reuses the Blackwell TMA double-buffering pattern to overlap bulk copies with compute, exposing the same speedup the book highlights for ThunderMLA-style kernels. This path requires Hopper/Blackwell GPUs (CUDA 13+) with Tensor Memory Accelerator support; the harness will mark it `SKIPPED` automatically on older hardware.

All scripts honor the benchmark protocol, emit NVTX ranges when enabled, and can serve as the backbone of MoE-focused workshops.

## Running the progression steps

```bash
# Deeper KV overlap with CUDA graphs
python labs/moe_cuda/optimized_kv_transfer_graphs.py

# Non-Flash decode attention fallback (runs on GB10/SM121)
python labs/moe_cuda/optimized_decode_attention_math.py

# Vectorized router dispatch with CUDA graphs
python labs/moe_cuda/optimized_router_vectorized.py
```

Each script is discoverable by the shared benchmark harness and can be compared against
its corresponding baseline/optimized pair to show the incremental gains on GB10.
