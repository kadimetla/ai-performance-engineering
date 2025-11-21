# GPU/CUDA/PyTorch Tuning Showcase

This walkthrough stitches together the frontier-lab lessons that run through `book/ch*.md` and shows how to demonstrate them with the chapter code that already lives in this repo. The focus stays on Mixture-of-Experts (MoE) GPU/CUDA tuning, leaning on real deployments from DeepSeek, Google DeepMind, Microsoft, NVIDIA, and friends. Each stage is intentionally playful so teams can build intuition incrementally instead of trying to absorb every optimization in one sitting.

Run everything with `python tools/cli/benchmark_cli.py run --accept-regressions` to refresh expectations (no env vars needed). The harness now marks known hardware gaps as `SKIPPED`—DSMEM-only cluster kernels on GB10, NVFP4-only kernels on non-FP4 parts, and TF32 being disabled via `configure_tf32/restore_tf32` helpers—so your demo logs stay green.


## Frontier-Lab Inspirations

### 1. DeepSeek-V3 / DeepSeek-R1: DualPipe + FlashMLA + DeepEP
- **What happened:** `book/ch1.md` details how DeepSeek-V3 trained ~680B-parameter MoE models on export-limited H800s by activating only ~37B parameters per token and inventing the DualPipe overlap strategy. `book/ch9.md` and `book/ch18.md` expand on DeepEP’s inline PTX cache hints and the FlashMLA decode kernel that keeps single-token latency low.
- **GPU/CUDA tuning angles:** DualPipe overlaps NVLink-lite copies with compute, FlashMLA fuses multi-head latent attention into one kernel, and DeepEP bypasses L1 with custom PTX for all-to-all dispatch buffers.
- **Repo demo:**  
  1. Start with `python tools/cli/benchmark_cli.py run --targets ch15:expert_parallelism` to capture the dense versus sparse MoE pair (`ch15/baseline_expert_parallelism.py`, `ch15/optimized_expert_parallelism.py`).  
  2. Layer in overlapping collectives by running `ch4/optimized_no_overlap.py` with NVTX ranges enabled (`--profile`).  
  3. Jump to `ch18/flashmla_kernel.cu` and `ch18/run_vllm_decoder.py` to benchmark FlashMLA-style decode workers, then contrast with `ch18/baseline_streams.py` to visualize idle gaps.  
  4. Use `ch9/optimized_cutlass_gemm.py` plus `nsight-compute` reports to see how much arithmetic intensity the custom PTX adds when you switch the DeepEP-style load path on/off.

### 2. Google Switch Transformer + GLaM: Load-balanced routing
- **What happened:** The Switch Transformer write-up in `book/ch1.md` shows how Google pushed 1.6T parameters with sparse routing, while `book/ch15.md` highlights GLaM’s load-balancing losses and expert gating noise to avoid straggler experts.
- **GPU/CUDA tuning angles:** Capacity-factor-aware routing, top-2 expert assignment, and expert placement heuristics to keep NVLink/NVSwitch traffic local.
- **Repo demo:**  
  1. Use `python tools/cli/benchmark_cli.py run --targets ch13:expert_parallelism --targets ch15:expert_parallelism` to compare PyTorch-profiler-guided versus production-serving MoE dispatchers.  
  2. Modify `ch15/optimized_expert_parallelism.py` to toggle top-1 vs top-2 gating and plot load variance with `common/python/metrics_extractor.py`.  
  3. Pair the dispatch experiments with the NVSwitch visualizations in `book/ch2.md` by running `ch2/nvlink_c2c_bandwidth_benchmark.py` and `ch2/optimized_memory_transfer_nvlink.cu` to observe when all-to-all hops spill beyond a single switch tray.  
  4. Inject the training-time penalties discussed in `book/ch15.md` by sweeping the capacity factor inside `ch15/disaggregated_inference.py` and logging the per-expert token counts.

### 3. Microsoft DeepSpeed + NVIDIA TensorRT-LLM: Continuous batching & paged KV caches
- **What happened:** `book/ch17.md` walks through how DeepSpeed and TensorRT-LLM adopted continuous batching, paged KV caches, and length-bucket schedulers to avoid decode underutilization. The same chapter also explains NVIDIA’s Dynamo Planner for real-time prefill/decode resource rebalancing.
- **GPU/CUDA tuning angles:** GPU stream pooling, pinned-memory KV paging, latency-aware routing, and disaggregated inference planning.
- **Repo demo:**  
  1. Replay the baseline/optimized continuous batching pair with `python tools/cli/benchmark_cli.py run --targets ch15:continuous_batching`.  
  2. Combine `ch11/optimized_streams.py` (for CUDA streams) with `ch15/optimized_kv_cache_management.py` to mirror the queueing discipline described in `book/ch17.md`.  
  3. Exercise `ch17/dynamic_routing.py` (prefill vs decode orchestrator) and feed its telemetry into `tools/analysis/analyze_expectations.py` to experiment with Dynamo-style routing policies.  
  4. When paged KV caching becomes memory-bound, switch to `ch19/dynamic_quantized_cache.py` for FP8/FP4 cache segments as described later in the book.

### 4. NVIDIA + DeepSeek-R1 + ThunderMLA: AI-assisted kernel search
- **What happened:** `book/ch20.md` recounts NVIDIA’s experiment where the DeepSeek-R1 reasoning model autogenerates a relative-position attention kernel that beats FlexAttention by up to 2.1×. `book/ch18.md` also highlights ThunderMLA from Stanford as the decode “megakernel” follow-up to FlashMLA.
- **GPU/CUDA tuning angles:** Use AI agents to mutate CUDA/Triton kernels, then verify with Nsight + unit tests; fold the winner into decode workers to collapse kernel launches and tail effects.
- **Repo demo:**  
  1. Use `ch20/ai_kernel_generator.py` as the harness for programmatically prompting DeepSeek-R1 (or another reasoning model) and plug its outputs into `ch18/flex_attention_enhanced.py` to quantify the benefit.  
  2. Compare `ch18/optimized_warp_specialization_attention.py` to the generated kernels to ensure we do not regress tensor-core occupancy.  
  3. Keep the iterations fun by timing each “boss fight” with `tools/cli/benchmark_cli.py run --targets ch18:warp_specialization_attention --profile` and using the artifacts explorer in `artifacts/<timestamp>/` as a trading-card logbook.

### 5. FlexAttention CuTe DSL: block-sparse decode masks
- **What happened:** The Colfax FlexAttention guide (“A User’s Guide to FlexAttention in FlashAttention (CuTe DSL)”) walks through Python-side `score_mod`/`mask_mod` hooks and block-sparse masks that route attention without materializing dense `[T x T]` score matrices.
- **GPU/CUDA tuning angles:** Inline relative-position bias via `score_mod`, enforce document/segment boundaries with `create_block_mask`, and rely on `torch.compile` to emit fused FlexAttention kernels instead of the eager fallback that builds dense scores.
- **Repo demo:**  
  1. Run `python tools/cli/benchmark_cli.py run --targets labs/flexattention:flex_attention --profile` to compare the eager path (`labs/flexattention/baseline_flex_attention.py`) against the compiled variant (`labs/flexattention/optimized_flex_attention.py`).  
  2. Toggle `TORCH_COMPILE_MODE=reduce-overhead` in quick runs or bump `BLOCK_SIZE`/`doc_span` to experiment with different block-sparse patterns; the lab ships the blog’s document-boundary mask plus a relative-bias `score_mod` to mirror the post exactly.  
  3. Use the harness artifacts (`artifacts/<run_id>/labs_flexattention_*`) with `tools/analysis/deep_profiling_report.py` to confirm whether the compiled path avoided dense materialization and to compare kernel counts against the eager baseline.

### 6. Modular Grace-Blackwell Matmul: CUDA 13 hardware walkthrough
- **What happened:** Modular’s four-part series ([intro](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction), [TMA](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul), [85% SOTA](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance), [breaking SOTA](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota)) explains how Blackwell/Grace-Blackwell developers stage GEMMs from a naïve baseline to a DSMEM-backed, cluster-aware pipeline.
- **GPU/CUDA tuning angles:** Warp-specialized pipelines that emulate TMA copies with `cuda::pipeline`, a real TMA path on SM100/103, `torch.compile(mode="max-autotune")` wrappers for the high-level path, and CTA clusters that share tiles via DSMEM so only one block per cluster touches HBM.
- **Repo demo:**  
  1. Explore the `labs/blackwell_matmul/` project (see its README). Build the kernels implicitly via `from labs.blackwell_matmul import baseline_blackwell_matmul`.  
  2. Drive every baseline/optimized pair via the benchmark harness so Nsight profiling artifacts land in one run folder:  
     `python tools/cli/benchmark_cli.py run --targets labs/blackwell_matmul:blackwell_matmul --targets labs/blackwell_matmul:blackwell_matmul_pseudo --targets labs/blackwell_matmul:blackwell_matmul_tma --targets labs/blackwell_matmul:blackwell_matmul_pipeline --targets labs/blackwell_matmul:blackwell_matmul_cluster --profile`.  
     The `--profile` flag keeps `nsys` + `ncu` glued to the harness, matching the Modular telemetry captures.  
  3. Use `python labs/blackwell_matmul/run_blackwell_matmul.py --variant cluster --size 4096` for the DSMEM demo; it fails fast if cluster launch is unsupported.  
  4. For deeper metrics, open the `.ncu-rep` and `.nsys-rep` files that the harness drops under `benchmark_profiles/labs/blackwell_matmul/*` instead of launching Nsight tools manually—this mirrors the Modular instrumentation flow.  
  5. Real TMA copies only run when `tma_supported()` reports true (B100/B200/B300). On GB10-or-older GPUs the TMA module fails fast; use the pseudo-TMA target to keep the lesson runnable. The DSMEM cluster path fails fast when `cudaDevAttrClusterLaunch` is 0.

## Blackwell/Grace-Blackwell Use Case Builder

Each of the following 15 scenarios came straight from the Blackwell/Grace-Blackwell playbook you highlighted. For every one we list the provenance, the GitHub repos with runnable code, and the `baseline_`/`optimized_` targets we can wire into `tools/cli/benchmark_cli.py` so the harness can collect Nsight traces, NVTX ranges, and structured metrics side-by-side.

### Use Case 1 – TMEM + async MMA + TMA triple overlap
**Essence:** Stream tiles with TMA, accumulate in TMEM with `tcgen05.mma` barriers, and overlap drains to registers while the next tiles load.
- **Sources:** [Modular Part 2](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul), [CUDA 13 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).
- **GitHub repos:** [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass), [openai/triton](https://github.com/openai/triton).
- **Baseline target:** Clone `ch10/baseline_double_buffered_pipeline.cu` into `ch10/baseline_tmem_triple_overlap.cu` so accumulators live entirely in registers and all GMEM↔SMEM movement happens via synchronous copies. Register it in the harness as `ch10:tmem_triple_overlap_baseline`.
- **Optimized target:** Use `ch10/tma_2d_pipeline_blackwell.cu` + `ch10/tcgen05_blackwell.cu` to create `ch10/optimized_tmem_triple_overlap.cu` that stages three TMA stages (load, MMA into TMEM, drain) with `cuda::pipeline` barriers. Expose it as `ch10:tmem_triple_overlap`.
- **Harness focus:** `python tools/cli/benchmark_cli.py run --targets ch10:tmem_triple_overlap_baseline --targets ch10:tmem_triple_overlap --profile --ncu-metric-set minimal` captures SM occupancy, Tensor Core duty cycle, register pressure, and GMEM/SMEM throughput in Nsight Systems while Nsight Compute tracks TMEM issue efficiency.

### Use Case 2 – Cluster TMA multicast + dual-SM MMA
**Essence:** Share tiles across CTA clusters so two SMs reuse the same DSMEM line and halve TMA traffic.
- **Sources:** [Modular Part 3](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance), [NVIDIA Blackwell architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/).
- **GitHub repos:** [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples), [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass).
- **Baseline target:** `ch10/baseline_cluster_group_no_dsmem.cu` (single-CTA tiles, no multicast) becomes `ch10/baseline_cluster_multicast.cu` inside the harness as `ch10:cluster_multicast_baseline`.
- **Optimized target:** `ch10/optimized_cluster_group.cu` and `ch10/optimized_cluster_group_single_cta.cu` already implement DSMEM-backed multicast. Wrap them inside `tools/cli/benchmark_cli.py` under `ch10:cluster_multicast`. Add knobs to sweep `clusterDim` so we can show how pairing SMs doubles arithmetic intensity.
- **Harness focus:** Compare NVLink/SMEM traffic, DSMEM hit rate, and Tensor Core utilization with `python tools/cli/benchmark_cli.py run --targets ch10:cluster_multicast_baseline --targets ch10:cluster_multicast --profile --ncu-metric-set roofline`, logging cluster residency (from CUPTI) into `artifacts/*/metrics.json`.

### Use Case 3 – Triton/PyTorch warp specialization beyond FlashAttention
**Essence:** Partition producer/consumer warps for fused attention, MoE routers, and FP4 GEMMs using Triton’s warp specialization knobs.
- **Sources:** [PyTorch Warp Specialization blog](https://pytorch.org/blog/warp-specialization/), [Ian Barber’s post](https://ianbarber.blog/2025/02/16/warp-specialization/), [OpenAI Triton on Blackwell](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/).
- **GitHub repos:** [pytorch/pytorch](https://github.com/pytorch/pytorch), [openai/triton](https://github.com/openai/triton).
- **Baseline target:** `ch9/baseline_warp_specialization_producer_consumer.py` (single buffer, no `num_consumer_groups`) stays our pedagogical `baseline_warp_specialization.py` target.
- **Optimized target:** Extend `ch10/optimized_warp_specialized_pipeline.py` (CUDA) and `ch9/optimized_warp_specialization_producer_consumer.py` (Triton) to honor Triton’s `num_consumer_groups` and `num_buffers_warp_spec` flags, exposing `ch10:warp_specialization_blackwell` that drives both kernels under `torch.compile(mode="max-autotune")`.
- **Harness focus:** Run `python tools/cli/benchmark_cli.py run --targets ch9:warp_specialization_baseline --targets ch10:warp_specialization_blackwell --profile --ncu-metric-set deep_dive` and correlate warp stall reasons, L2/TMEM throughput, and occupancy against Nsight’s warp state breakdown.

### Use Case 4 – NVFP4 end-to-end inference with TensorRT-LLM autotune
**Essence:** Quantize dense + MoE weights to NVFP4 with Model Optimizer, then autotune fused TRT-LLM kernels (paged KV, in-flight batching, NVFP4 matmuls).
- **Sources:** [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), [Blackwell MLPerf results](https://developer.nvidia.com/blog/nvidia-blackwell-platform-sets-new-llm-inference-records-in-mlperf-inference-v4-1/), [TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/overview.html), [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer).
- **GitHub repos:** [NVIDIA/TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer), [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
- **Baseline target:** Keep `ch15/baseline_inference_monolithic.py` in FP16 (no fusion) and save metrics under `artifacts/*baseline_nvfp4*/`.
- **Optimized target:** Add `ch18/optimized_nvfp4_trtllm.py` that invokes TensorRT Model Optimizer, feeds quantized checkpoints to TRT-LLM’s autotuner, and toggles paged KV + in-flight batching. Register both inside the CLI as `ch18:nvfp4_inference_baseline` and `ch18:nvfp4_inference_trtllm`.
- **Harness focus:** `python tools/cli/benchmark_cli.py run --targets ch18:nvfp4_inference_baseline --targets ch18:nvfp4_inference_trtllm --profile --ncu-metric-set roofline --repeat 3` logs TTFT, tokens/s, accuracy deltas, and NVFP4 tensor statistics so we can recreate NVIDIA’s MLPerf comparisons.
- *Note:* We reference TensorRT-LLM for provenance, but the repo implements the scheduling/paged-KV concepts natively via `ch18/optimized_moe_inference.py`, `ch18/optimized_paged_attn_vllm.py`, and the Chapter 19 NVFP4 scripts so contributors can run the workflow without proprietary runtimes.

### Use Case 5 – NVFP4 pre-training/fine-tuning with Transformer Engine
**Essence:** Use Transformer Engine’s microscaling path (MXFP8/NVFP4) to train or LoRA-tune while keeping accuracy near FP8.
- **Sources:** [Transformer Engine FP8/FP4 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html), [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).
- **GitHub repos:** [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine), [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
- **Baseline target:** `ch19/native_fp8_training.py` (FP8 activations, FP16 weights) becomes the `ch19:nvfp4_training_baseline` CLI target.
- **Optimized target:** Wire `ch19/native_fp4_quantization.py`, `ch19/token_precision_switching.py`, and Transformer Engine’s NVFP4 modules into `ch19/optimized_nvfp4_training.py`, then register `ch19:nvfp4_training_optimized`.
- **Harness focus:** Sweep sequence lengths and expert counts via `python tools/cli/benchmark_cli.py run --targets ch19:nvfp4_training_baseline --targets ch19:nvfp4_training_optimized --profile --ncu-metric-set deep_dive --kwargs \"batch_size=32\"` and capture calibration histograms + loss deltas.

### Use Case 6 – Wide Expert Parallelism on NVL72 with GroupGEMM reuse
**Essence:** Spread an MoE across all 72 GPUs in an NVLink Switch domain so each GPU hosts fewer experts and reuses GroupGEMM weights more often.
- **Sources:** [Scaling Large MoE Models with Wide EP on NVL72](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/), [GB200 NVL72 platform](https://www.nvidia.com/en-us/data-center/gb200-nvl72/).
- **GitHub repos:** [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
- **Baseline target:** `ch15/baseline_expert_parallelism.py` (EP confined to four GPUs) feeds `ch15:expert_parallelism_baseline`.
- **Optimized target:** Extend `ch15/optimized_expert_parallelism.py` with NVL72 topology awareness + GroupGEMM reuse (think `--moe-widening-factor`), save it as `ch15/optimized_wide_ep.py`, and expose it via `ch15:wide_ep`.
- **Harness focus:** Launch `python tools/cli/benchmark_cli.py run --targets ch15:expert_parallelism_baseline --targets ch15:wide_ep --profile --ncu-metric-set roofline` so we can report expert load variance, NVLink bandwidth, and per-GPU memory footprint before/after Wide-EP.

### Use Case 7 – Prefill/Decode disaggregation + Expert Parallelism for serving
**Essence:** Split compute-heavy prefill from KV-heavy decode pools, assigning experts/KV caches to decode GPUs while prefill GPUs focus on large matmuls.
- **Sources:** [SGLang + NVIDIA disaggregated inference deep dive](https://lmsys.org/blog/2025-10-14-sa-inference-max/), [vLLM docs](https://docs.vllm.ai/en/stable/index.html).
- **GitHub repos:** [sgl-project/sglang](https://github.com/sgl-project/sglang), [vllm-project/vllm](https://github.com/vllm-project/vllm).
- **Baseline target:** `ch15/baseline_inference_monolithic.py` (single pool) drives `ch15:prefill_decode_baseline`.
- **Optimized target:** Pair `ch17/dynamic_routing.py` with `ch18/run_vllm_decoder.py` and `ch15/optimized_continuous_batching.py` to create `ch17/optimized_prefill_decode_disagg.py`, adding harness entries `ch17:prefill_decode_disagg` and `ch17:prefill_decode_prefetch`.
- **Harness focus:** Use `python tools/cli/benchmark_cli.py run --targets ch15:prefill_decode_baseline --targets ch17:prefill_decode_disagg --profile --ncu-metric-set minimal --kwargs \"prefill_gpus=2 decode_gpus=2\"` to track TTFT, decode tok/s, queue depth, and EP balance.

### Use Case 8 – KV cache offload to Grace over NVLink-C2C (UVM + prefetch)
**Essence:** Keep hot KV stripes in HBM while UVM-prefetching cold ranges to Grace LPDDR via 900 GB/s NVLink-C2C.
- **Sources:** [Grace-Blackwell KV offload blog](https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/), [Blackwell architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/).
- **GitHub repos:** [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vllm-project/vllm](https://github.com/vllm-project/vllm).
- **Baseline target:** `ch19/baseline_disaggregated_memory.py` (HBM-only KV cache) becomes `ch19:kv_offload_baseline`.
- **Optimized target:** Combine `ch19/optimized_kv_prefetch_overlap.cu`, `ch19/dynamic_quantized_cache.py`, and managed-memory hints (`cudaMemAdvise`, `cudaMemPrefetchAsync`) into `ch19/optimized_kv_offload_grace.py`, exposing `ch19:kv_offload_grace`.
- **Harness focus:** `python tools/cli/benchmark_cli.py run --targets ch19:kv_offload_baseline --targets ch19:kv_offload_grace --profile --ncu-metric-set minimal --kwargs \"context=64000\"` logs UVM page residency, NVLink-C2C bandwidth, and decode latency so we can show how Grace absorbs long contexts.

### Use Case 9 – NCCL NVLink-SHARP (NVLS) + copy-engine collectives
**Essence:** Offload AllReduce/AllGather math into NVSwitch SHARP engines and drive transfers with copy engines so SMs stay busy with compute.
- **Sources:** [NCCL 2.28 env guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html), [GB200 NVL72 platform](https://www.nvidia.com/en-us/data-center/gb200-nvl72/).
- **GitHub repos:** [NVIDIA/nccl](https://github.com/NVIDIA/nccl), [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
- **Baseline target:** `ch4/baseline_nccl.py` runs standard SM-driven collectives (`ch4:nccl_baseline`).
- **Optimized target:** Create `ch4/optimized_nvls_collectives.py` that sets `NCCL_NVLS_ENABLE=1`, uses NCCL’s device API for copy-engine launches, and fuses collectives around MoE FFNs. Register it as `ch4:nccl_nvls`.
- **Harness focus:** Execute `NCCL_NVLS_ENABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 python tools/cli/benchmark_cli.py run --targets ch4:nccl_baseline --targets ch4:nccl_nvls --profile --ncu-metric-set minimal` to record step time, CE utilization, and SHARP offload percentages.

### Use Case 10 – Overlap MoE all-to-all with compute (tile-wise fusion)
**Essence:** Break the MoE dispatcher into tiles and fuse each tile’s GEMM + all-to-all segment so comms overlap with expert compute.
- **Sources:** [Comet fine-grained overlap paper](https://arxiv.org/html/2502.19811v3), [Megatron-Core MoE docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html).
- **GitHub repos:** [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed).
- **Baseline target:** `ch4/baseline_no_overlap.py` + `ch15/baseline_expert_parallelism.py` become the `ch15:moe_overlap_baseline` harness target (blocking NCCL all-to-all).
- **Optimized target:** Use `ch4/optimized_no_overlap.py`’s stream-ordered rewriter and add a fused dispatcher kernel (`ch15/optimized_moe_shared_expert_overlap.py`) that imitates Megatron-Core’s `--moe-shared-expert-overlap`. Register it under `ch15:moe_overlap`.
- **Harness focus:** `CUDA_DEVICE_MAX_CONNECTIONS=1 python tools/cli/benchmark_cli.py run --targets ch15:moe_overlap_baseline --targets ch15:moe_overlap --profile --ncu-metric-set deep_dive` gives overlap % vs expectation, NCCL timeline gaps, and per-expert latency.

### Use Case 11 – CUDA Graphs with conditional nodes for router-dependent paths
**Essence:** Capture an entire inference iteration and let router choices stay inside the graph via CUDA 12.8 conditional nodes.
- **Sources:** [Dynamic control flow in CUDA Graphs](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/), [TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/overview.html).
- **GitHub repos:** [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples), [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
- **Baseline target:** `ch12/baseline_cuda_graphs.py` captures a static sequence per batch and serves as `ch12:cuda_graphs_baseline`.
- **Optimized target:** `ch12/optimized_cuda_graphs_conditional.cu` + `ch17/dynamic_routing.py` become `ch12/optimized_cuda_graphs_router.cu`, capturing gating logic inside the graph and binding stream-ordered allocators. Hook it up as `ch12:cuda_graphs_router`.
- **Harness focus:** Run `python tools/cli/benchmark_cli.py run --targets ch12:cuda_graphs_baseline --targets ch12:cuda_graphs_router --profile --ncu-metric-set minimal` to show launch-overhead removal, router latency, and NVTX-based conditional node firing rates.

### Use Case 12 – GB200/GB300 rack-scale sharding with NVLink domains
**Essence:** Treat an NVL72 (GB200 today, GB300 tomorrow) as a single logical GPU before crossing racks so reductions stay inside the 130 TB/s fabric.
- **Sources:** [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/), [GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/), [Blackwell architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/).
- **GitHub repos:** [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed).
- **Baseline target:** `ch4/multi_node_blackwell.py` + `ch4/gb200_grace_numa_optimization.py` provide `ch4:nvl72_baseline` (TP+DP on a partial rack, Ethernet between racks).
- **Optimized target:** `ch4/training_8xb200_pipeline.py` and `ch4/nccl_blackwell_config.py` already capture the NVL72 launch plan; extend them with rack-aware NCCL trees + SHARP reductions as `ch4/optimized_nvl72_sharding.py`, registered as `ch4:nvl72_optimized`.
- **Harness focus:** `python tools/cli/benchmark_cli.py run --targets ch4:nvl72_baseline --targets ch4:nvl72_optimized --profile --ncu-metric-set roofline --kwargs \"num_gpus=72\"` surfaces per-hop latency, SHARP utilization, and cross-rack traffic so we can justify the NVLink-domain-first strategy.

### Use Case 13 – On-GPU decompression in the data path
**Essence:** Feed compressed data shards over GDS, decompress them with Blackwell’s hardware block before kernels consume them, and keep the CPU out of the loop.
- **Sources:** [Blackwell B200 datasheet](https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf), [CUDA 13 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).
- **GitHub repos:** [NVIDIA/nvcomp](https://github.com/NVIDIA/nvcomp), [NVIDIA/cufile-samples](https://github.com/NVIDIA/cufile-samples).
- **Baseline target:** Extend `ch5/baseline_storage_cpu.py` into `ch5/baseline_cpu_decompression.py` (CPU-bound gzip/snappy) and register it as `ch5:gpu_input_baseline`.
- **Optimized target:** Build `ch5/optimized_gpu_decompression.py` that uses GPUDirect Storage + `nvCOMP::decompress_async`, hooking directly into MoE prefills. Expose it via `ch5:gpu_input_nvcomp`.
- **Harness focus:** `python tools/cli/benchmark_cli.py run --targets ch5:gpu_input_baseline --targets ch5:gpu_input_nvcomp --profile --ncu-metric-set roofline --kwargs \"compression=lz4\"` measures ingest throughput, GPU decompressor occupancy, and CPU utilization.

### Use Case 14 – PyTorch DTensor for TP/PP/DP/EP/SP meshes
**Essence:** Use DTensor to define a 5-D device mesh and keep a single code path while varying tensor/sequence/data/expert parallelism.
- **Sources:** [PyTorch DTensor docs](https://docs.pytorch.org/docs/stable/distributed.tensor.html), [OpenAI Triton on Blackwell](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/).
- **GitHub repos:** [pytorch/pytorch](https://github.com/pytorch/pytorch), [pytorch/torchtitan](https://github.com/pytorch/torchtitan).
- **Baseline target:** `ch13/baseline_data_parallelism.py` is our DP-only `ch13:dtensor_baseline`.
- **Optimized target:** Author `ch13/optimized_dtensor_mesh.py` that configures DTensor meshes, integrates EP routing hooks, and supports `torch.compile`. Register it as `ch13:dtensor_mesh`.
- **Harness focus:** `python tools/cli/benchmark_cli.py run --targets ch13:dtensor_baseline --targets ch13:dtensor_mesh --profile --ncu-metric-set roofline --kwargs \"mesh=tp2_pp2_dp2_ep4_sp2\"` logs memory balance, reshaping overhead, and collective latencies per dimension.

### Use Case 15 – vLLM/SGLang scheduling: paged KV + chunked prefill + speculative decode
**Essence:** Combine paged KV caches, continuous batching, chunked prefill, and speculative decoding so decode GPUs stay busy under multi-tenant load.
- **Sources:** [vLLM docs](https://docs.vllm.ai/en/stable/index.html), [SGLang + NVIDIA write-up](https://lmsys.org/blog/2025-10-14-sa-inference-max/), [TensorRT-LLM overview](https://nvidia.github.io/TensorRT-LLM/overview.html).
- **GitHub repos:** [vllm-project/vllm](https://github.com/vllm-project/vllm), [sgl-project/sglang](https://github.com/sgl-project/sglang).
- **Baseline target:** `ch18/baseline_streams.py` + `ch15/baseline_continuous_batching.py` serve as `ch18:scheduling_baseline`.
- **Optimized target:** `ch18/run_vllm_decoder.py`, `ch17/dynamic_routing.py`, and `ch19/dynamic_quantized_cache.py` compose the optimized stack—capture it as `ch18:scheduling_vllm_sglang` with knobs for paged attention, chunked prefill, and speculative helpers.
- **Harness focus:** Run `python tools/cli/benchmark_cli.py run --targets ch18:scheduling_baseline --targets ch18:scheduling_vllm_sglang --profile --ncu-metric-set minimal --kwargs \"arrival_poisson=0.8\"` to report throughput, queue depth, page hit rate, and speculation accept ratios.

### Use Case 16 – vLLM CUDA spelunking: PagedAttention v2, fused RMSNorm/quant, and chunked prefill capture
**Essence:** Transplant the coolest CUDA-level optimizations from `vllm/csrc/` into our harness so teams can contrast baseline_ vs optimized_ kernels with Nsight and our own metrics pipeline.
- **Sources:** [PagedAttention v2 deep dive](https://blog.vllm.ai/2024/05/28/pagedattention-v2.html), [vLLM fused RMSNorm kernel (`csrc/ops/fused_add_rmsnorm.cu`)](https://github.com/vllm-project/vllm/blob/main/csrc/ops/fused_add_rmsnorm.cu), [vLLM continuous batching + CUDA Graph notes](https://docs.vllm.ai/en/stable/serving/continuous_batching.html#cuda-graphs).
- **GitHub repos:** [vllm-project/vllm](https://github.com/vllm-project/vllm) (paged attention, fused ops), [sgl-project/sglang](https://github.com/sgl-project/sglang) (spec decode helpers), [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (graph capture recipes).
- **Baseline target:** `ch18/baseline_paged_attn.py` lands in the harness as `ch18:paged_attn`. It sticks to PyTorch’s stock FP16 `scaled_dot_product_attention`, dense KV tensors, and a single CUDA stream while decorating `prefill`, `router`, `decode`, and `moe_ep` NVTX ranges so Nsight Systems immediately shows the idle gaps.
- **Optimized target:** `ch18/optimized_paged_attn_vllm.py` becomes `ch18:paged_attn_vllm` and ports three tricks straight out of vLLM:  
  1. **PagedAttention v2 tiles** – Synthetic shared-memory tiles stream KV blocks through a scoreboard bitmap so we can reason about residency, block size, and chunk length knobs.  
  2. **Fused RMSNorm + quantized matvec** – A lightweight fused kernel applies RMSNorm, per-token FP8 quantization, and the expert matvec in one launch to mirror vLLM’s fused ops.  
  3. **Chunked prefill CUDA Graph** – The prefill path captures a canonical chunk via `torch.cuda.CUDAGraph` and replays it while Python stitches the output into the paged KV cache, mimicking vLLM’s chunked prefill capture.
- **Harness focus:**  
  - `python tools/cli/benchmark_cli.py run --targets ch18:paged_attn --targets ch18:paged_attn_vllm --profile` records both traces. Override problem sizes with environment variables such as `PAGED_ATTN_CONTEXT=32000 PAGED_ATTN_CHUNK=2048 PAGED_ATTN_BLOCK_SIZE=128` before invoking the CLI.  
  - Feed the generated artifacts into `python tools/analysis/deep_profiling_report.py artifacts/<ts>/ch18_paged_attn_* artifacts/<ts>/ch18_paged_attn_vllm_*` to auto-compare SM occupancy, DRAM/L2 throughput, and NVLink overlap.  
  - Use `python common/python/metrics_extractor.py artifacts/<ts>` to emit TTFT, TPOT, page-hit rate, and fused-RMSNorm timing deltas so CI can assert the speedups by diffing `artifacts/*baseline*/metrics.json` vs `artifacts/*optimized*/metrics.json`.

### Use Case 17 – Piece-graph regional compilation vs monolithic capture
**Essence:** Show why naïvely recapturing a full CUDA graph each time (“piece graphs”) wastes time on module loads and allocations, while caching smaller regional graphs eliminates that overhead.
- **Sources:** [CUDA Graph best practices](https://developer.nvidia.com/blog/cuda-graphs/), [torch.compile regional compilation knobs](https://pytorch.org/docs/stable/generated/torch.compile.html), [TorchDynamo config flags](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/config.py).
- **Baseline target:** `ch16/baseline_piece_graphs.py` (CLI label `ch16:piece_graphs_baseline`) captures a monolithic CUDA graph per sequence bucket. Nsight Systems (`benchmark_profiles/ch16/piece_graphs_baseline_manual.nsys-rep`) shows a single NVTX span `:piece_graph_monolith` (~0.26 ms) but 45 % of CUDA API time tied up in `cuLibraryLoadData` and 16.8 % in `cudaMalloc` while the graph is re-instantiated.
- **Optimized target:** `ch16/optimized_piece_graphs.py` (`ch16:piece_graphs_optimized`) splits the transformer stub into head/tail regions, captures two smaller graphs (`:piece_graph_head`, `:piece_graph_tail`), and simply copies inputs between them. Profiling (`benchmark_profiles/ch16/piece_graphs_optimized_manual.nsys-rep`) shows balanced NVTX spans (~0.16 ms / 0.13 ms), 4 572 lightweight `cudaLaunchKernel` calls, and graph-instantiation overhead dropping to 3.8 % of CUDA API time.
- **Harness focus:** Run `python tools/cli/benchmark_cli.py run --targets ch16:piece_graphs --profile --iterations 2 --warmup 0 --timeout-multiplier 4 --suite-timeout 900` so Nsight artifacts finish under the enlarged per-stage timeouts (each benchmark now allows 240 s for setup/measurement). For single-iteration traces, use `PIECEGRAPH_PROFILE_ONCE=1 nsys profile --force-overwrite=true -o benchmark_profiles/ch16/piece_graphs_<baseline|optimized>_manual python ch16/<target>.py`.
- **Nsight + deep profiling:** Convert the `.nsys-rep` files with `nsys stats benchmark_profiles/ch16/piece_graphs_<...>.nsys-rep` to inspect NVTX/CUDA summaries, then run `python tools/analysis/deep_profiling_report.py --ncu-csv benchmark_profiles/ch16/piece_graphs_baseline_manual.csv --ncu-csv benchmark_profiles/ch16/piece_graphs_optimized_manual.csv --print-markdown > artifacts/20251114_091043/reports/piece_graphs_profile.md` to capture the GPU-roofline diff.

---

## Incremental Build Series (a.k.a. the MoE Gauntlet)

1. **Act I — Router Rumble (Chapters 1–5):**  
   - Profile the dense MoE baseline with `ch1/baseline_performance.py` + NVTX, then swap to the sparse dispatcher from `ch1/optimized_context_parallelism.py`.  
   - Apply the system setup scripts in `ch3/system_tuning.sh` (NUMA, IRQ, governor) before benchmarking DualPipe-style parallelism.  
   - Feed tokens faster by configuring GDS/3FS-inspired data paths using `ch5/baseline_storage_cpu.py` → `ch5/optimized_storage_cpu.py` and logging throughput deltas just like Table 5-1 in `book/ch5.md`.

2. **Act II — Expert Gauntlet (Chapters 6–10):**  
   - Teach newcomers CUDA with `ch6/baseline_coalescing.py`, `ch6/optimized_coalescing.cu`, and `ch6/baseline_triton.py` so they can reason about router microkernels.  
   - Explore tiling, async copies, and Tensor Memory Accelerator ideas using `ch7/baseline_memory_access.py`, `ch8/optimized_double_buffering_pipelined.cu`, and `ch10/optimized_matmul.py`.  
   - Use the results to refactor the MoE router into a fused Triton kernel via `ch14/triton_examples.py` before you move back up the stack.

3. **Act III — Serving Olympics (Chapters 11–15):**  
   - Combine `ch11/optimized_streams.py`, `ch12/optimized_cuda_graphs.py`, and `ch13/optimized_expert_parallelism.py` to capture concurrency wins.  
   - Deploy the disaggregated inference stack inside `ch15/disaggregated_inference.py`, experimenting with capacity-factor sweeps inspired by Google GLaM.  
   - Track everything with `common/python/benchmark_harness.py` to maintain Proof-of-Benefit tables in `ch15/expectations_gb10.json`.

4. **Act IV — Frontier Tricks (Chapters 16–20):**  
   - Layer speculative decoding + multi-agent routing (`ch16/symmetric_memory_inference.py`, `ch17/dynamic_routing.py`).  
   - Swap in FlashMLA/ThunderMLA decode kernels (`ch18/flashmla_kernel.cu`, `ch18/run_vllm_decoder.py`) and toggle FP8/FP4 cache paths from `ch19/native_fp8_training.py` + `ch19/dynamic_precision_switching.py`.  
   - Finish by auto-generating kernels via `ch20/ai_kernel_generator.py` and capturing regression tests in `ch20/optimized_end_to_end_bandwidth.py`.
   - For hands-on CUDA, run `labs/moe_cuda/baseline_decode_kernel.py` → `labs/moe_cuda/optimized_decode_kernel.py`. The optimized kernel requires Hopper/Blackwell GPUs (CUDA 13+ TMA); the harness will mark it `SKIPPED` on older parts.

Each act is self-contained but chained so you can hand teammates a “quest log” that gradually pulls in all `book/ch*.md` concepts.

## Perplexity TransferEngine × pplx-garden (2025)

Harness users asked how to recreate the new Perplexity AI release—[RDMA Point-to-Point Communication for LLM Systems](https://arxiv.org/abs/2510.27656) and the companion [pplx-garden](https://github.com/perplexityai/pplx-garden)—inside this repo. Below is the distilled context plus a reproducibility recipe that mirrors the complexity of the original effort while remaining native to `tools/cli/benchmark_cli.py`.

### Extracted highlights from the resources
- **Portable RDMA primitives:** TransferEngine exposes two-sided `SEND/RECV`, paged one-sided `WRITE` (for KvCache moves), scatter/barrier helpers, and a 32-bit `IMM_COUNTER` completion primitive that hides transport ordering. It aggregates 1–4 NICs per GPU, balances ConnectX-7 and AWS EFA quirks, and lets host proxies poll via GDRCopy so GPUs stay SM-free while RDMA runs. Peak throughput hits 400 Gbps on both NIC families.
- **P2P MoE dispatch/combine kernel:** The pplx-garden implementation splits send/recv halves, pipelines NVLink for intra-node copies, keeps the GPU SMs fully utilized while RDMA DMA engines move data, and supports CUDA Graph capture plus micro-batching. Reported decode latency (128 tokens, EP64) is 266.7 µs on EFA and 187.5 µs on CX7; prefill (4096 tokens) ranges 2.7–9.8 ms depending on expert count.
- **Three production proof points from the paper:** (1) KvCache transfers that let decode workers dynamically borrow/cache context from prefill nodes, (2) RLHF/asynchronous fine-tuning weight pushes in ≈1.3 s for Kimi-K2/DeepSeek-V3/Qwen3-scale checkpoints, and (3) MoE dispatch/combine that beats DeepEP latency on CX7 while making EFA viable.
- **Repo scaffolding you can mirror:** `pplx-garden/fabric-lib` (TransferEngine core), `p2p-all-to-all` (MoE kernels), `python/pplx_garden` + `python-ext` (bindings), and `rust/` utilities. The README walks through `cargo build --release --bin fabric-debug`, `python3 -m benchmarks.bench_all_to_all --world-size ... --nets-per-gpu ... --nvlink=8`, and publishes the decode/prefill latency tables shown above.

### Complexity map for the harness
| Building block | Resource cues | Harness hook |
|----------------|---------------|--------------|
| Portable RDMA TransferEngine | Section 3 of the paper describes the `TransferEngine` trait, `IMM_COUNTER` signaling, multi-NIC domain groups, and host-proxy polling via GDRCopy. | Use `ch3/system_tuning.sh` as the RDMA tuning checklist, `ch5/gpudirect_storage_example.py` + `ch5/optimized_storage_cpu.py` to rehearse paged writes, and `python tools/cli/benchmark_cli.py run --targets ch11:optimized_streams --profile` to capture how our CUDA streams overlap DMA while the CPU orchestrates transfers. |
| Disaggregated KvCache streaming | Section 4 (“KvCache transfer”) shows how prefill nodes source context via paged writes and immediate notifications. pplx-garden’s `benchmarks/bench_all_to_all` benchmarks the same path. | Run `python tools/cli/benchmark_cli.py run --targets ch15:disaggregated_inference --targets ch15:optimized_kv_cache_management` and feed the resulting artifacts into `common/python/metrics_extractor.py` to track per-hop latency before/after RDMA tuning. |
| MoE dispatch/combine micro-batching | Figures 6–12 plus `p2p-all-to-all/` outline the split kernels, NVLink prefills, scatter limits, and decode/prefill latency curves. | Chain `python tools/cli/benchmark_cli.py run --targets ch15:expert_parallelism` with `ch18/flashmla_kernel.cu` (compile via `make -C ch18 flashmla_kernel`) and `ch11:optimized_streams` to rehearse the same dual-path dispatch/combine scheduling. |
| Asynchronous RL / weight pushes | Section 5 demonstrates 1.3 s host-to-host pushes by chunking writes per inference node and tying completion to immutable counters. pplx-garden reuses the same TransferEngine code path. | Pair `python tools/cli/benchmark_cli.py run --targets ch17:dynamic_routing` with `ch16/symmetric_memory_inference.py` (for weight swaps) and `ch20/ai_kernel_generator.py` (for regression kernels) to stress-test background updates while decode threads stay hot. |

### Step-by-step harness walkthrough
1. **Calibrate the RDMA plumbing.**  
   - Run `python tools/cli/benchmark_cli.py run --targets ch5:optimized_storage_cpu --targets ch11:optimized_streams --profile` to ensure paged copy paths + stream overlap hit expected throughput.  
   - Mirror pplx-garden’s host proxy by launching `ch5/gpudirect_storage_example.py` with Nsight Systems—verify GPU kernels stay SM-free while DMA progresses, as emphasized in the paper.
2. **Validate all-to-all throughput vs pplx-garden.**  
   - Clone `pplx-garden` under `vendor/` (read-only) and follow its README: `cargo build --release --bin fabric-debug`, then `python3 -m benchmarks.bench_all_to_all --world-size <nodes×8> --nets-per-gpu 2 --init-method=tcp://$MASTER_IP:29500 --node-rank=$NODE_RANK --nvlink=8`.  
   - Record the decode/prefill latencies from the table and store them next to our harness artifacts (e.g., `artifacts/<ts>/rdma_transferengine.json`) so later comparisons line up with the published 187–5334 µs numbers.
3. **Map the MoE dispatch/combine pipeline.**  
   - Execute `python tools/cli/benchmark_cli.py run --targets ch15:expert_parallelism --targets ch18:flexdecoding` to benchmark our current gating + decode kernels.  
   - Layer in NVLink-style speculative fills by toggling the shared/private buffer sizes inside `ch15/optimized_expert_parallelism.py` (search for `capacity_factor`) and re-run with `--profile` to check NVLink bandwidth counters, mimicking Figure 7’s behavior.
4. **Exercise disaggregated inference + weight updates.**  
   - Launch `python tools/cli/benchmark_cli.py run --targets ch15:disaggregated_inference --targets ch17:dynamic_routing --targets ch19:dynamic_quantized_cache` to stress paged KV caches, Dynamo-style schedulers, and FP8/FP4 cache compression simultaneously.  
   - While those jobs run, spin up a lightweight controller that mimics the paper’s asynchronous RL pushes: `torchrun --nproc_per_node=8 ch16/symmetric_memory_inference.py --demo multi` hot-swaps weight snapshots while decode GPUs stay busy, with background kernels generated via `ch20/ai_kernel_generator.py` standing in for TransferEngine regression tests.  
   - Capture telemetry using `tools/analysis/analyze_expectations.py artifacts/<ts>` so your graphs align with the pplx-garden throughput tables.

The combination above gives harness users a faithful mental model of how TransferEngine’s multi-NIC orchestration, pplx-garden’s split MoE kernels, and our own `book/ch*.md` lessons fit together without leaving this repo.

---

## Chapter-by-Chapter tie-in

| Chapter | Book focus | MoE tuning hook | Repo starting point |
|---------|------------|-----------------|---------------------|
| ch1 | Performance basics, profiling, roofline (per README table and DeepSeek story) | Profile DualPipe vs dense MoE and reproduce Figure 1-4 | `ch1/baseline_performance.py`, `ch1/optimized_context_parallelism.py` |
| ch2 | GPU architecture, NVLink/NVSwitch | Map expert placement to switch trays like in `book/ch2.md` | `ch2/nvlink_c2c_bandwidth_benchmark.py`, `ch2/optimized_memory_transfer_nvlink.cu` |
| ch3 | System tuning (NUMA, Docker, K8s) | Pin Decode GPUs to NUMA nodes + MIG slices for MoE pods | `ch3/system_tuning.sh`, `ch3/baseline_moe.py`, `ch3/optimized_moe.py` |
| ch4 | Multi-GPU tensor/pipeline parallelism | Recreate DualPipe overlap + NCCL scheduling | `ch4/optimized_no_overlap.py`, `ch4/nvshmem_pipeline_parallel.py` |
| ch5 | Storage / IO (GDS, DeepSeek 3FS) | Ensure MoE sharded checkpoints stream via RDMA | `ch5/baseline_storage_cpu.py`, `ch5/optimized_storage_cpu.py` |
| ch6 | CUDA fundamentals | Teach router kernels, inline PTX safety | `ch6/baseline_coalescing.py`, `ch6/optimized_triton.py` |
| ch7 | Memory access patterns | Tile token shards before dispatch | `ch7/baseline_memory_access.py`, `ch7/optimized_memory_access.py` |
| ch8 | Occupancy, ILP | Balance expert kernels across SMs, mimic DeepEP’s overlap | `ch8/optimized_double_buffering_pipelined.cu`, `ch8/optimized_hbm_vectorized.cu` |
| ch9 | Arithmetic intensity, inline PTX (DeepEP) | Toggle `ld.global.nc.l1::no_allocate.l2::256b` loads in MoE all-to-all buffers | `ch9/optimized_cutlass_gemm.py`, `ch9/inline_ptx_example.cu` |
| ch10 | Tensor cores, warp specialization | Use TMA and WMMA for expert FFNs | `ch10/optimized_matmul.py`, `ch10/optimized_warp_specialized_pipeline.cu` |
| ch11 | CUDA streams and concurrency | Overlap router compute + all-to-all copies | `ch11/optimized_streams.py` |
| ch12 | CUDA Graphs | Capture gating + expert kernels as conditional graphs | `ch12/optimized_cuda_graphs.py` |
| ch13 | PyTorch profiling, allocators | Inspect fragmentation + capture NVTX lanes for MoE experts | `ch13/baseline_expert_parallelism.py`, `ch13/custom_allocator.py` |
| ch14 | `torch.compile` + OpenAI Triton | Generate fused top-k dispatch kernels | `ch14/triton_examples.py` |
| ch15 | Disaggregated inference | Run the full MoE serving stack + capacity-factor sweeps | `ch15/disaggregated_inference.py`, `ch15/optimized_kv_cache_management.py` |
| ch16 | Production inference tricks | Add speculative decoding + continuous batching to MoE | `ch16/symmetric_memory_inference.py`, `ch16/inference_serving_8xb200.py` |
| ch17 | Dynamic routing / scheduler | Implement Dynamo Planner-style routing logic | `ch17/dynamic_routing.py`, `ch17/optimized_routing_static.py` |
| ch18 | Flash/Flex/MLA | Adopt FlashMLA/ThunderMLA for decode | `ch18/flashmla_kernel.cu`, `ch18/flexdecoding.py` |
| ch19 | Low-precision training/inference | Switch KV caches and experts to FP8/FP4 | `ch19/dynamic_quantized_cache.py`, `ch19/native_fp8_training.py` |
| ch20 | AI-assisted optimization | Integrate DeepSeek-R1 kernel search to keep MoE stack evergreen | `ch20/ai_kernel_generator.py`, `ch20/optimized_end_to_end_bandwidth.py` |

Use the table as a checklist when crafting internal demos: every row corresponds to at least one concrete file that can be run under `tools/cli/benchmark_cli.py`.

---

## Suggested Flow for a Demo Session

1. **Kickoff (15 min):** Recount the DeepSeek story from `book/ch1.md` and show the dual baseline/optimized metrics from `ch15:expert_parallelism`.  
2. **Workshop blocks (4 × 30 min):** Each block maps to one Act above. Hand out “boss fight” cards describing which chapter pair to run and which metric to collect (throughput %, latency ms/token, NVLink Gb/s, etc.).  
3. **Show-and-tell (20 min):** Review artifacts stored under `artifacts/<timestamp>/ch15_expert_parallelism/results/` and tie the wins back to the relevant book chapter sections.  
4. **Stretch goal:** Stream a small DeepSeek-R1 reasoning model locally and let participants try to beat the ThunderMLA baseline by editing prompts in `ch20/ai_kernel_generator.py`. Celebrate any valid kernel by checking it into `artifacts/.../ai_kernel_generator/`.

---

## Why this stays fun
- Every stage references a public frontier lab so you can frame the exercise as “Recreate DeepSeek/Google/Microsoft/NVIDIA’s trick on our own hardware.”  
- The build-up mirrors the structure of `book/ch*.md`, so teams naturally review each chapter while tinkering with runnable code.  
- Because every recommendation maps to an existing script, you can hand newcomers reproducible commands instead of abstract theory.  
- Artifacts double as a progress log—treat them like collectible cards tracking speedups, NVLink utilization, and power draw for each MoE stunt.

---

## GPT-OSS-20B MoE Baseline ↔ Optimized Plan

You can turn the above inspirations into a concrete “baseline_ vs optimized_” story by treating `gpt-oss-20b-moe` as the reference model and staging two contrasting inference engines. The baseline stays intentionally simple (HuggingFace-style eager generation) while the optimized stack mirrors vLLM’s paged attention, NVIDIA Dynamo’s prefill/decode planner, and SGLang’s speculative decode loop.

### Model + hardware envelope
- **Model shape:** 20B visible parameters, 32–64 experts with top-2 gating, FFN dim 16 384, expert capacity factor 1.25, rotary attention, context 32 k. Routing two experts/token keeps the activation budget near 4.5 B parameters.
- **Cluster:** 4× H100 SXM (80 GB) with NVLink4 plus dual EPYC host sockets. Tests also collapse to a single 80 GB GPU by trimming the batch.
- **Metrics:** throughput (tokens/s), TPOT (tokens processed per observation window), TTFT (ms), p95 latency, steady-state memory footprint, NVLink bandwidth, SM occupancy, router imbalance.

### Baseline vs optimized stack

| Component | `baseline_moe_inference.py` (pedagogical baseline_) | `optimized_moe_inference.py` (vLLM + Dynamo + SGLang inspired) |
| --- | --- | --- |
| Runtime core | PyTorch eager `generate()`; sequential prefill/ decode; experts colocated on two GPUs. | vLLM V1 executor (`ch18/run_vllm_decoder.py`) on PyTorch 2.9 baseline with async event loop, request queues, and `FULL_AND_PIECEWISE` CUDA Graph capture as the default graph mode. |
| Batching | Static batch at arrival (≤8 reqs), first-in-first-out scheduling, no prompt prefix consolidation. | Continuous batching with load-aware reordering. Adopt SGLang’s prefix tree to merge shared prompts and replay tokens once. |
| Router | Top-1 gating, deterministic tie-breaks, no aux loss → straggler experts common. | Top-2 gating with capacity-factor penalties (`ch15/optimized_expert_parallelism.py`); enforce per-NVSwitch affinity to minimize cross-switch hops. |
| Prefill vs decode | Single CUDA stream; decode blocked until prefill ends; tensor parallel only. | NVIDIA Dynamo-style disaggregation: 2 GPUs run prefill (tensor/sequence parallel), 2 GPUs run decode (token parallel). Controlled by `ch17/dynamic_routing.py`. |
| KV cache | Dense FP16 tensors per request, freed at completion; no paging or sharing. | vLLM paged KV cache (4 k-token blocks) + host spill; FlashAttention2 prefill + FlashMLA decode for cache-friendly kernels. |
| Speculative decoding | None; 1 token/iteration. | SGLang speculative draft with a 7B helper or reduced expert subset; accept tokens opportunistically to cut TTFT and boost TPOT. |
| Expert compute | Experts pinned to GPU0/1; NCCL all-to-all blocking; no overlap with compute. | Expert-parallel groups across all GPUs; DualPipe overlap (`ch4/optimized_no_overlap.py`) and DeepEP-style PTX loads (`ch9`) keep copies off critical path. |
| Communication | Default NCCL priorities; no pipelining with routing. | DualPipe-style overlap and priority-tuned NVLink lanes; optional NVSHMEM path for dispatcher buffers. |
| Memory tactics | FP16 weights/activations; KV cache consumes ~55 GB at 32 k context; allocator fragmentation unchecked. | FP8 experts via TransformerEngine, KV cache compressed with `ch19/dynamic_quantized_cache.py` (FP8/FP4 mix), allocator tuned per `ch13/custom_allocator.py`. PyTorch 2.9 is the tested serving stack for this graph mode until 2.10 reaches parity. |
| Observability | PyTorch profiler trace only. | NVTX ranges for each stage, Nsight Systems + Nsight Compute captures, `tools/analysis/analyze_expectations.py` to track routing skew + TPOT in one JSON artifact. |

### Benchmark harness recipe
1. **Baseline target:** `ch15/baseline_moe_inference.py` already captures the HuggingFace-style stack. Run it with `python tools/cli/benchmark_cli.py run --targets ch15:moe_inference --profile` to log TTFT/TPOT/NVLink deltas.
2. **Optimized target:** `ch18/optimized_moe_inference.py` layers CUDA Graph-captured prefill, paged-style KV chunks, speculative decoding, and `torch.compile`d routers inspired by vLLM, NVIDIA Dynamo, and SGLang. Execute it with `python tools/cli/benchmark_cli.py run --targets ch18:moe_inference --profile` to measure the improved overlap.
3. **CLI glue:** Run the two targets back-to-back (e.g., `--targets ch15:moe_inference --targets ch18:moe_inference`) so the harness emits comparable artifacts while keeping the optimized implementation in the high-level (Chapter 18) playbook.

### Measurement + tuning workflow
1. **TTFT focus:** Fire 2–16 token prompts and sweep batch size. Expect optimized_ to cut TTFT 35–45% via speculative decoding + overlap. Plot TTFT vs batch in `tools/notebooks/plot_moe_ttft.ipynb`.
2. **Throughput/TPOT sweep:** Replay Poisson arrivals (0.5–1.5× saturation) with `tools/loadgen/request_stream.py`. Optimized_ should deliver 2.3–2.8× TPOT thanks to continuous batching and healthier expert utilization.
3. **Latency heatmap:** Measure p50/p95 latencies at context {2 k, 8 k, 32 k}. Baseline degrades sharply at 32 k because of dense KV; optimized_ stays flatter through paged/quantized caches.
4. **Memory telemetry:** Record `torch.cuda.memory_allocated()`, allocator stats, and NVML board power. FP8 weights + paged KV should claw back ~18 GB and lower power/TPS.
5. **Kernel proof:** Run Nsight Compute on (a) baseline attention, (b) FlashMLA decode, (c) DeepEP all-to-all. Publish SM occupancy + DRAM throughput deltas alongside benchmark results, then feed the `.nsys-rep` artifacts into `python tools/profiling/parse_nvlink_metrics.py artifacts/<run_id>/*nsys-rep` to grab the NVLink sections emitted during `--profile` runs.
   - If Nsight Compute’s default preset is too heavy (the “full” set can require dozens of passes on TMEM/TMA kernels), append `--ncu-metric-set minimal` to your `tools/cli/benchmark_cli.py --profile` command. The harness then switches to the speed-of-light preset automatically and focuses on `gpu__time_duration`, `sm__throughput`, and `gpu__dram_throughput`, which keeps decode timelines profiled in a single pass.

### Rollout roadmap
1. **Week 1:** Land baseline script, correctness tests vs HuggingFace reference, simple profiler captures.  
2. **Week 2:** Add vLLM executor, paged KV cache, and tensor-parallel parity with the baseline.  
3. **Week 3:** Enable NVIDIA Dynamo-style prefill/decode disaggregation + DualPipe overlaps; capture Nsight traces verifying concurrency.  
4. **Week 4:** Layer SGLang speculative decoding and FP8/FP4 cache compression; tune draft accept ratio vs accuracy.  
5. **Demo:** Compare `artifacts/*baseline*/metrics.json` vs `artifacts/*optimized*/metrics.json`, highlight TTFT, throughput, TPOT, latency, and memory gains.

This plan keeps the baseline approachable while making every optimized_ win traceable back to vLLM, NVIDIA Dynamo, and SGLang design choices—ideal for a pedagogically meaningful MoE serving showcase.

## Wild CUDA Optimization Labs (baseline_ vs optimized_ variants)

### 5. Warp-Specialized Producer/Consumer Pipelines (Chapter 9)
- **What happened:** `book/ch9.md` dives into warp specialization and inline PTX so router microkernels quit stalling. We mirror that arc with the `ch9/baseline_warp_specialization_producer_consumer.py` PyTorch loop versus `ch9/optimized_warp_specialization_producer_consumer.py`, which calls the Triton `warp_specialized_triton.py` kernel and leans on `inline_ptx_example.cu` for cp.async.L2 prefetches and `%smid`/`%laneid` instrumentation.
- **GPU/CUDA tuning angles:** Producer/consumer warps are split via `warp_specialize=True` and `PIPELINE_STAGES=4`, letting loads overlap compute and keeping every warp busy; inline PTX (`cp.async.bulk.prefetch.L2.global`, `ld.global.cg`) bypasses L1 thrash on routing buffers; the harness itself records NVTX ranges so Nsight will show exactly when Triton’s pipeline feeds the consumer warps.
- **Repo demo:**  
  1. `python tools/cli/benchmark_cli.py run --targets ch9:warp_specialization_producer_consumer --profile` to capture baseline vs optimized traces and confirm the Triton kernel’s producer/consumer split.  
  2. `nvcc -std=c++17 -O3 -arch=sm_100 ch9/inline_ptx_example.cu -o build/ch9_inline_ptx && ./build/ch9_inline_ptx` to replay DeepEP-style cache hints (`cp.async`, `.cg` loads, `%smid` reads).  
  3. `ncu --section LaunchStats --section MemoryWorkloadAnalysis ./build/ch9_inline_ptx` (or point `ncu` at the CLI run) to verify higher L2 hit rate and improved warp residency once the optimized_ path is enabled.

### 6. DualPipe Warp Specialization + Stream-Ordered Pipelines (Chapter 11)
- **What happened:** `book/ch11.md` modernizes DualPipe for Hopper/Blackwell by leaning on CUDA 13 pipelines. Here `ch11/baseline_warp_specialized_two_pipelines_multistream.py` launches the legacy two-pipeline kernel with one stream, while `ch11/optimized_warp_specialized_two_pipelines_multistream.py` + `optimized_warp_specialized_two_pipelines_multistream.cu` add cooperative-groups warp roles, four CUDA streams, and `cuda::pipeline_shared_state` double-buffering.
- **GPU/CUDA tuning angles:** Stage-rotating shared memory tiles feed compute warps while a dedicated warp handles stores; `cuda::pipeline` lets the optimized kernel overlap producer, compute, and drain stages so each CTA permanently hosts inflight tiles; the Python benchmark pins four high-priority streams so NVLink copies, cp.async loads, and SM math all overlap DualPipe-style.
- **Repo demo:**  
  1. `python tools/cli/benchmark_cli.py run --targets ch11:warp_specialized_two_pipelines_multistream --profile` to compare the single-stream baseline to the fully overlapped optimized_ kernel (watch the NVTX ranges fan out).  
  2. `nsys profile -o artifacts/ch11_dualpipe python tools/cli/benchmark_cli.py run --targets ch11:warp_specialized_two_pipelines_multistream --iterations 5 --warmup 1` to see the stream-level overlap plus SM idling eliminated.  
  3. `ncu --set full --target-processes all --kernel-name regex:warp_specialized_kernel_two_pipelines_multistream python tools/cli/benchmark_cli.py run --targets ch11:warp_specialized_two_pipelines_multistream --iterations 1` to capture occupancy, shared-memory, and DRAM-throughput jumps caused by the optimized pipeline.

### 7. FlashMLA-Style Warp-Specialized Attention (Chapter 18)
- **What happened:** `book/ch18.md` explains how FlashMLA/ThunderMLA collapse decode kernels into a warp-specialized megakernel. In-repo, the baseline `ch18/baseline_warp_specialization_attention.py` sticks to eager FP16 SDPA, while `ch18/optimized_warp_specialization_attention.py` dispatches the Triton `warp_specialized_triton_forward_ch18` kernel, toggles NVTX-driven micro-batching, and lines up with `ch18/flashmla_kernel.cu` and `ch18/run_vllm_decoder.py` for real decode workers.
- **GPU/CUDA tuning angles:** Each warp owns a role (Q/K/V producers, attention accumulation, KV cache drains) so the optimized benchmark can micro-batch into `WORKLOAD.micro_batches`; Triton handles warp partitioning while the CUDA sketch shows how to fuse exp/denom accumulation inline; decode tokens recycle paged KV cache entries per head, exposing FlashMLA’s SM-occupancy win over dense SDPA.
- **Repo demo:**  
  1. `python tools/cli/benchmark_cli.py run --targets ch18:warp_specialization_attention --profile` to baseline vs optimized (and to catch Triton falling back when running on SM 12.1 hardware).  
  2. `nvcc -std=c++17 -arch=sm_120 ch18/flashmla_kernel.cu -o build/flashmla_kernel && ncu --section Regex:sm__pipe_fma_cycles_active ./build/flashmla_kernel` to sanity-check the fused decode loop before wiring it into serving.  
  3. `python ch18/run_vllm_decoder.py --max-tokens 256` paired with Nsight Systems to watch paged KV cache copies, FlashMLA launches, and router telemetry line up with the optimized_ attention benchmark.

### 8. Graph-Captured Paged Attention + Fused Quantization (Chapter 18)
- **What happened:** The paged-attention saga in `book/ch18.md` plus vLLM docs shows how to amortize prefill and MoE routing. `ch18/baseline_paged_attn.py` keeps dense FP16 SDPA and unfused expert routing, whereas `ch18/optimized_paged_attn_vllm.py` adds `torch.cuda.CUDAGraph`-captured prefill chunks, fused RMSNorm+INT8 fake quantization, paged KV cache residency tracking, and disaggregated router/expert execution.
- **GPU/CUDA tuning angles:** CUDA Graph capture (`self.prefill_graph`) replays QKV projection without kernel launch jitter; paged KV caches (`block_resident` + block-sized caches) keep decode DRAM traffic flat even at 32 k tokens; fused RMSNorm/quant saves bandwidth before tokens touch the MoE router; WorkloadMetadata feeds TPOT/TTFT calculations straight into `common/python/benchmark_harness.py`.
- **Repo demo:**  
  1. `python tools/cli/benchmark_cli.py run --targets ch18:paged_attn --profile` to see how graph-captured, quantized prefill shrinks TTFT versus the dense baseline.  
  2. `python tools/cli/benchmark_cli.py run --targets ch18:paged_attn --cold-start --iterations 1 --warmup 0 --artifacts-dir artifacts/paged_attn_flashmla` to stash Nsight traces showing paged-block residency and router load balance.  
  3. Feed the emitted metrics JSON into `tools/notebooks/plot_moe_ttft.ipynb` (or `common/python/metrics_extractor.py`) to visualize throughput, TPOT, and memory deltas between the baseline_ and optimized_ pipelines before integrating them into broader MoE serving demos.

### 9. Lab: End-to-End Production Graph (Scenario 09)
- **What happened:** The full-stack lab in `labs/fullstack_cluster/scenario_benchmark.py` threads Chapters 1–20 into production stories. Scenario “09_end_to_end” stitches `labs/fullstack_cluster/baseline_09_end_to_end.py` (single-stage trainer, sequential pipeline, naive KV cache) with `labs/fullstack_cluster/optimized_09_end_to_end.py`, which drives the `ch20/optimized_*` phases (graph-captured trainer, overlapped pipeline executor, bandwidth validation, and production-grade KV integration).
- **GPU/CUDA tuning angles:** Optimized phases use `torch.cuda.CUDAGraph` capture inside `ch20/optimized_training_single.py`, stage-to-stage overlap with CUDA Graph updates in `ch20/optimized_pipeline_sequential.py`, and stream-ordered KV cache orchestration from `ch20/optimized_integrated_kv_cache.py`. The scenario benchmark enforces consistent iterations so Nsight traces show how graph capture drops launch overhead and keeps the full pipeline saturated.
- **Repo demo:**  
  1. `python tools/cli/benchmark_cli.py run --targets labs/fullstack_cluster:09_end_to_end --profile` to compare the sequential baseline phases against the optimized graph-captured stack inside one CLI invocation.  
  2. `python labs/fullstack_cluster/run_lab_fullstack_cluster.py --size 4096 --iters 5 --baseline-iters 1` to time the lab’s CUDA kernels directly and print TFLOPS + baseline/optimized deltas for large GEMMs.  
  3. `nsys profile -o artifacts/labs_fullstack_cluster_e2e python tools/cli/benchmark_cli.py run --targets labs/fullstack_cluster:09_end_to_end --iterations 2` to visualize how CUDA Graph capture collapses kernel launches while the optimized pipeline keeps NVLink, Tensor Cores, and KV cache copies overlapped.

### 10. Blackwell Memory Forge (Chapter 19)
- **What happened:** Chapter 19 pushes FP8/FP4 memory tactics, async copy, and custom allocators. The `ch19/baseline_memory_double_buffering.py`, `baseline_vectorization_memory.py`, and `baseline_cutlass_memory.py` trio show naive coalescing. Their optimized counterparts plus `ch19/optimized_kv_prefetch_overlap.cu`, `dynamic_precision_switching.py`, and `dynamic_quantized_cache.py` demonstrate pipeline-aware cp.async, TransformerEngine FP8 autocast, and runtime-quantized KV caches.
- **GPU/CUDA tuning angles:** The optimized kernels rely on CUDA 13 `cp.async.bulk.tensor.2d.shared::cluster` paths, cross-SM shared-memory staging, and Tensor Memory Accelerator layout directives to keep HBM3e saturated; FP8/FP4 conversion picks between TE FP8 autocast and simulated NVFP4 fallback while monitoring allocator pressure; KV caches spill into quantized tensors when NVML-reported memory, via `dynamic_precision_switching.py`, breaches thresholds.
- **Repo demo:**  
  1. `python tools/cli/benchmark_cli.py run --targets ch19:memory_double_buffering --targets ch19:vectorization_memory --targets ch19:cutlass_memory --profile` to log throughput, SM occupancy, and copy/compute overlap across the baseline_ vs optimized_ kernels.  
  2. `nvcc -std=c++20 -O3 -arch=sm_120 ch19/optimized_kv_prefetch_overlap.cu -o build/ch19_kv_prefetch && ncu --section LaunchStats --section MemoryWorkloadAnalysis ./build/ch19_kv_prefetch` to inspect cp.async stages, shared-memory reuse, and KV prefetch overlap.  
  3. `python ch19/dynamic_precision_switching.py` (optionally after editing thresholds) to watch the runtime swap between FP16/BF16, FP8, and simulated FP4 as entropy/confidence scores change, then pair it with `python ch19/dynamic_quantized_cache.py --samples 4 --target-precision fp8` to see how paged KV caches compress when memory pressure spikes.

### 11. Adaptive Routing + Disaggregated Inference (Chapters 15 & 17)
- **What happened:** `book/ch17.md` covers dynamic routing for disaggregated prefill/decode fleets. In-repo, `ch17/baseline_routing_static.py` hardcodes router assignments, whereas `ch17/optimized_routing_static.py` plus `ch17/dynamic_routing.py` implement request-complexity estimation, hysteretic admission control, and worker-cost scoring. Pair these routers with `ch15/baseline_inference_monolithic.py` and `ch15/disaggregated_inference.py` to show how routing choices influence MoE serving efficiency.
- **GPU/CUDA tuning angles:** Optimized routing feeds the disaggregated executor with priority-tagged batches so `ch15/disaggregated_inference.py` can dedicate GPU sets to prefill vs decode; speculative tokens, FP8 expert activations, and `torch.compile`d prefill graphs only trigger for “complex” prompts; the dynamic router toggles NVSHMEM-ready queues and sets CUDA Graph flags (via `ch17/dynamic_routing.py`’s architecture hooks) when running on Blackwell to keep kernel replay compatible with changing batch shapes.
- **Repo demo:**  
  1. `python tools/cli/benchmark_cli.py run --targets ch17:routing_static --profile` to compare the static baseline_ router against the optimized admission-control path and export routing latency metrics.  
  2. `python ch17/dynamic_routing.py --config ch17/dynamo_config.yaml` (edit the YAML to match your GPU pool) to simulate live clusters, inspect latency-cost readouts, and watch how worker pools rebalance as request bursts arrive.  
  3. `python tools/cli/benchmark_cli.py run --targets ch15:inference_monolithic --profile` to capture the baseline_/optimized_ decode service pair, then launch `python ch15/disaggregated_inference.py` (after editing its configuration block to load `ch17/dynamo_config.yaml`) so you can see how the adaptive router reshapes TTFT/TPOT once prefill and decode live on separate GPU pools.
