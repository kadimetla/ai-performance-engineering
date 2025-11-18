# FlexAttention Lab

This lab lifts the Colfax FlexAttention CuTe DSL blog post into the shared harness so you can compare the eager path against the compiled fused kernel.

## What it runs
- `baseline_flex_attention.py`: eager `flex_attention` with a Python `score_mod` and a block-sparse `block_mask`.
- `optimized_flex_attention.py`: same mask and score mod, wrapped in `torch.compile` to generate the fused FlexAttention kernel.
- `baseline_flex_attention_cute.py`: CuTe DSL path driving FlashAttention’s `_flash_attn_fwd` (no custom mask_mod/score_mod in the current FlashAttention API).
- `optimized_flex_attention_cute.py`: same CuTe path wrapped in `torch.compile` to keep parity with the blog’s compiled entry.

Run both via the harness:
```bash
python tools/cli/benchmark_cli.py run --targets labs/flexattention:flex_attention --profile
python tools/cli/benchmark_cli.py run --targets labs/flexattention:flex_attention_cute --profile
```

# Tunable knobs
- `block_size`: block size fed to `create_block_mask` (controls sparsity granularity). Defaults: 128 (64 in quick mode).
- `doc_span`: tokens per “document” when building the document-boundary mask (prevents cross-doc attention). Defaults: 256 (128 in quick mode).
- Shapes: `seq_len` (1024 / 512 quick), `batch` (2 / 1 quick), `heads` (8 / 4 quick), `head_dim` (64).
- `TORCH_COMPILE_MODE`: set to `reduce-overhead` for faster compile or leave default for maximum fusion.
- Shapes are fixed for now to keep the lab predictable: seq_len=1024, batch=2, heads=8, head_dim=64, block_size=128, doc_span=256.

To try different sparsity patterns, export env vars before running:
```bash
TORCH_COMPILE_MODE=reduce-overhead BLOCK_SIZE=64 DOC_SPAN=128 \
python tools/cli/benchmark_cli.py run --targets labs/flexattention:flex_attention --profile
```

## What to inspect
- Harness artifacts under `artifacts/<run_id>/labs_flexattention_*` for timing and Nsight traces.
- Use `tools/analysis/deep_profiling_report.py artifacts/<run_id>` to confirm the compiled path avoids dense score materialization and reduces kernel launches relative to the eager baseline.
