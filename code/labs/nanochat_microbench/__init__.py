"""NanoChat microbench lab (formerly fast_nanochat) with decode-focused benchmarks."""

__all__ = [
    "baseline_fast_nanochat",
    "optimized_fast_nanochat_pinned",
    "optimized_fast_nanochat_streams",
    "optimized_fast_nanochat_compile",
    "optimized_fast_nanochat_graph",
    "optimized_fast_nanochat_graph_full",
    "optimized_fast_nanochat_fp8",
    "optimized_fast_nanochat_fp4",
    "optimized_fast_nanochat_warp_specialized",
    "optimized_fast_nanochat_double_buffer_tma_cuda",
    "optimized_fast_nanochat_8xgpu",
    "nanochat_common",
]
