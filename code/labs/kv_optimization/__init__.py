"""KV Cache Optimization Lab.

Demonstrates KV cache compression and optimization techniques for Blackwell:
- FP8 quantization (2× memory savings)
- NVFP4 quantization (4× memory savings)
- Dynamic scaling without calibration
- NVLink-pooled KV cache
"""

__all__ = [
    'baseline_kv_standard',
    'optimized_kv_fp8_compressed',
]


