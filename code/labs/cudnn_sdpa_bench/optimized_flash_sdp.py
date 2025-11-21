"""Optimized variant for cuDNN/Flash SDPA lab (shares implementation with baseline)."""

from __future__ import annotations

from labs.cudnn_sdpa_bench.baseline_flash_sdp import FlashSDPLabBenchmark


def get_benchmark() -> FlashSDPLabBenchmark:
    # Reuse the same benchmark; backend is controlled via target overrides/CLI.
    return FlashSDPLabBenchmark()
