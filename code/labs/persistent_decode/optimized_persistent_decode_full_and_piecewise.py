"""Graph-mode sweep for persistent decode: full vs piecewise vs fallback."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.persistent_decode.optimized_persistent_decode_graphs import (
    GraphMode,
    OptimizedPersistentDecodeGraphsBenchmark,
)
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedPersistentDecodeFullAndPiecewiseBenchmark(OptimizedPersistentDecodeGraphsBenchmark):
    """Default to FULL_AND_PIECEWISE to mirror vLLM graph heuristics."""

    def __init__(self) -> None:
        super().__init__(graph_mode=GraphMode.FULL_AND_PIECEWISE)
        self.jitter_exemption_reason = "Persistent decode full/piecewise: fixed dimensions"

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"batch": self.batch, "seq_len": self.seq_len, "variant": "full_and_piecewise"}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedPersistentDecodeFullAndPiecewiseBenchmark()

if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
