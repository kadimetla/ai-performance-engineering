"""Shape definitions for CUTLASS profiler sweeps.

These match typical transformer GEMMs (prefill and decode) so the profiler
searches kernels we actually care about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GemmShape:
    """Simple struct for GEMM shapes."""

    name: str
    m: int
    n: int
    k: int
    dtype: str = "f16"

    def as_dict(self) -> dict:
        return {"name": self.name, "m": self.m, "n": self.n, "k": self.k, "dtype": self.dtype}


def transformer_gemm_shapes() -> List[GemmShape]:
    """Curated transformer-ish GEMM shapes seen in prefill and decode."""

    # M ~ batch * sequence, N/K follow hidden sizes for decoder-only models.
    return [
        GemmShape(name="prefill_mlp_gather_m8192_n16384_k4096", m=8192, n=16384, k=4096),
        GemmShape(name="prefill_mlp_expand_m16384_n16384_k8192", m=16384, n=16384, k=8192),
        GemmShape(name="decode_mlp_m4096_n4096_k8192", m=4096, n=4096, k=8192),
        GemmShape(name="kv_proj_m4096_n4096_k4096", m=4096, n=4096, k=4096),
        GemmShape(name="attention_qk_m8192_n4096_k4096", m=8192, n=4096, k=4096),
    ]


__all__ = ["GemmShape", "transformer_gemm_shapes"]
