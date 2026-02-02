"""Chapter 8 tcgen05 tiling loader entrypoint used by the book examples."""

from core.common.tcgen05 import load_tiling_tcgen05_module as _load_tiling_tcgen05_module

def load_tiling_tcgen05_module():
    """Compile (if needed) and return the Chapter 8 tcgen05 tiling extension."""
    return _load_tiling_tcgen05_module()


__all__ = ["load_tiling_tcgen05_module"]
