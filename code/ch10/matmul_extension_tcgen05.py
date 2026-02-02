"""Chapter 10 tcgen05 loader entrypoint used by the book examples."""

from core.common.tcgen05 import load_matmul_tcgen05_module as _load_matmul_tcgen05_module

def load_matmul_tcgen05_module():
    """Compile (if needed) and return the Chapter 10 tcgen05 matmul extension."""
    return _load_matmul_tcgen05_module()


__all__ = ["load_matmul_tcgen05_module"]
