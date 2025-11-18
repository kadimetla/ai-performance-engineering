from __future__ import annotations

import functools
from pathlib import Path

from torch.utils.cpp_extension import load


@functools.lru_cache(None)
def load_grace_blackwell_module():
    """Compile (if needed) and return the CUDA 13 extension."""
    # GB10 (sm_121) lacks TMA/DSMEM: quietly skip instead of failing the suite.
    try:
        import torch
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) == (12, 1):
            raise RuntimeError("SKIPPED: Grace-Blackwell extension requires DSMEM/TMA; SM121 does not enable them.")
    except RuntimeError:
        # propagate skip
        raise
    except Exception:
        pass

    root = Path(__file__).resolve().parent
    src = root / "grace_blackwell_kernels.cu"

    extra_cuda_cflags = [
        "-std=c++20",
        "-gencode=arch=compute_121,code=sm_121",
        "-gencode=arch=compute_100,code=sm_100",
        "-lineinfo",
        "-Xptxas=-v",
    ]
    extra_cflags = ["-std=c++20"]

    module = load(
        name="grace_blackwell_capstone_ext",
        sources=[str(src)],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=extra_cflags,
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
    return module
