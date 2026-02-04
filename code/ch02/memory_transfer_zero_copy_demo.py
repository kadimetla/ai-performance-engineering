"""Run the zero-copy memory transfer CUDA demo binary."""

from __future__ import annotations

import sys
from pathlib import Path

from core.utils.cuda_demo_runner import run_cuda_demo


if __name__ == "__main__":
    chapter_dir = Path(__file__).parent
    sys.exit(run_cuda_demo(chapter_dir, "memory_transfer_zero_copy_demo"))
