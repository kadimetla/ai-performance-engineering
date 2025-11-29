"""Lightweight GPU memory logging primitives used by the benchmark harness."""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, TextIO

import torch


def should_enable_gpu_memory_logging(enabled: bool) -> bool:
    """Honor the configuration flag; default is False."""
    return bool(enabled)


def resolve_gpu_log_interval(default_interval: float) -> float:
    """Clamp logging interval to a practical minimum."""
    return max(0.5, default_interval)


def resolve_gpu_log_path(config_path: Optional[str]) -> Path:
    """Use configured log path or fall back to a temp file."""
    if config_path:
        return Path(config_path).expanduser()
    timestamp = int(time.time() * 1e3)
    return Path(tempfile.gettempdir()) / f"gpu_mem_{timestamp}_{threading.get_ident()}.log"


class GpuMemoryLogger:
    """Background logger that samples torch.cuda.mem_get_info at fixed intervals."""

    def __init__(self, device: torch.device, interval: float, log_path: Path):
        self.device = device
        self.interval = max(0.5, interval)
        self.log_path = log_path
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._file: Optional[TextIO] = None

    def start(self) -> bool:
        """Start sampling loop. Returns False when CUDA is unavailable."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return False

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("a", buffering=1)
        self._file.write("# timestamp,used_bytes,total_bytes\n")
        self._thread = threading.Thread(target=self._run, name="gpu-mem-logger", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> Optional[Path]:
        """Stop the logging loop and flush the file."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)
        if self._file:
            self._file.flush()
            self._file.close()
        return self.log_path

    def _run(self) -> None:
        device_ctx = torch.cuda.device(self.device)
        while not self._stop.is_set():
            timestamp = time.time()
            line = None
            try:
                with device_ctx:
                    free, total = torch.cuda.mem_get_info(self.device)
                used = total - free
                line = f"{timestamp:.6f},{used},{total}\n"
            except Exception as exc:  # pragma: no cover - defensive
                line = f"{timestamp:.6f},ERROR,{exc}\n"
            finally:
                if self._file:
                    self._file.write(line)
            self._stop.wait(self.interval)
