"""NVML power sampling utilities."""

from __future__ import annotations

import threading
import time
from typing import Dict, Iterable, List, Optional

try:
    import pynvml  # type: ignore
except ImportError as exc:  # pragma: no cover - required dependency
    raise RuntimeError(
        "NVML power sampling requires pynvml (nvidia-ml-py) when CUDA is available."
    ) from exc

_NVML_INITIALIZED = False


def ensure_nvml_initialized() -> None:
    global _NVML_INITIALIZED
    if _NVML_INITIALIZED:
        return
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        raise RuntimeError(f"NVML initialization failed: {exc}") from exc
    _NVML_INITIALIZED = True


def shutdown_nvml() -> None:
    global _NVML_INITIALIZED
    if not _NVML_INITIALIZED:
        return
    try:
        pynvml.nvmlShutdown()
    except Exception as exc:
        raise RuntimeError(f"NVML shutdown failed: {exc}") from exc
    _NVML_INITIALIZED = False


class PowerSampler:
    """Background NVML sampler gathering aggregate power statistics."""

    def __init__(self, gpu_indices: Iterable[int], interval: float) -> None:
        ensure_nvml_initialized()
        self.interval = interval
        self.gpu_indices = list(gpu_indices)
        self.samples: List[tuple[float, List[float], float]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._error: Optional[BaseException] = None
        self._error_lock = threading.Lock()
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in self.gpu_indices]

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("PowerSampler already running.")
        self._error = None
        self.samples = []
        self._stop_event = threading.Event()
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, object]:
        if self._thread is None:
            return self._build_metrics()
        assert self._stop_event is not None
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        self._stop_event = None
        self._raise_if_error()
        self._sample_once()
        return self._build_metrics()

    def close(self) -> None:
        self._raise_if_error()
        shutdown_nvml()
        self._handles = []

    def _run(self) -> None:
        assert self._stop_event is not None
        try:
            while not self._stop_event.wait(self.interval):
                self._sample_once()
        except BaseException as exc:
            self._record_error(exc)
            self._stop_event.set()

    def _sample_once(self) -> None:
        timestamp = time.time()
        per_device: List[float] = []
        total_power = 0.0
        for handle in self._handles:
            milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
            watts = milliwatts / 1000.0
            per_device.append(float(watts))
            total_power += watts
        self.samples.append((timestamp, per_device, total_power))

    def _build_metrics(self) -> Dict[str, object]:
        if len(self.samples) < 2:
            raise RuntimeError(
                "PowerSampler requires at least two samples to compute energy metrics."
            )

        totals = [sample[2] for sample in self.samples]
        timestamps = [sample[0] for sample in self.samples]
        duration = timestamps[-1] - timestamps[0]

        energy = 0.0
        for idx in range(1, len(self.samples)):
            dt = timestamps[idx] - timestamps[idx - 1]
            energy += 0.5 * (totals[idx - 1] + totals[idx]) * dt

        per_device_stats: List[Dict[str, float]] = []
        per_device_series = list(zip(*[sample[1] for sample in self.samples]))
        for gpu_idx, series in zip(self.gpu_indices, per_device_series):
            series_list = list(series)
            per_device_stats.append(
                {
                    "gpu_index": gpu_idx,
                    "min_watts": float(min(series_list)),
                    "max_watts": float(max(series_list)),
                    "avg_watts": float(sum(series_list) / len(series_list)),
                }
            )

        return {
            "avg_watts": float(sum(totals) / len(totals)),
            "max_watts": float(max(totals)),
            "min_watts": float(min(totals)),
            "duration_s": float(duration),
            "energy_joules": float(energy),
            "per_device": per_device_stats,
        }

    def _record_error(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc

    def _raise_if_error(self) -> None:
        with self._error_lock:
            if self._error is None:
                return
            raise RuntimeError("NVML power sampling failed.") from self._error
