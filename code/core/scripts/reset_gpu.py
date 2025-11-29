#!/usr/bin/env python3
"""Best-effort GPU reset utility.

Default behavior is user-space only: terminate compute processes, flush torch caches,
and (optionally) invoke `nvidia-smi --gpu-reset`.
With `--full-system` (root required) the script also mirrors `reset-gpu.sh` by
stopping display/helper services, toggling persistence mode, reloading kernel modules,
and issuing PCIe function-level resets.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def _log(msg: str) -> None:
    print(f"[reset-gpu.py] {msg}")


def _warn(msg: str) -> None:
    print(f"[reset-gpu.py][WARN] {msg}", file=sys.stderr)


def _collect_gpu_pids() -> List[int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:  # pragma: no cover
        _log(f"nvidia-smi unavailable: {exc}")
        return []

    pids: List[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or not line.isdigit():
            continue
        pid = int(line)
        if pid == os.getpid():
            continue
        pids.append(pid)
    return pids


def _terminate_pid(pid: int) -> None:
    for sig, wait in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 0.5)):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            _log(f"Insufficient permissions to signal PID {pid}")
            return
        time.sleep(wait)


def _flush_torch_cache() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            _log("Cleared torch CUDA cache")
    except Exception as exc:  # pragma: no cover
        _log(f"torch cuda cache flush failed: {exc}")


def _attempt_gpu_reset(device: int) -> None:
    try:
        subprocess.run(
            ["nvidia-smi", "--gpu-reset", "-i", str(device)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        # Ignore failures (often requires root)
        return


class SystemResetContext:
    DISPLAY_MANAGER_UNITS = (
        "display-manager",
        "gdm",
        "gdm3",
        "lightdm",
        "sddm",
        "lxdm",
    )
    DISPLAY_PROCESSES = (
        "Xorg",
        "X",
        "Xwayland",
        "gnome-shell",
        "kwin_x11",
        "kwin_wayland",
        "sway",
        "weston",
    )
    GPU_HELPER_UNITS = (
        "nvidia-persistenced",
        "nvidia-dcgm",
        "nvidia-powerd",
        "nvidia-fabricmanager",
        "nvidia-vgpu-mgr",
        "nvidia-vgpud",
        "nvidia-gridd",
        "nvidia-vgpu-dma",
    )
    MODULES_TO_REMOVE = ("nvidia_drm", "nvidia_modeset", "nvidia_uvm", "nvidia")
    MODULES_TO_ADD = ("nvidia", "nvidia_modeset", "nvidia_uvm", "nvidia_drm")

    def __init__(self) -> None:
        self.gpu_ids: List[str] = []
        self.gpu_persistence: Dict[str, str] = {}
        self.stopped_helper_units: List[str] = []
        self.stopped_display_units: List[str] = []
        self.systemctl_path = shutil.which("systemctl")
        self.pgrep_path = shutil.which("pgrep")

    def prepare(self) -> None:
        self._discover_gpus()
        self._stop_helper_services()
        self._stop_display_managers()
        self._terminate_display_processes()
        self._disable_persistence_mode()

    def cleanup(self) -> None:
        self._restore_persistence_mode()
        self._restart_display_managers()
        self._restart_helper_services()

    def reload_kernel_modules(self) -> None:
        removed_any = False
        for module in self.MODULES_TO_REMOVE:
            if not self._module_loaded(module):
                continue
            removed_any = True
            if self._run_command(["modprobe", "-r", module]):
                _log(f"Removed kernel module {module}")
            else:
                _warn(f"Failed to remove kernel module {module}")
        if not removed_any:
            _log("No NVIDIA kernel modules were loaded prior to reload step")
        for module in self.MODULES_TO_ADD:
            if self._run_command(["modprobe", module]):
                _log(f"Loaded kernel module {module}")
            else:
                _warn(f"Kernel module {module} could not be loaded (continuing)")

    def pci_function_level_reset(self) -> None:
        bus_ids = self._query_bus_ids()
        for raw in bus_ids:
            normalized = self._normalize_bus_id(raw)
            if not normalized:
                _warn(f"Unable to normalize PCI bus id: {raw}")
                continue
            device_path = Path("/sys/bus/pci/devices") / normalized
            if not device_path.is_dir():
                _warn(f"PCI device path not found: {device_path}")
                continue
            driver_name = ""
            driver_link = device_path / "driver"
            if driver_link.is_symlink():
                driver_name = driver_link.resolve().name

            if driver_name:
                unbind_path = Path("/sys/bus/pci/drivers") / driver_name / "unbind"
                if unbind_path.exists():
                    if self._write_sysfs(unbind_path, normalized):
                        _log(f"Unbound {normalized} from driver {driver_name}")
                    else:
                        _warn(f"Failed to unbind {normalized} from driver {driver_name}")

            reset_path = device_path / "reset"
            if reset_path.exists():
                if self._write_sysfs(reset_path, "1"):
                    _log(f"Issued PCIe function-level reset for {normalized}")
                else:
                    _warn(f"Failed to trigger PCIe reset for {normalized}")
            else:
                _warn(f"PCI reset interface not available for {normalized}")

            if driver_name:
                bind_path = Path("/sys/bus/pci/drivers") / driver_name / "bind"
                if bind_path.exists():
                    if self._write_sysfs(bind_path, normalized):
                        _log(f"Rebound {normalized} to driver {driver_name}")
                    else:
                        _warn(f"Failed to rebind {normalized} to driver {driver_name}")

    def _discover_gpus(self) -> None:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,persistence_mode",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            _warn("Unable to query GPU persistence state via nvidia-smi")
            return
        for line in result.stdout.splitlines():
            parts = [field.strip() for field in line.split(",", 1)]
            if not parts or not parts[0]:
                continue
            idx = parts[0]
            state = parts[1] if len(parts) > 1 else "Enabled"
            if idx not in self.gpu_ids:
                self.gpu_ids.append(idx)
            self.gpu_persistence[idx] = state
        if not self.gpu_ids:
            _warn("No NVIDIA GPUs detected while preparing full-system reset")

    def _stop_helper_services(self) -> None:
        self._stop_units(self.GPU_HELPER_UNITS, self.stopped_helper_units)

    def _restart_helper_services(self) -> None:
        self._restart_units(self.stopped_helper_units)
        self.stopped_helper_units.clear()

    def _stop_display_managers(self) -> None:
        self._stop_units(self.DISPLAY_MANAGER_UNITS, self.stopped_display_units)

    def _restart_display_managers(self) -> None:
        self._restart_units(self.stopped_display_units)
        self.stopped_display_units.clear()

    def _stop_units(self, candidates, bucket: List[str]) -> None:
        if not self.systemctl_path:
            return
        for base in candidates:
            unit = self._resolve_active_unit(base)
            if not unit or unit in bucket:
                continue
            if self._systemctl(["stop", unit]):
                bucket.append(unit)
                _log(f"Stopped service {unit}")
            else:
                _warn(f"Failed to stop service {unit}")

    def _restart_units(self, units: List[str]) -> None:
        if not self.systemctl_path:
            return
        for unit in reversed(units):
            if self._systemctl(["start", unit]):
                _log(f"Restarted service {unit}")
            else:
                _warn(f"Failed to restart service {unit}")

    def _resolve_active_unit(self, base: str) -> Optional[str]:
        if not self.systemctl_path:
            return None
        for candidate in (base, f"{base}.service"):
            if self._systemctl(["is-active", "--quiet", candidate]):
                return candidate
        return None

    def _systemctl(self, args) -> bool:
        if not self.systemctl_path:
            return False
        result = subprocess.run(
            [self.systemctl_path, *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def _terminate_display_processes(self) -> None:
        if not self.pgrep_path:
            return
        first_pass = True
        for _ in range(2):
            for proc in self.DISPLAY_PROCESSES:
                pids = self._collect_pids(proc)
                if not pids:
                    continue
                action = "Terminating" if first_pass else "Force killing"
                _log(f"{action} display process {proc} (PIDs: {pids})")
                for pid in pids:
                    self._signal_pid(pid, signal.SIGTERM if first_pass else signal.SIGKILL)
            if first_pass:
                time.sleep(1)
            first_pass = False

    def _collect_pids(self, name: str) -> List[int]:
        if not self.pgrep_path:
            return []
        result = subprocess.run(
            [self.pgrep_path, "-x", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        pids: List[int] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return pids

    def _signal_pid(self, pid: int, sig: signal.Signals) -> None:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            _warn(f"Insufficient permissions to signal PID {pid}")

    def _disable_persistence_mode(self) -> None:
        for gpu_id in self.gpu_ids:
            if self._run_command(["nvidia-smi", "-i", gpu_id, "-pm", "0"]):
                _log(f"Disabled persistence mode on GPU {gpu_id}")
            else:
                _warn(f"Unable to disable persistence mode on GPU {gpu_id}")

    def _restore_persistence_mode(self) -> None:
        for gpu_id, state in self.gpu_persistence.items():
            target = "0" if state.lower().startswith("disabled") else "1"
            if self._run_command(["nvidia-smi", "-i", gpu_id, "-pm", target]):
                _log(f"Restored persistence mode ({state}) on GPU {gpu_id}")
            else:
                _warn(f"Failed to restore persistence mode on GPU {gpu_id}")

    def _run_command(self, cmd) -> bool:
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return result.returncode == 0
        except FileNotFoundError as exc:
            _warn(f"Command not found while executing {' '.join(cmd)}: {exc}")
            return False

    @staticmethod
    def _module_loaded(module: str) -> bool:
        try:
            with open("/proc/modules", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith(f"{module} "):
                        return True
        except OSError:
            return False
        return False

    def _query_bus_ids(self) -> List[str]:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=pci.bus_id",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            _warn("Unable to enumerate GPU PCI bus IDs via nvidia-smi")
            return []
        bus_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return bus_ids

    @staticmethod
    def _normalize_bus_id(bdf: str) -> Optional[str]:
        if bdf.count(":") < 2:
            return None
        try:
            domain, bus, device_func = bdf.split(":", 2)
        except ValueError:
            return None
        if "." not in device_func:
            return None
        domain = domain[-4:].zfill(4)
        normalized = f"{domain}:{bus}:{device_func}".lower()
        return normalized

    @staticmethod
    def _write_sysfs(path: Path, value: str) -> bool:
        try:
            path.write_text(f"{value}\n", encoding="utf-8")
            return True
        except OSError as exc:
            _warn(f"Failed to write to {path}: {exc}")
            return False

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason", default="unspecified", help="Reason for reset request")
    parser.add_argument("--device", type=int, default=0, help="GPU index to reset (default: 0)")
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Only terminate processes / flush cache, skip nvidia-smi --gpu-reset",
    )
    parser.add_argument(
        "--full-system",
        action="store_true",
        help="Attempt service stop, kernel module reload, and PCIe FLR (requires root)",
    )
    parser.add_argument(
        "--skip-module-reload",
        action="store_true",
        help="Skip kernel module reload even when --full-system is set",
    )
    parser.add_argument(
        "--skip-pci-reset",
        action="store_true",
        help="Skip PCIe function-level reset (only applicable with --full-system)",
    )
    args = parser.parse_args()

    if args.full_system and os.geteuid() != 0:
        parser.error("--full-system requires root privileges")

    if (args.skip_module_reload or args.skip_pci_reset) and not args.full_system:
        _warn("--skip-module-reload/--skip-pci-reset have no effect without --full-system")

    system_reset: Optional[SystemResetContext] = SystemResetContext() if args.full_system else None

    _log(f"Starting best-effort GPU reset (reason: {args.reason})")

    try:
        if system_reset:
            system_reset.prepare()

        pids = _collect_gpu_pids()
        if pids:
            _log(f"Terminating {len(pids)} GPU compute processes: {pids}")
            for pid in pids:
                _terminate_pid(pid)
        else:
            _log("No active compute processes found via nvidia-smi")

        _flush_torch_cache()

        if not args.skip_reset:
            _log("Attempting nvidia-smi --gpu-reset (may require elevated privileges)")
            _attempt_gpu_reset(args.device)

        if system_reset:
            if not args.skip_module_reload:
                system_reset.reload_kernel_modules()
            if not args.skip_pci_reset:
                system_reset.pci_function_level_reset()
    finally:
        if system_reset:
            system_reset.cleanup()

    _log("Best-effort GPU reset complete")


if __name__ == "__main__":
    # Ensure script works even when executed from other directories
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
