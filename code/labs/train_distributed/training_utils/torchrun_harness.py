"""Lightweight helpers for launching training demos via the benchmark harness."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.benchmark.verification import simple_signature


class TorchrunScriptBenchmark(BaseBenchmark):
    """Wrap a script-based training demo so the harness can launch it via torchrun."""

    verification_not_applicable_reason = "torchrun benchmarks execute in external processes"

    def __init__(
        self,
        *,
        script_path: Path,
        base_args: Optional[List[str]] = None,
        target_label: Optional[str] = None,
        config_arg_map: Optional[Dict[str, str]] = None,
        multi_gpu_required: bool = True,
        default_nproc_per_node: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self._script_path = Path(script_path)
        self._base_args = list(base_args) if base_args else []
        self._config_arg_map = config_arg_map or {}
        self._multi_gpu_required = multi_gpu_required
        self._default_nproc_per_node = default_nproc_per_node
        self._target_label = target_label
        self.name = name or self._script_path.stem
        # Compliance: verification interface
        self.register_workload_metadata(requests_per_iteration=1.0)

    def skip_input_verification(self) -> bool:
        return True

    def skip_output_verification(self) -> bool:
        return True

    def get_config(self) -> BenchmarkConfig:
        cfg = BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            multi_gpu_required=self._multi_gpu_required,
            nproc_per_node=self._default_nproc_per_node,
        )
        cfg.target_label = self._target_label
        return cfg

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=self._script_path,
            script_args=list(self._base_args),
            multi_gpu_required=self._multi_gpu_required,
            config_arg_map=self._config_arg_map,
            name=self.name,
        )

    def benchmark_fn(self) -> None:
        """Required abstract method; execution happens in torchrun subprocess."""
        raise RuntimeError("TorchrunScriptBenchmark should be executed via torchrun launcher.")
    
    def get_input_signature(self) -> Optional[dict]:
        """Return input signature for verification (static script identity)."""
        identity = (
            f"target={self._target_label}|"
            f"script={self._script_path.name}|"
            f"multi_gpu_required={int(self._multi_gpu_required)}|"
            f"nproc_per_node={self._default_nproc_per_node}"
        )
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        workload_id = int.from_bytes(digest[:4], byteorder="little", signed=False) % 10_000_000 + 1
        return simple_signature(batch_size=1, dtype="float32", workload=workload_id, script_len=len(self._script_path.name))

    def get_verify_output(self) -> "torch.Tensor":
        """Torchrun benchmarks run externally; verification not applicable."""
        raise RuntimeError("SKIPPED: torchrun benchmark verification not applicable (external process)")

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)
