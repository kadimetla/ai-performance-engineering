"""Optimized: sweep speculator configs and emit acceptance/chunk metrics to artifacts."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.artifact_manager import ArtifactManager  # noqa: E402
from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from ch18.run_vllm_decoder import GraphMode, VLLMMoEInferenceBenchmark  # noqa: E402


class SpecConfigSweepBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Run the VLLM MoE benchmark across multiple speculator configs."""

    def __init__(self) -> None:
        super().__init__()
        self.config_dir = REPO_ROOT / "ch18" / "spec_configs"
        explicit = os.getenv("SPEC_SWEEP_CONFIGS")
        if explicit:
            self.config_paths: List[Path] = [Path(p).expanduser() for p in explicit.split(",") if p.strip()]
        else:
            self.config_paths = sorted(self.config_dir.glob("*.json")) + sorted(self.config_dir.glob("*.yml")) + sorted(
                self.config_dir.glob("*.yaml")
            )
        if not self.config_paths:
            raise FileNotFoundError(f"No speculator configs found under {self.config_dir}")
        self.inner_iterations = int(os.getenv("SPEC_SWEEP_ITER", "2"))
        self.inner_warmup = int(os.getenv("SPEC_SWEEP_WARMUP", "1"))
        run_id = os.getenv("SPEC_SWEEP_RUN_ID", f"spec_config_sweep_{int(time.time())}")
        self.artifacts = ArtifactManager(run_id=run_id)
        self._custom_metrics: Dict[str, float] = {}
        self._verify_meta = torch.tensor([[float(len(self.config_paths)), float(self.inner_iterations), float(self.inner_warmup)]])
        self.register_workload_metadata(requests_per_iteration=1.0)

    def get_config(self) -> BenchmarkConfig:
        # Single outer iteration; inner harness handles per-config runs.
        return BenchmarkConfig(iterations=1, warmup=0)

    def benchmark_fn(self) -> None:
        results: Dict[str, Dict[str, float]] = {}
        summary: Dict[str, Dict[str, float]] = {}

        for cfg_path in self.config_paths:
            bench = VLLMMoEInferenceBenchmark()
            bench.spec_config_path = cfg_path
            bench.graph_mode = GraphMode.EAGER
            bench.enable_graphs = False

            cfg = bench.get_config()
            cfg.iterations = self.inner_iterations
            cfg.warmup = self.inner_warmup

            harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=cfg)
            run_result = harness.benchmark(bench)
            metrics = run_result.custom_metrics or {}
            record = {
                "accept_rate": float(metrics.get("optimized_moe.spec_accept_rate", 0.0)),
                "chunk_size": float(metrics.get("optimized_moe.spec_chunk_size", 0.0)),
                "throughput_tok_s": float(metrics.get("optimized_moe.throughput_tok_s", 0.0)),
                "ttft_mean_ms": float(metrics.get("optimized_moe.ttft_mean_ms", 0.0)),
                "tpot_mean_ms": float(metrics.get("optimized_moe.tpot_mean_ms", 0.0)),
            }
            results[cfg_path.name] = record
            for key, val in record.items():
                summary[f"{cfg_path.stem}.{key}"] = val

        payload = {
            "run_id": self.artifacts.run_id,
            "config_paths": [str(p) for p in self.config_paths],
            "iterations": self.inner_iterations,
            "warmup": self.inner_warmup,
            "results": results,
        }
        out_path = self.artifacts.get_result_path("spec_config_sweep.json")
        out_path.write_text(json.dumps(payload, indent=2))
        self._custom_metrics = summary

    def get_custom_metrics(self) -> Dict[str, float]:
        return self._custom_metrics

    def capture_verification_payload(self) -> None:
        if not self._custom_metrics:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        keys = sorted(self._custom_metrics.keys())
        values = [float(self._custom_metrics[k]) for k in keys]
        output = torch.tensor([values], dtype=torch.float32)
        self._set_verification_payload(
            inputs={"meta": self._verify_meta},
            output=output,
            batch_size=1,
            parameter_count=0,
            output_tolerance=(1e-4, 1e-4),
        )



def get_benchmark() -> BaseBenchmark:
    return SpecConfigSweepBenchmark()


def main() -> None:
    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    harness.benchmark(bench)


if __name__ == "__main__":
    main()
