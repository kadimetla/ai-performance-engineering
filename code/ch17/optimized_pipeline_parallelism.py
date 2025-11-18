"""optimized_pipeline_parallelism.py - Optimized pipeline parallelism across GPUs."""

from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedPipelineParallelismBenchmark(BaseBenchmark):
    """Optimized: Pipeline parallelism with layers split across GPUs."""

    def __init__(self):
        super().__init__()
        self.pipeline_stages: List[nn.Module] = []
        self.hidden_size = 1024
        self.batch_size = 256
        self.micro_batches = 4
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.stage_streams: List[torch.cuda.Stream] = []
        self.stage_events: List[List[torch.cuda.Event]] = []
        self.microbatch_inputs: Optional[List[torch.Tensor]] = None
        tokens = self.batch_size * self.hidden_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.micro_batches),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()

        torch.manual_seed(42)
        layers_per_stage = [
            [nn.Linear(self.hidden_size, self.hidden_size * 4), nn.GELU()],
            [nn.Linear(self.hidden_size * 4, self.hidden_size * 4), nn.GELU()],
            [nn.Linear(self.hidden_size * 4, self.hidden_size * 2), nn.GELU()],
            [nn.Linear(self.hidden_size * 2, self.hidden_size)],
        ]

        self.pipeline_stages = []
        for stage_id, layer_stack in enumerate(layers_per_stage):
            gpu_id = stage_id % self.num_gpus
            stage = nn.Sequential(*layer_stack).to(torch.device(f"cuda:{gpu_id}"), dtype=torch.bfloat16).eval()
            self.pipeline_stages.append(stage)

        self.microbatch_inputs = torch.randn(
            self.batch_size, self.hidden_size, device=torch.device("cuda:0"), dtype=torch.bfloat16
        ).chunk(self.micro_batches, dim=0)

        self.stage_streams = [torch.cuda.Stream(priority=-1) for _ in self.pipeline_stages]
        self.stage_events = [
            [torch.cuda.Event(enable_timing=False) for _ in range(self.micro_batches)]
            for _ in self.pipeline_stages
        ]
        self._synchronize()

    def benchmark_fn(self) -> None:
        if not self.pipeline_stages or self.microbatch_inputs is None:
            raise RuntimeError("Pipeline not initialized")

        num_stages = len(self.pipeline_stages)
        stage_buffers: List[List[Optional[torch.Tensor]]] = [
            [None for _ in range(self.micro_batches)] for _ in range(num_stages + 1)
        ]
        stage_buffers[0] = list(self.microbatch_inputs)

        stage_devices = [next(stage.parameters()).device for stage in self.pipeline_stages]

        with self._nvtx_range("optimized_pipeline_parallelism"):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for micro_idx in range(self.micro_batches + num_stages - 1):
                    for stage_idx, stage in enumerate(self.pipeline_stages):
                        chunk_idx = micro_idx - stage_idx
                        if chunk_idx < 0 or chunk_idx >= self.micro_batches:
                            continue
                        stream = self.stage_streams[stage_idx]
                        with torch.cuda.stream(stream):
                            if stage_idx > 0:
                                stream.wait_event(self.stage_events[stage_idx - 1][chunk_idx])
                            x = stage_buffers[stage_idx][chunk_idx]
                            if x is None:
                                continue
                            out = stage(x.to(stage_devices[stage_idx]))
                            next_stage_idx = stage_idx + 1
                            if next_stage_idx < len(stage_devices):
                                next_device = stage_devices[next_stage_idx]
                                if next_device != stage_devices[stage_idx]:
                                    out = out.to(next_device)
                            stage_buffers[next_stage_idx][chunk_idx] = out
                            self.stage_events[stage_idx][chunk_idx].record(stream)

        for stream in self.stage_streams:
            stream.synchronize()
        self._synchronize()

    def teardown(self) -> None:
        self.pipeline_stages = []
        self.microbatch_inputs = None
        self.stage_streams = []
        self.stage_events = []
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=12,
            warmup=2,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if not self.pipeline_stages:
            return "Pipeline stages not initialized"
        return None


def get_benchmark() -> OptimizedPipelineParallelismBenchmark:
    """Factory function for harness discovery."""
    return OptimizedPipelineParallelismBenchmark()
