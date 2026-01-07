# Chapter 4 - Multi-GPU Distribution

## Summary
Demonstrates how to scale training and inference across multiple Blackwell GPUs with NVLink/NVSwitch fabric awareness, NCCL tuning, NVSHMEM collectives, and symmetric memory patterns.

## Learning Goals
- Benchmark data-parallel and tensor-parallel training loops with and without overlap.
- Quantify NVLink bandwidth and topology effects when mixing local and disaggregated GPUs.
- Experiment with NVSHMEM pipelines to reduce host involvement in GPU synchronization.
- Adopt symmetric memory pools to simplify KV-cache replication and optimizer state sharding.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_dataparallel.py`, `optimized_dataparallel.py` | Single-GPU DataParallel anti-pattern vs direct GPU execution. |
| `baseline_dataparallel_multigpu.py`, `optimized_dataparallel_multigpu.py` | Multi-GPU DataParallel vs manual gradient reduction with pre-staged shards. |
| `baseline_no_overlap.py`, `optimized_no_overlap.py` | Overlap studies that stage compute/comm concurrency and pipeline microbatches to hide allreduce latency. |
| `baseline_nvlink.py`, `optimized_nvlink.py`, `baseline_nvlink_topology_aware.py`, `optimized_nvlink_topology_aware.py`, `baseline_nvlink_multigpu.py`, `optimized_nvlink_multigpu.py`, `baseline_nvlink_topology_aware_multigpu.py`, `optimized_nvlink_topology_aware_multigpu.py` | NVLink exercises for validating peer bandwidth and topology effects (single- and multi-GPU). |
| `baseline_continuous_batching.py`, `optimized_continuous_batching.py`, `baseline_disaggregated.py`, `optimized_disaggregated.py`, `baseline_continuous_batching_multigpu.py`, `optimized_continuous_batching_multigpu.py`, `baseline_disaggregated_multigpu.py`, `optimized_disaggregated_multigpu.py` | Continuous batching + disaggregated inference demos that showcase pooling and remote KV reuse. |
| `baseline_gradient_compression_fp16.py`, `optimized_gradient_compression_fp16.py`, `baseline_gradient_compression_int8.py`, `optimized_gradient_compression_int8.py`, `baseline_gradient_compression_fp16_multigpu.py`, `optimized_gradient_compression_fp16_multigpu.py`, `baseline_gradient_compression_int8_multigpu.py`, `optimized_gradient_compression_int8_multigpu.py` | Gradient compression all-reduce benchmarks comparing small-bucket vs full-buffer compression (single GPU and multi-GPU FP16/INT8 paths). |
| `baseline_gradient_compression_fp16_comm_only.py`, `optimized_gradient_compression_fp16_comm_only.py`, `baseline_gradient_compression_int8_comm_only.py`, `optimized_gradient_compression_int8_comm_only.py`, `baseline_gradient_compression_fp16_comm_only_multigpu.py`, `optimized_gradient_compression_fp16_comm_only_multigpu.py`, `baseline_gradient_compression_int8_comm_only_multigpu.py`, `optimized_gradient_compression_int8_comm_only_multigpu.py` | Communication-only gradient compression benchmarks with pre-quantized buffers (single GPU and multi-GPU FP16/INT8 paths). |
| `baseline_pipeline_parallel.py`, `optimized_pipeline_parallel_1f1b.py`, `baseline_tensor_parallel.py`, `optimized_tensor_parallel_async.py`, `baseline_torchcomms.py`, `optimized_torchcomms.py`, `baseline_pipeline_parallel_multigpu.py`, `optimized_pipeline_parallel_multigpu_1f1b.py`, `baseline_tensor_parallel_multigpu.py`, `optimized_tensor_parallel_multigpu.py`, `baseline_tensor_parallel_allgather_multigpu.py`, `optimized_tensor_parallel_allgather_multigpu.py`, `baseline_torchcomms_multigpu.py`, `optimized_torchcomms_multigpu.py` | Pipeline/tensor-parallel and torchcomms overlap studies (single- and multi-GPU). |
| `baseline_nvshmem_pipeline_parallel_multigpu.py`, `optimized_nvshmem_pipeline_parallel_multigpu.py`, `baseline_nvshmem_training_example_multigpu.py`, `optimized_nvshmem_training_example_multigpu.py` | NVSHMEM pipeline and training samples highlighting device-driven synchronization benefits. |
| `baseline_symmetric_memory_perf.py`, `optimized_symmetric_memory_perf.py`, `baseline_symmetric_memory_multigpu.py`, `optimized_symmetric_memory_multigpu.py`, `baseline_symmetric_memory_perf_multigpu.py`, `optimized_symmetric_memory_perf_multigpu.py` | Symmetric memory utilities and perf probes for KV cache and optimizer shards. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `bandwidth_benchmark_suite_multigpu.py`, `nccl_benchmark.py` | Harness driver plus standalone NCCL/NVLink sweepers for topology bring-up. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python ch04/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch04
python -m cli.aisp bench run --targets ch04 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python compare.py --examples dataparallel_multigpu` shows the optimized pair overlapping compute and communication with lower latency.
- `python bandwidth_benchmark_suite_multigpu.py --profile minimal` surfaces >=250 GB/s links on connected GPU pairs and highlights any slow hops.
- NVSHMEM samples emit consistent outputs when `NVSHMEM_SYMMETRIC_SIZE` is sized to hold the workload; mismatched config raises clear errors.

## Notes
- `symmetric_memory_*` helpers hold user-space allocators for pooling KV-cache lines across GPUs without NVSwitch penalties.
- Use `nccl_blackwell_config.py` to seed NCCL env vars (min NRings, IB mapping) before launching multi-node tests.
