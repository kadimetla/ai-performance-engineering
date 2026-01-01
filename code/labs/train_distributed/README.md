# Lab - Distributed Training Playbook

## Summary
Collects distributed-training recipes for Blackwell clusters: DDP, FSDP, ZeRO-1/2/3, symmetric memory, and flash-attention-aware all-reduce handling, all runnable through the harness.

## Learning Goals
- Benchmark standard DDP vs optimized overlap-aware variants.
- Exercise FSDP and ZeRO strategies with shared helper utilities.
- Validate symmetric-memory training modes that pool NVLink bandwidth.
- Reuse launcher utilities (torchrun) with consistent configuration.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_ddp.py`, `optimized_ddp.py`, `baseline_ddp_flash.py`, `optimized_ddp_flash.py`, `baseline_ddp_multigpu.py`, `optimized_ddp_multigpu.py`, `baseline_ddp_flash_multigpu.py`, `optimized_ddp_flash_multigpu.py`, `baseline_ddp_compression_multigpu_int8.py`, `optimized_ddp_compression_multigpu_int8.py`, `baseline_ddp_compression_multigpu_powersgd.py`, `optimized_ddp_compression_multigpu_powersgd.py`, `ddp.py` | DDP workloads including flash-attention and compression variants (single + multi GPU). |
| `baseline_fsdp.py`, `optimized_fsdp.py`, `baseline_fsdp_multigpu.py`, `optimized_fsdp_multigpu.py`, `baseline_fsdp2.py`, `optimized_fsdp2.py`, `baseline_fsdp2_multigpu.py`, `optimized_fsdp2_multigpu.py`, `train_fsdp.py`, `train_fsdp2.py` | FSDP/FSDP2 scripts that demonstrate shard-by-shard memory savings. |
| `baseline_pipeline_1f1b.py`, `optimized_pipeline_1f1b.py`, `baseline_pipeline_gpipe.py`, `optimized_pipeline_gpipe.py`, `baseline_pipeline_dualpipe.py`, `optimized_pipeline_dualpipe.py`, `baseline_pipeline_dualpipev.py`, `optimized_pipeline_dualpipev.py`, `baseline_pipeline_1f1b_multigpu.py`, `optimized_pipeline_1f1b_multigpu.py`, `baseline_pipeline_gpipe_multigpu.py`, `optimized_pipeline_gpipe_multigpu.py`, `baseline_pipeline_1f1b_to_gpipe_multigpu.py`, `optimized_pipeline_1f1b_to_gpipe_multigpu.py`, `baseline_pipeline_gpipe_to_dualpipe_multigpu.py`, `optimized_pipeline_gpipe_to_dualpipe_multigpu.py`, `baseline_pipeline_gpipe_to_dualpipev_multigpu.py`, `optimized_pipeline_gpipe_to_dualpipev_multigpu.py`, `baseline_pipeline_dualpipe_multigpu.py`, `optimized_pipeline_dualpipe_multigpu.py`, `baseline_pipeline_dualpipev_multigpu.py`, `optimized_pipeline_dualpipev_multigpu.py`, `pipeline_*.py` | Pipeline parallelism schedules (single GPU simulations + multi-GPU execution). |
| `baseline_symmem_training.py`, `optimized_symmem_training.py`, `baseline_symmem_training_multigpu.py`, `optimized_symmem_training_multigpu.py` | Symmetric-memory strategies for optimizer state replication. |
| `baseline_zero1.py`, `baseline_zero2.py`, `baseline_zero3.py`, `optimized_zero1.py`, `optimized_zero2.py`, `optimized_zero3.py`, `baseline_zero1_multigpu.py`, `baseline_zero2_multigpu.py`, `baseline_zero3_multigpu.py`, `optimized_zero1_multigpu.py`, `optimized_zero2_multigpu.py`, `optimized_zero3_multigpu.py`, `zero1.py`, `zero2.py`, `zero3.py` | ZeRO implementations (1/2/3) plus helpers for parameter partitioning. |
| `training_utils/`, `utils.py`, `__init__.py` | Shared launch utilities, argument parsing, and harness exports. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/train_distributed
python -m cli.aisp bench run --targets labs/train_distributed --profile minimal
```
- Targets follow the `labs/train_distributed:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/train_distributed:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/train_distributed --profile minimal` runs every distributed configuration registered with the harness.
- `python labs/train_distributed/train_fsdp.py --validate` confirms numerical parity between FSDP shards and the baseline DDP path.
- `python labs/train_distributed/optimized_zero3_multigpu.py --summary` shows reduced peak memory vs the baseline script.

## Notes
- Set `TORCHRUN_ARGS` or pass `--torchrun-env` via the CLI when launching multi-node tests.
- `utils.py` exposes helper functions (like `resolve_topology()`) that can be reused in other labs.
- FSDP/FSDP2 benchmarks default to `labs/train_distributed/data/tinystories_packed_seq256.jsonl` plus `labs/train_distributed/data/tinyllama_config.json`, with `AISP_TINYSTORIES_LAYERS=4` to keep the model small. Override with `AISP_TINYSTORIES_PACKED_PATH`, `AISP_TINYSTORIES_LOCAL_PATH`, `AISP_TINYSTORIES_CONFIG_PATH`, or `AISP_TINYSTORIES_LAYERS`.
- Scale up by increasing `AISP_TINYSTORIES_LAYERS` or swapping to a larger config and pairing it with a packed dataset that matches the new sequence length.
- Set `AISP_FSDP_DISABLE_FP8=1` to keep the minimal BF16 path; unset it when you want to exercise the FP8 conversion on larger workloads.
