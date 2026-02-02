# TODO

## Objective
Finish the remaining work with minimal-only profiling runs: tcgen05 validation, queue-driven benchmarks, NVTX sweep, and test cleanup.

## Current Status
- setup.sh is complete (setup.log shows "Setup Complete!" on 2026-02-02).
- Long-running ncu processes were terminated; queue replaced with minimal-only commands.
- Queue runner is running the minimal-only sequence.
- tcgen05 functional check passed for warp_specialized / warp_specialized_cutlass / warpgroup_specialized (N>=2048).
- MCP tool selection hints and dashboard API mappings verified.
- allow_virtualization is now enabled by default (CLI + harness) to avoid hypervisor failures.

## Remaining
1. Queue minimal tcgen05 benchmarks for the three targets with --update-expectations.
   - If speedup < 1.05x, increase sizes for baseline/optimized equivalently and re-run.
2. Queue a minimal run for ch11:tensor_cores_streams (deep_dive replaced by minimal).
3. Queue a minimal run for ch12:kernel_fusion_llm_dedicated_stream_and_prefetch_for_blackwell (deep_dive compare replaced by minimal).
4. Resume the full-suite minimal run via the queue runner; scan queue.log and benchmark.log for NVTX failures, patch, and requeue until clean.
5. Re-run the full pytest suite after remaining code changes and fix any failures.
