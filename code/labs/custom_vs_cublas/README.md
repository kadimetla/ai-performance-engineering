# Lab: Matching cuBLAS on Blackwell

## Overview

This **self-contained** lab demonstrates progressive, compounding optimizations 
of a tcgen05 GEMM kernel towards matching NVIDIA's cuBLAS on Blackwell B200 GPUs.

Each stage builds on the previous one, showing how optimizations compound.

**No dependencies on other chapters** - everything needed is in this directory.

## Results Summary

### At 16384×16384 (Large Workload) - Best for demonstrating optimizations

| Stage | Optimization | TFLOPS | % of cuBLAS | Improvement |
|-------|--------------|--------|-------------|-------------|
| 0 | cuBLAS (target) | **1338** | 100% | — |
| 2 | + Tensor Cores (tcgen05) | 296 | 22% | baseline |
| 3 | + 2-Stage Pipeline | 402 | **30%** | **+1.36x** |
| 4 | + 3-Stage Pipeline | 533 | **40%** | **+1.33x** |
| 5 | + Swizzled Scheduling | 541 | 40% | +1.02x |
| 6 | + 4-Stage Deep Pipeline | 562 | **42%** | +1.04x |
| 7 | **+ No-Wait Pattern** | **805** | **60%** | **+1.43x** |
| 8 | + Swizzled Tiles | **840** | **63%** | +1.04x |
| 9 | + TMA before wait | **847** | **63%** | +1.01x |
| 10 | + Cluster Launch (2x1) | **855** | **64%** | +1.01x |
| **11** | **CUTLASS CollectiveBuilder** | **911** | **68%** | **+1.07x** |

### Key Breakthrough: The No-Wait Pattern (+43% improvement!)

The **single biggest optimization** was removing the MMA barrier wait inside the mainloop:

**Before (42% of cuBLAS):**
```cpp
for (int k = 0; k < num_k_tiles; ++k) {
  wait_barrier(tma_barrier);    // Wait for TMA
  gemm(...);                    // Compute MMA
  umma_arrive(&mma_barrier);    // Signal MMA done
  wait_barrier(mma_barrier);    // ❌ WAIT FOR MMA (unnecessary!)
}
```

**After (60% of cuBLAS):**
```cpp
for (int k = 0; k < num_k_tiles; ++k) {
  wait_barrier(tma_barrier);    // Wait for TMA
  gemm(...);                    // Compute MMA
  umma_arrive(&empty_barrier);  // Signal stage consumed (for pipeline)
  // ✅ NO WAIT - MMA hardware handles dependencies internally
}
// Wait once before epilogue
umma_arrive(&mma_barrier);
wait_barrier(mma_barrier);
```

### At 8192×8192

| Stage | Optimization | TFLOPS | % of cuBLAS |
|-------|--------------|--------|-------------|
| 0 | cuBLAS | 1350 | 100% |
| 6 | 4-Stage Pipeline | 489 | 36% |
| 8 | + No-Wait + Swizzle | 674 | **50%** |

### At 4096×4096

| Stage | Optimization | TFLOPS | % of cuBLAS |
|-------|--------------|--------|-------------|
| 0 | cuBLAS | 1017 | 100% |
| 6 | 4-Stage Pipeline | 347 | 34% |
| 8 | + No-Wait + Swizzle | 439 | **43%** |

**Key insight**: Larger matrices show better relative performance due to:
- Higher compute-to-memory ratio
- Better amortization of kernel launch overhead
- More tiles for pipeline to fill

**Note**: B200 theoretical peak is ~1800 TFLOPS FP16. cuBLAS achieves ~75% of peak.

## The Gap Analysis

We achieved **64% of cuBLAS** performance at 16K (855 vs 1338 TFLOPS). 
The remaining **36% gap** comes from:

### What cuBLAS Does That We Don't

1. **True Warp Specialization** (~15-20% of gap)
   - Dedicated producer warps (ONLY do TMA loads)
   - Dedicated consumer warps (ONLY do MMA compute)
   - Warps run IN PARALLEL via `PipelineTmaUmmaAsync`
   - Producer can run 3-4 tiles ahead of consumer
   - **We**: Same warp does both roles sequentially

2. **Persistent Kernels** (~10-15% of gap)
   - CTAs stay resident and process multiple tiles
   - Amortizes kernel launch overhead (~5µs per launch)
   - Better L2 cache locality between consecutive tiles
   - **We**: Launch new CTAs per tile, exit immediately

3. **Cluster Launch + TMA Multicast** (~5-10% of gap)
   - `cudaLaunchKernelEx` with cluster dimensions
   - TMA multicast delivers data to multiple CTAs
   - Shared B tile across M-dimension CTAs
   - **We**: Regular launch, no multicast

4. **SASS-level Tuning** (~5% of gap)
   - Hand-tuned instruction scheduling
   - Optimized register allocation
   - **We**: Compiler-generated code

## Running the Lab

```bash
# Run comprehensive benchmark
python run_lab.py

# Run with specific size
python run_lab.py --size 16384

# Verify correctness
python run_lab.py --verify
```

## Files

- `run_lab.py` - Main lab runner with all stages
- `tcgen05_loader.py` - JIT compiler for CUDA kernels
- `tcgen05_gemm.cu` - Stage 2: Basic tcgen05 kernel
- `tcgen05_pipelined.cu` - Stage 3: 2-stage pipeline
- `tcgen05_3stage.cu` - Stage 4: 3-stage pipeline
- `tcgen05_swizzled.cu` - Stage 5: Swizzled tile scheduling
- `tcgen05_warp_spec.cu` - Stage 6: 4-stage deep pipeline
- `tcgen05_no_wait.cu` - Stage 7: No-wait CUTLASS pattern (**key optimization!**)
- `tcgen05_no_wait_swizzle.cu` - Stage 8: No-wait + swizzled tiles
- `tcgen05_warp_parallel.cu` - Stage 9: TMA issued before wait
- `tcgen05_cluster.cu` - Stage 10: Cluster launch (2x1) (**best performance**)
- `kernels.cu` - Stage 1: Naive SMEM kernel (reference)
- `cutlass_gemm/` - Stage 11: CUTLASS CollectiveBuilder implementation (**68%**)

## Key Concepts Demonstrated

### Stage 2: Tensor Cores (tcgen05)
- SM100_MMA_F16BF16_SS operation (128×256 tiles)
- TMA (Tensor Memory Accelerator) for async loads
- TMEM (Tensor Memory) for accumulator storage

### Stage 3-6: Pipeline Depth
- 2-stage → 3-stage → 4-stage progression
- More stages = better TMA latency hiding
- 4 stages optimal (limited by SMEM)

### Stage 7: No-Wait Pattern (THE KEY INSIGHT)
- CUTLASS doesn't wait for MMA after every k_tile!
- MMA hardware handles internal dependencies
- `umma_arrive` signals stage consumed for pipeline
- Wait only once before epilogue
- **+43% performance improvement!**

### Stage 8: Swizzled Scheduling
- XOR swizzle pattern for tile ordering
- Improves L2 cache hit rate
- Additional +3% on top of no-wait

### Stage 9: TMA Before Wait
- Issue NEXT TMA before waiting for current one
- Maximizes overlap - TMA k+3 runs while we compute k
- Small but measurable improvement

### Stage 10: Cluster Launch
- Uses `cudaLaunchKernelEx` with 2x1 cluster
- `__cluster_dims__(2, 1, 1)` kernel attribute
- Enables TMA multicast potential (B shared across M tiles)
- **64% of cuBLAS achieved!**

### Stage 11: CUTLASS CollectiveBuilder (BEST!)
- Uses CUTLASS's high-level `CollectiveBuilder` API
- Automatically configures:
  - SM100 TMA with multicast (`SM100_TMA_2SM_LOAD_MULTICAST`)
  - True warp specialization (`MainloopSm100TmaUmmaWarpSpecialized`)
  - `PipelineTmaUmmaAsync` for producer/consumer parallelism
  - 2x2 cluster launch with TMA multicast
- **68% of cuBLAS - best hand-tunable result!**
- See `cutlass_gemm/` directory for implementation

## To Go Further (Reaching 80-90%)

To close the remaining gap to cuBLAS:

1. **Implement True Warp Specialization** (~15-20% gain)
   - Use CUTLASS `PipelineTmaUmmaAsync`
   - Separate producer/consumer warp roles
   - Requires:
     - `SM100_TMA_2SM_LOAD` / `SM100_TMA_2SM_LOAD_MULTICAST` (NOT SM90!)
     - `make_tma_atom_B_sm100` helper with cluster shape
     - `WarpCategory` roles (MMA, MainloopLoad, EpilogueLoad)

2. **Enable TMA Multicast** (~5-10% gain)
   - Use `SM90_TMA_LOAD_MULTICAST` or `SM100_TMA_2SM_LOAD_MULTICAST`
   - `create_tma_multicast_mask<Mode>(cta_layout, cta_coord)`
   - Only cluster leader loads B, others receive via multicast
   - Barrier arrival counts must match multicast pattern

3. **Study CUTLASS 4.x**
   - `examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu` - Complete example
   - `sm100_mma_warpspecialized.hpp` - How CUTLASS does it
   - `sm100_pipeline.hpp` - PipelineTmaUmmaAsync abstractions
   - Use `CollectiveBuilder` for automatic optimization

### Achieving 68% with CUTLASS

Using CUTLASS `CollectiveBuilder` we achieved **68% of cuBLAS**:

```python
from cutlass_gemm import cutlass_gemm
result = cutlass_gemm(A, B)  # A @ B^T
```

CUTLASS automatically handles:
1. **SM100 TMA operations** - Uses `SM100_TMA_2SM_LOAD_MULTICAST` for optimal Blackwell support
2. **True warp specialization** - `MainloopSm100TmaUmmaWarpSpecialized` with producer/consumer warps
3. **Deep pipeline** - `PipelineTmaUmmaAsync` manages complex state machines
4. **2x2 cluster with TMA multicast** - Shares B tiles across CTAs

### Why Not Higher?

The remaining **32% gap** to cuBLAS comes from:
- SASS-level instruction scheduling (hand-tuned by NVIDIA)
- Proprietary optimizations not in public CUTLASS
- cuBLAS may use different tile sizes per GPU SKU

## Requirements

- NVIDIA B200 or newer (SM 10.0+)
- CUDA 13.0+
- PyTorch 2.x with CUDA support
- CUTLASS 4.x (included in third_party/)
