FlashAttention baseline_ vs optimized_ pairs (Blackwell)
=========================================================
Context: Blackwell shifts FlashAttention from a single software-pipelined warp group to warp-specialized partitions (loads, MMAs, softmax) coordinated via M-barriers. Use these pairs as a checklist when converting a naive implementation to a tuned one.

- baseline_pipeline: One warp group interleaves cp.async-style loads, MMAs, and softmax in a single loop (Ampere-style software pipelining).  
  optimized_pipeline: Warp-specialized partitions (e.g., load/TMA, QK MMA, softmax subtile 0, softmax subtile 1, PV MMA, correction) running concurrently and synced with M-barriers.

- baseline_layouts: Rely on compiler-chosen default block layouts; implicit layout conversions occur wherever the compiler decides.  
  optimized_layouts: Explicit layouts per tensor (block for global→shared loads, MMA layout for QK/PV operands, slice/swizzle layouts for softmax) with conversions placed deliberately to avoid hidden stalls.

- baseline_softmax: Single softmax over the full M tile after MMAs finish.  
  optimized_softmax: Subtile along M (e.g., split Q into top/bottom) so two softmax partitions overlap with ongoing MMAs; keeps the X unit saturated ~100% of the time.

- baseline_sync: Generic barriers relied on implicitly by the compiler; wait order is whichever PTX emits.  
  optimized_sync: Hand-ordered M-barrier acquires/releases so non-critical partitions wait first; softmax partitions remain on the critical path.

- baseline_memory_use: Accumulators and operands live wherever the compiler fits them; tensor memory usage is implicit.  
  optimized_memory_use: Tensor memory deliberately assigned (e.g., park PV left operand in tensor memory when available), with explicit lifetime management and handoff between partitions to reuse space without stalling.

- baseline_loads: Each warp issues its own small loads; load bandwidth depends on many active warps.  
  optimized_loads: Single TMA-issued bulk loads per tile; one warp saturates the TMA unit, freeing others for compute/softmax.

- baseline_critical_path: MMAs and softmax contend; critical path determined by compiler ordering.  
  optimized_critical_path: Profile-driven schedule tuned so softmax dominates and stays busy; MMAs and loads are overlapped to hide latency.

- baseline_autotune: Leave schedule and subtile choices entirely to compiler heuristics.  
  optimized_autotune: Human-guided choices for partition count, subtile size, and barrier ordering; encode them in the source (e.g., via Gluon meta-programming helpers).

- baseline_masking: Causal/non-causal masking handled in the monolithic loop; extra passes serialize work.  
  optimized_masking: Masking folded into warp-specialized softmax partitions so masking cost overlaps with MMA progress on other subtiles.

- baseline_reliance_on_compiler: Assume PTX reordering and software pipelining are “good enough.”  
  optimized_reliance_on_compiler: Explicit schedule expressed in source (layouts, partition roles, barrier order) so compiler heuristics cannot reorder away the intended overlap.
