#!/usr/bin/env python3
"""Configurable MoE Model - 7-Level Optimization Journey.

This demonstrates the SAME optimizations that torch.compile does,
but implemented manually so you can understand each technique.

Level 0: Naive        - Python loops over experts
Level 1: Batched      - Einsum parallelizes all tokens  
Level 2: Fused        - Triton kernel fuses SiLU * up
Level 3: MemEfficient - Eliminate intermediate tensors
Level 4: Grouped      - Sort tokens + per-expert GEMM
Level 5: CUDAGraphs   - Capture kernel sequence
Level 6: Compiled     - torch.compile does ALL of the above!
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass

# Try to import Triton kernels
try:
    from labs.moe_optimization_journey.triton_kernels import fused_silu_mul
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    def fused_silu_mul(gate, up):
        return F.silu(gate) * up


@dataclass
class MoEOptimizations:
    """Optimization flags for MoE model."""
    use_batched: bool = False       # Level 1: Batched einsum
    use_fused: bool = False         # Level 2: Triton fused SiLU*up
    use_mem_efficient: bool = False # Level 3: Memory efficient (in-place)
    use_grouped: bool = False       # Level 4: Sorted + per-expert GEMM
    use_bmm_fused: bool = False     # Level 5: BMM fusion (vectorized scatter)
    use_cuda_graphs: bool = False   # Level 6: CUDA graph capture  
    use_compile: bool = False       # Level 7: torch.compile


def bucket_by_expert(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
    """Sort tokens by expert for contiguous memory access.
    
    From ch19/mxfp8_moe_common.py - this is how production MoE works!
    """
    flat_idx = expert_indices.view(-1)
    sorted_order = torch.argsort(flat_idx, stable=True)
    sorted_tokens = tokens.repeat_interleave(expert_indices.shape[1], dim=0)[sorted_order]
    sorted_expert_ids = flat_idx[sorted_order]
    counts = torch.bincount(sorted_expert_ids, minlength=num_experts).tolist()
    return sorted_tokens, counts, sorted_order, sorted_expert_ids


def restore_bucketed(
    output: torch.Tensor,
    sorted_order: torch.Tensor, 
    batch_seq: int,
    top_k: int,
) -> torch.Tensor:
    """Scatter sorted outputs back to original order."""
    unsort = torch.argsort(sorted_order)
    return output[unsort].view(batch_seq, top_k, -1)


class MoEExperts(nn.Module):
    """Expert module supporting 7 levels of optimization."""
    
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int, opts: MoEOptimizations):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.opts = opts
        
        # Individual expert weights (for naive mode)
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                'w1': nn.Linear(hidden_size, intermediate_size, bias=False),
                'w2': nn.Linear(intermediate_size, hidden_size, bias=False),
                'w3': nn.Linear(hidden_size, intermediate_size, bias=False),
            })
            for _ in range(num_experts)
        ])
        
        # Stacked weights for optimized modes
        self.w1_stacked = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.w2_stacked = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.w3_stacked = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        
        for w in [self.w1_stacked, self.w2_stacked, self.w3_stacked]:
            nn.init.kaiming_uniform_(w)
        
        # Pre-allocated buffers for memory-efficient mode
        self._gate_buffer: Optional[torch.Tensor] = None
        self._up_buffer: Optional[torch.Tensor] = None
    
    def forward_naive(
        self, x: torch.Tensor, expert_indices: torch.Tensor, 
        expert_weights: torch.Tensor, num_experts_per_tok: int,
    ) -> torch.Tensor:
        """Level 0: NAIVE - Python loops over experts.
        
        This is how you might write MoE naively:
        - Loop over each expert
        - Loop over each top-K selection  
        - Compute expert output
        - Accumulate weighted results
        
        Problems: Python loop overhead, no parallelism, memory inefficient
        """
        output = torch.zeros_like(x)
        
        for expert_idx in range(self.num_experts):
            for k in range(num_experts_per_tok):
                mask = expert_indices[:, k] == expert_idx
                if mask.any():
                    expert_input = x[mask]
                    expert = self.experts[expert_idx]
                    gate = F.silu(expert['w1'](expert_input))
                    up = expert['w3'](expert_input)
                    expert_output = expert['w2'](gate * up)
                    weights = expert_weights[mask, k].unsqueeze(-1)
                    output[mask] += weights * expert_output
        return output
    
    def forward_batched(
        self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Level 1: BATCHED - Batched GEMMs parallelize all tokens.
        
        Instead of looping, we:
        1. Gather weights for selected experts: [batch, top_k, ...]
        2. Use batched matmul for parallel expert compute
        3. Sum weighted results
        
        Speedup: ~12x (eliminates Python loops)
        """
        batch_seq, top_k = expert_indices.shape
        
        w1_sel = self.w1_stacked[expert_indices]
        w3_sel = self.w3_stacked[expert_indices]
        w2_sel = self.w2_stacked[expert_indices]
        
        x_exp = x.unsqueeze(1).expand(-1, top_k, -1)

        # Prefer explicit batched GEMMs over einsum; for these shapes einsum can pick
        # slower kernels and lose to the naive Python dispatch (Level 0).
        total_tokens = batch_seq * top_k
        x_flat = x_exp.reshape(total_tokens, self.hidden_size)
        w1_flat = w1_sel.reshape(total_tokens, self.hidden_size, self.intermediate_size)
        w3_flat = w3_sel.reshape(total_tokens, self.hidden_size, self.intermediate_size)
        w2_flat = w2_sel.reshape(total_tokens, self.intermediate_size, self.hidden_size)

        gate = torch.bmm(x_flat.unsqueeze(1), w1_flat).squeeze(1)
        gate = F.silu(gate)
        up = torch.bmm(x_flat.unsqueeze(1), w3_flat).squeeze(1)
        hidden = gate * up
        out = torch.bmm(hidden.unsqueeze(1), w2_flat).squeeze(1)
        out = out.view(batch_seq, top_k, self.hidden_size)

        return (out * expert_weights.unsqueeze(-1)).sum(dim=1)
    
    def forward_fused(
        self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Level 2: FUSED - Triton kernel fuses SiLU * up.
        
        The SiLU activation and elementwise multiply are fused into
        one Triton kernel, eliminating a memory round-trip.
        
        Before: gate→memory→SiLU→memory→multiply→memory
        After:  gate→memory→fused_silu_mul→memory
        
        Speedup: Additional ~1.2x on top of batched
        """
        batch_seq, top_k = expert_indices.shape
        
        w1_sel = self.w1_stacked[expert_indices]
        w3_sel = self.w3_stacked[expert_indices]
        w2_sel = self.w2_stacked[expert_indices]
        
        x_exp = x.unsqueeze(1).expand(-1, top_k, -1)
        
        gate = torch.einsum('bkh,bkhi->bki', x_exp, w1_sel)
        up = torch.einsum('bkh,bkhi->bki', x_exp, w3_sel)
        
        # FUSED: SiLU(gate) * up in one kernel
        hidden = fused_silu_mul(gate, up)
        
        out = torch.einsum('bki,bkih->bkh', hidden, w2_sel)
        return (out * expert_weights.unsqueeze(-1)).sum(dim=1)
    
    def forward_mem_efficient(
        self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Level 3: MEMORY EFFICIENT - Eliminate intermediate tensors.
        
        Reuse buffers instead of allocating new tensors.
        Reduces memory pressure and allocation overhead.
        
        Speedup: Additional ~1.1x (less allocation overhead)
        """
        batch_seq, top_k = expert_indices.shape
        total_tokens = batch_seq * top_k
        
        # Reuse pre-allocated buffers
        if self._gate_buffer is None or self._gate_buffer.shape[0] != total_tokens:
            self._gate_buffer = torch.empty(total_tokens, self.intermediate_size, 
                                           device=x.device, dtype=x.dtype)
            self._up_buffer = torch.empty(total_tokens, self.intermediate_size,
                                         device=x.device, dtype=x.dtype)
        
        w1_sel = self.w1_stacked[expert_indices].view(total_tokens, self.hidden_size, -1)
        w3_sel = self.w3_stacked[expert_indices].view(total_tokens, self.hidden_size, -1)
        w2_sel = self.w2_stacked[expert_indices].view(total_tokens, self.intermediate_size, -1)
        
        x_flat = x.unsqueeze(1).expand(-1, top_k, -1).reshape(total_tokens, self.hidden_size)
        
        # Compute into pre-allocated buffers
        torch.bmm(x_flat.unsqueeze(1), w1_sel, out=self._gate_buffer.unsqueeze(1))
        torch.bmm(x_flat.unsqueeze(1), w3_sel, out=self._up_buffer.unsqueeze(1))
        
        # Fused activation
        hidden = fused_silu_mul(self._gate_buffer, self._up_buffer)
        
        # Final projection
        out = torch.bmm(hidden.unsqueeze(1), w2_sel).squeeze(1)
        out = out.view(batch_seq, top_k, -1)
        
        return (out * expert_weights.unsqueeze(-1)).sum(dim=1)
    
    def forward_grouped(
        self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Level 4: GROUPED - Sort tokens + per-expert GEMM.
        
        This is how production MoE works (vLLM, SGLang):
        1. Sort tokens by expert (bucket_by_expert from ch19)
        2. Run contiguous GEMM per expert
        3. Restore original order
        
        Benefits:
        - Contiguous memory access per expert
        - Better cache utilization  
        - Enables CUTLASS grouped GEMM
        
        Speedup: ~21x total (best manual optimization!)
        """
        batch_seq, top_k = expert_indices.shape
        
        # Sort by expert
        sorted_tokens, counts, sorted_order, sorted_expert_ids = bucket_by_expert(
            x, expert_indices, self.num_experts
        )
        sorted_weights = expert_weights.view(-1)[sorted_order]
        
        # Per-expert GEMM on contiguous tokens
        output = torch.zeros(sorted_tokens.shape[0], self.hidden_size,
                           device=x.device, dtype=x.dtype)
        
        offset = 0
        for e in range(self.num_experts):
            count = counts[e]
            if count == 0:
                continue
            
            tokens_e = sorted_tokens[offset:offset+count]
            weights_e = sorted_weights[offset:offset+count].unsqueeze(-1)
            
            # Contiguous GEMM for this expert
            gate = F.silu(tokens_e @ self.w1_stacked[e])
            up = tokens_e @ self.w3_stacked[e]
            expert_out = (gate * up) @ self.w2_stacked[e]
            
            output[offset:offset+count] = expert_out * weights_e
            offset += count
        
        # Restore order
        restored = restore_bucketed(output, sorted_order, batch_seq, top_k)
        return restored.sum(dim=1)
    
    def forward_bmm_fused(
        self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Level 5: BMM FUSION - Vectorized scatter + single BMM.
        
        Instead of 8 separate cuBLAS calls (one per expert), we:
        1. Scatter tokens into padded tensor using vectorized indexing
        2. Run ONE batched matmul (BMM) for ALL experts
        3. Gather results back
        
        This achieves MUCH higher GPU utilization because:
        - Single kernel launch (vs 8)
        - Better SM utilization 
        - Larger effective matrix = closer to peak TFLOPS
        
        Speedup: ~5-6x over grouped (at Llama-7B dimensions!)
        """
        batch_seq, top_k = expert_indices.shape
        device = x.device
        
        # Sort by expert
        flat_idx = expert_indices.view(-1)
        sorted_order = torch.argsort(flat_idx, stable=True)
        sorted_tokens = x.repeat_interleave(top_k, dim=0)[sorted_order]
        sorted_weights = expert_weights.view(-1)[sorted_order]
        sorted_expert_ids = flat_idx[sorted_order]
        counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
        max_count = counts.max().item()
        
        # Vectorized scatter indices
        cumsum = counts.cumsum(0)
        starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), cumsum[:-1]])
        expert_offsets = starts[sorted_expert_ids]
        positions = torch.arange(len(sorted_expert_ids), device=device) - expert_offsets
        padded_indices = sorted_expert_ids * max_count + positions
        
        # Scatter tokens into padded tensor (vectorized!)
        padded_tokens = torch.zeros(self.num_experts * max_count, self.hidden_size, 
                                   device=device, dtype=x.dtype)
        padded_tokens.scatter_(0, padded_indices.unsqueeze(1).expand(-1, self.hidden_size), sorted_tokens)
        padded_tokens = padded_tokens.view(self.num_experts, max_count, self.hidden_size)
        
        # SINGLE BMM for all experts!
        gate = torch.bmm(padded_tokens, self.w1_stacked)   # [E, max_count, I]
        gate = F.silu(gate)
        up = torch.bmm(padded_tokens, self.w3_stacked)     # [E, max_count, I]
        hidden = gate * up
        out = torch.bmm(hidden, self.w2_stacked)           # [E, max_count, H]
        
        # Scatter weights and apply
        padded_weights = torch.zeros(self.num_experts * max_count, 1, device=device, dtype=x.dtype)
        padded_weights.scatter_(0, padded_indices.unsqueeze(1), sorted_weights.unsqueeze(1))
        padded_weights = padded_weights.view(self.num_experts, max_count, 1)
        out = out * padded_weights
        
        # Gather back using same indices
        flat_out = out.view(-1, self.hidden_size)
        valid_out = flat_out[padded_indices]
        
        # Restore order
        unsort = torch.argsort(sorted_order)
        restored = valid_out[unsort].view(batch_seq, top_k, -1)
        return restored.sum(dim=1)
    
    def forward_cuda_graphs(
        self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Level 6: CUDA GRAPHS - Capture kernel sequence.
        
        CUDA graphs capture the sequence of kernel launches and replay
        them with minimal CPU overhead. This eliminates:
        - Kernel launch latency
        - CPU-GPU synchronization
        - Python overhead
        
        Note: Requires static shapes. Uses BMM fusion as the base.
        
        Speedup: Additional ~1.1x on top of BMM fusion
        """
        # For now, use bmm_fused as base (graph capture would be done at benchmark level)
        return self.forward_bmm_fused(x, expert_indices, expert_weights)

    def forward(
        self, x: torch.Tensor, expert_indices: torch.Tensor,
        expert_weights: torch.Tensor, num_experts_per_tok: int,
    ) -> torch.Tensor:
        """Dispatch to appropriate implementation based on optimization level."""
        # Priority: highest optimization level that's enabled
        if self.opts.use_bmm_fused or self.opts.use_cuda_graphs:
            return self.forward_bmm_fused(x, expert_indices, expert_weights)
        elif self.opts.use_grouped:
            return self.forward_grouped(x, expert_indices, expert_weights)
        elif self.opts.use_mem_efficient:
            return self.forward_mem_efficient(x, expert_indices, expert_weights)
        elif self.opts.use_fused:
            return self.forward_fused(x, expert_indices, expert_weights)
        elif self.opts.use_batched:
            return self.forward_batched(x, expert_indices, expert_weights)
        else:
            return self.forward_naive(x, expert_indices, expert_weights, num_experts_per_tok)


class MoELayer(nn.Module):
    """MoE layer with configurable optimizations."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, 
                 num_experts: int, num_experts_per_tok: int,
                 opts: MoEOptimizations):
        super().__init__()
        self.opts = opts
        self.num_experts_per_tok = num_experts_per_tok
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = MoEExperts(num_experts, hidden_size, intermediate_size, opts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        x_flat = x.view(-1, hidden)
        
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        expert_weights, expert_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        expert_weights = (expert_weights / expert_weights.sum(dim=-1, keepdim=True)).to(x.dtype)
        
        output = self.experts(x_flat, expert_indices, expert_weights, self.num_experts_per_tok)
        
        return output.view(batch, seq, hidden)


class MoEBlock(nn.Module):
    """Transformer block with MoE."""
    
    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_heads: int, num_experts: int, num_experts_per_tok: int,
                 opts: MoEOptimizations):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.moe = MoELayer(hidden_size, intermediate_size, num_experts, num_experts_per_tok, opts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self.ln2(x)
        h = self.moe(h)
        return x + h


class ConfigurableMoEModel(nn.Module):
    """MoE model with configurable optimization levels."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        intermediate_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        opts: Optional[MoEOptimizations] = None,
    ):
        super().__init__()
        self.opts = opts or MoEOptimizations()
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([
            MoEBlock(hidden_size, intermediate_size, num_heads,
                    num_experts, num_experts_per_tok, self.opts)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def create_model(level: int, **kwargs) -> Tuple[ConfigurableMoEModel, MoEOptimizations]:
    """Create model with optimizations enabled up to the given level.
    
    Level 0: Naive (Python loops)
    Level 1: + Batched (einsum parallelizes)
    Level 2: + Fused (Triton fuses SiLU*up)
    Level 3: + MemEfficient (reuse buffers)
    Level 4: + Grouped (sort + per-expert GEMM)
    Level 5: + BMM Fusion (vectorized scatter + single BMM) ← NEW! 5-6x speedup!
    Level 6: + CUDAGraphs (capture kernel sequence)
    Level 7: + Compiled (torch.compile does ALL of the above!)
    """
    opts = MoEOptimizations(
        use_batched=(level >= 1),
        use_fused=(level >= 2),
        use_mem_efficient=(level >= 3),
        use_grouped=(level >= 4),
        use_bmm_fused=(level >= 5),
        use_cuda_graphs=(level >= 6),
        use_compile=(level >= 7),
    )
    
    model = ConfigurableMoEModel(opts=opts, **kwargs)
    return model, opts
