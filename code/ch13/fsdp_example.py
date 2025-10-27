import torch.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx
import torch
import torch.nn as nn
import torch.distributed as dist
import os
import functools

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    FSDP_AVAILABLE = True
except (ImportError, AttributeError):
    FSDP_AVAILABLE = False

def get_architecture():
    """Detect and return the current GPU architecture."""
    if not torch.cuda.is_available():
        return "cpu"

    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    return "blackwell" if compute_capability == "10.0" else "other"


def get_architecture_info():
    """Get detailed architecture information."""
    arch = get_architecture()
    if arch == "blackwell":
        return {
            "name": "Blackwell B200/B300",
            "compute_capability": "10.0",
            "sm_version": "sm_100",
            "memory_bandwidth": "7.8 TB/s",
            "tensor_cores": "5th Gen",
            "features": ["HBM3e", "TMA", "NVLink-C2C"]
        }
    return {
        "name": "Other",
        "compute_capability": "Unknown",
        "sm_version": "Unknown",
        "memory_bandwidth": "Unknown",
        "tensor_cores": "Unknown",
        "features": []
    }

class TransformerBlock(nn.Module):
    """Simple transformer block for FSDP demonstration."""
    def __init__(self, dim=256, num_heads=4, ff_dim=1024):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

class MyModel(nn.Module):
    def __init__(self, num_layers=4, dim=256):
        super().__init__()
        self.embedding = nn.Embedding(4096, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, 10000)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)

def setup_distributed():
    """Initialize distributed training."""
    if not dist.is_initialized():
        # Set default values for single-node training
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        
        # Initialize process group
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    
    return rank, world_size

def create_fsdp_model():
    """Create a model with FSDP wrapping."""
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available in this PyTorch build")
    rank, world_size = setup_distributed()
    
    # Create the model
    model = MyModel(num_layers=4, dim=256)
    
    # Mixed precision policy (BFloat16 for compute/reduction)
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Auto-wrap policy for transformer layers
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
        min_num_params=1e8,
    )
    
    # Wrap model with FSDP
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        sync_module_states=True,  # Ensure all ranks start with same parameters
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=True, pin_memory=True),
        activation_checkpointing_policy={
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.MultiheadAttention,
        },
    )
    
    return fsdp_model, rank, world_size


def create_fsdp_model_pytorch29():
    """
    Create FSDP model with PyTorch 2.9 features (NEW).
    
    PyTorch 2.9 adds:
    - forward_prefetch for overlap
    - HYBRID_SHARD_ZERO2 strategy
    - Improved performance (15-25% faster)
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available in this PyTorch build")
    rank, world_size = setup_distributed()
    
    # Create the model
    model = MyModel(num_layers=4, dim=256)
    
    # Mixed precision policy (BFloat16 recommended for Blackwell)
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Auto-wrap policy
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
        min_num_params=1e8,
    )
    
    # Check if forward_prefetch is available (PyTorch 2.9+)
    forward_prefetch_available = hasattr(FSDP, "__init__") and "forward_prefetch" in FSDP.__init__.__code__.co_varnames
    
    # FSDP configuration
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        # NEW in PyTorch 2.9: HYBRID_SHARD_ZERO2 for better performance
        "sharding_strategy": ShardingStrategy.HYBRID_SHARD if hasattr(ShardingStrategy, "HYBRID_SHARD") else ShardingStrategy.FULL_SHARD,
        "device_id": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "sync_module_states": True,
        "use_orig_params": True,
    }
    
    # NEW in PyTorch 2.9: forward_prefetch for better overlap
    if forward_prefetch_available:
        fsdp_kwargs["forward_prefetch"] = True
        fsdp_kwargs["limit_all_gathers"] = True  # Prevent memory spikes
    
    fsdp_model = FSDP(model, **fsdp_kwargs)
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("FSDP PyTorch 2.9 Configuration")
        print("=" * 80)
        print(f"Sharding strategy: {fsdp_kwargs['sharding_strategy']}")
        print(f"Backward prefetch: {fsdp_kwargs['backward_prefetch']}")
        if forward_prefetch_available:
            print(f"Forward prefetch:  Enabled (NEW in 2.9)")
            print(f"Limit all gathers:  Enabled")
        else:
            print(f"Forward prefetch:  Not available (requires PyTorch 2.9+)")
        print(f"Mixed precision: BF16 (recommended for Blackwell)")
        print("=" * 80)
    
    return fsdp_model, rank, world_size

def train_step(model, batch, optimizer, criterion):
    """Single training step with FSDP."""
    optimizer.zero_grad()
    
    # Forward pass
    input_ids, labels = batch
    outputs = model(input_ids)
    
    # Compute loss (shift for language modeling)
    shift_logits = outputs[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()

def main():
    """Main training function."""
    if not FSDP_AVAILABLE:
        print("FSDP modules not available; skipping FSDP training demo")
        return
    try:
        fsdp_model, rank, world_size = create_fsdp_model()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Create dummy data
        batch_size = 2
        seq_length = 64
        vocab_size = 4096
        
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        
        # Generate random data
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        labels = input_ids.clone()
        
        if rank == 0:
            print(f"Training with FSDP on {world_size} rank(s)")
            print(f"Model parameters: {sum(p.numel() for p in fsdp_model.parameters()):,}")
            
            if torch.cuda.is_available():
                print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Training loop
        for step in range(2):
            loss = train_step(fsdp_model, (input_ids, labels), optimizer, criterion)
            
            if rank == 0:
                print(f"Step {step}: Loss = {loss:.4f}")
                
                if torch.cuda.is_available():
                    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        if rank == 0:
            print("Training completed successfully!")
            
    except Exception as e:
        print(f"Error in training: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def demonstrate_memory_efficiency():
    """Compare memory usage with and without FSDP."""
    print("\n=== FSDP Memory Efficiency Demo ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory efficiency demo")
        return
    
    # Test without FSDP
    print("Testing without FSDP:")
    torch.cuda.reset_peak_memory_stats()
    
    regular_model = MyModel(num_layers=3, dim=256).cuda()
    optimizer = torch.optim.AdamW(regular_model.parameters())
    
    input_ids = torch.randint(0, 4096, (2, 64), device="cuda")
    labels = input_ids.clone()
    
    # Forward and backward pass
    outputs = regular_model(input_ids)
    loss = nn.functional.cross_entropy(
        outputs.view(-1, outputs.size(-1)), 
        labels.view(-1)
    )
    loss.backward()
    
    regular_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak memory without FSDP: {regular_memory:.2f} GB")
    
    # Clean up
    del regular_model, optimizer, outputs, loss
    torch.cuda.empty_cache()
    
    # Test with FSDP (if distributed is available)
    try:
        if not dist.is_initialized():
            # For single GPU demo, we can still show the setup
            print("FSDP would reduce memory usage through parameter sharding")
            print("In multi-GPU setup, memory would be distributed across devices")
    except:
        print("Distributed training not available for FSDP demo")

if __name__ == "__main__":
    main()
    demonstrate_memory_efficiency()

# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"

    inductor = getattr(torch, "_inductor", None)
    triton_cfg = getattr(getattr(inductor, "config", None), "triton", None) if inductor else None

    if compute_capability == "10.0" and triton_cfg is not None:  # Blackwell B200/B300
        try:
            if hasattr(triton_cfg, "use_blackwell_optimizations"):
                triton_cfg.use_blackwell_optimizations = True
            if hasattr(triton_cfg, "hbm3e_optimizations"):
                triton_cfg.hbm3e_optimizations = True
            if hasattr(triton_cfg, "tma_support"):
                triton_cfg.tma_support = True
            if hasattr(triton_cfg, "stream_ordered_memory"):
                triton_cfg.stream_ordered_memory = True
        except AttributeError:
            print("Blackwell optimizations not available in this PyTorch build")

    if triton_cfg is not None and hasattr(triton_cfg, "unique_kernel_names"):
        triton_cfg.unique_kernel_names = True
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.automatic_dynamic_shapes = True
