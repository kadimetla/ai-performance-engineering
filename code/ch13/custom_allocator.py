import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from arch_config import ArchitectureConfig
from common.device_utils import cuda_supported, get_preferred_device
import torch

_ARCH_CFG = ArchitectureConfig()


def get_architecture():
    """Detect and return the current GPU architecture."""
    if not torch.cuda.is_available():
        return "cpu"
    return _ARCH_CFG.arch


def get_architecture_info():
    """Get detailed architecture information."""
    return {
        "name": _ARCH_CFG.get_architecture_name(),
        "compute_capability": _ARCH_CFG.config.get("compute_capability", "Unknown"),
        "sm_version": _ARCH_CFG.config.get("sm_version", "sm_unknown"),
        "memory_bandwidth": _ARCH_CFG.config.get("memory_bandwidth", "Unknown"),
        "tensor_cores": _ARCH_CFG.config.get("tensor_cores", "Unknown"),
        "features": _ARCH_CFG.config.get("features", []),
    }

def demonstrate_custom_allocator():
    """Demonstrate custom CUDA memory allocator setup."""
    print("=== Custom CUDA Allocator Demo ===")
    
    if not cuda_supported():
        _, err = get_preferred_device()
        print("CUDA not available or unsupported, skipping allocator demo")
        if err:
            print(f"Reason: {err}")
        return
    
    # Show current allocator info
    print(f"Default allocator backend: {torch.cuda.get_allocator_backend()}")
    
    # Memory allocation patterns
    def test_allocation_pattern(name, allocator_config=None):
        print(f"\nTesting {name}:")

        prev_alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF")
        # Migrate deprecated PYTORCH_CUDA_ALLOC_CONF if present
        prev_deprecated_conf = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

        try:
            if allocator_config and "PYTORCH_ALLOC_CONF" in allocator_config:
                # Use PyTorch 2.10+ unified allocator configuration
                os.environ["PYTORCH_ALLOC_CONF"] = allocator_config["PYTORCH_ALLOC_CONF"]
            elif prev_deprecated_conf:
                # Migrate deprecated variable to new API
                os.environ["PYTORCH_ALLOC_CONF"] = prev_deprecated_conf
            else:
                os.environ.pop("PYTORCH_ALLOC_CONF", None)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Allocate various tensor sizes
            tensors = []
            sizes = [1024, 2048, 4096, 8192, 16384]

            for size in sizes:
                tensor = torch.randn(size, size, device='cuda')
                tensors.append(tensor)

            allocated = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6

            print(f"  Allocated: {allocated:.1f} MB")
            print(f"  Reserved: {reserved:.1f} MB")
            if reserved:
                print(f"  Efficiency: {allocated / reserved * 100:.1f}%")

            # Free tensors
            del tensors
            torch.cuda.empty_cache()

            return allocated, reserved
        finally:
            # Restore prior allocator configuration
            if prev_alloc_conf is not None:
                os.environ["PYTORCH_ALLOC_CONF"] = prev_alloc_conf
            else:
                os.environ.pop("PYTORCH_ALLOC_CONF", None)
    
    # Test default allocator
    default_alloc, default_reserved = test_allocation_pattern("Default Allocator")
    
    # Test with custom configuration (PyTorch 2.10+ unified API)
    custom_config = {
        "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync,max_split_size_mb:256,garbage_collection_threshold:0.6"
    }
    custom_alloc, custom_reserved = test_allocation_pattern(
        "Custom Configuration", custom_config
    )
    
    # Show improvement
    if default_reserved and custom_reserved and default_alloc:
        baseline_eff = default_alloc / default_reserved
        custom_eff = custom_alloc / custom_reserved
        print(f"\nMemory efficiency improvement: {custom_eff / baseline_eff - 1:.1%}")
    else:
        print("\nUnable to compute efficiency improvement due to zero reserved memory.")

def demonstrate_memory_pool():
    """Demonstrate memory pool management."""
    print("\n=== Memory Pool Demo ===")
    
    if not cuda_supported():
        _, err = get_preferred_device()
        print("CUDA not available or unsupported, skipping memory pool demo")
        if err:
            print(f"Reason: {err}")
        return
    
    # Get current memory pool
    device = torch.cuda.current_device()
    
    print(f"Current device: {device}")
    
    # Memory statistics before allocation
    print("\nMemory stats before allocation:")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1e6:.1f} MB")
    
    # Allocate some tensors
    tensors = []
    for i in range(5):
        tensor = torch.randn(1024, 1024, device=device)
        tensors.append(tensor)
    
    print("\nMemory stats after allocation:")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1e6:.1f} MB")
    
    # Free some tensors but keep references
    del tensors[::2]  # Delete every other tensor
    
    print("\nMemory stats after partial deallocation:")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1e6:.1f} MB")
    
    # Empty cache to return memory to OS
    torch.cuda.empty_cache()
    
    print("\nMemory stats after empty_cache():")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1e6:.1f} MB")
    
    # Clean up remaining tensors
    del tensors
    torch.cuda.empty_cache()

def demonstrate_memory_snapshot():
    """Demonstrate memory snapshot for debugging."""
    print("\n=== Memory Snapshot Demo ===")
    
    if not cuda_supported():
        _, err = get_preferred_device()
        print("CUDA not available or unsupported, skipping snapshot demo")
        if err:
            print(f"Reason: {err}")
        return
    
    # Enable memory history tracking
    torch.cuda.memory._record_memory_history(max_entries=200000)
    
    # Simulate some allocations
    tensors = []
    for i in range(3):
        # Create tensors of different sizes
        size = 1024 * (i + 1)
        tensor = torch.randn(size, size, device='cuda')
        tensors.append(tensor)
        print(f"Allocated tensor {i+1}: {size}x{size}")
    
    # Take snapshot
    torch.cuda.memory._dump_snapshot("memory_snapshot.json")
    snapshot = torch.cuda.memory.memory_snapshot()
    print(f"\nSnapshot contains {len(snapshot)} memory records")
    
    # Analyze snapshot (simplified)
    total_allocated = 0
    total_freed = 0
    
    # Convert snapshot to list if it's not already iterable
    snapshot_list = list(snapshot) if hasattr(snapshot, '__iter__') else []
    
    for event in snapshot_list[:10]:  # Show first 10 events
        if isinstance(event, dict):
            action = event.get('action', 'unknown')
            size = event.get('size', 0)
            
            if action == 'alloc':
                total_allocated += size
            elif action == 'free':
                total_freed += size
        else:
            # Handle case where event might not be a dict
            continue
    
    print(f"Sample analysis - Allocated: {total_allocated / 1e6:.1f} MB, Freed: {total_freed / 1e6:.1f} MB")
    
    # Note: You can save this data for visualization
    # torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    # print("Memory snapshot saved to memory_snapshot.pickle")
    # print("Load this file in PyTorch memory visualizer: https://pytorch.org/memory_viz")
    
    # Stop recording
    torch.cuda.memory._record_memory_history(enabled=None)
    
    # Clean up
    del tensors
    torch.cuda.empty_cache()

def demonstrate_distributed_memory():
    """Demonstrate memory management in distributed settings."""
    print("\n=== Distributed Memory Demo ===")
    
    try:
        # Check if we're in a distributed environment
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"Running in distributed mode: rank {rank}/{world_size}")
        else:
            print("Not in distributed environment, simulating single rank")
            rank = 0
            world_size = 1
        
        if cuda_supported():
            # Set device based on rank
            device = rank % torch.cuda.device_count()
            torch.cuda.set_device(device)
            
            print(f"Using GPU {device}")
            
            # Memory allocation per rank
            tensor_size = 1024 // world_size  # Distribute memory load
            tensor = torch.randn(tensor_size, tensor_size, device=f'cuda:{device}')
            
            print(f"Allocated {tensor_size}x{tensor_size} tensor on GPU {device}")
            print(f"Memory used: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
            
            del tensor
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Distributed memory demo error: {e}")

if __name__ == "__main__":
    demonstrate_custom_allocator()
    demonstrate_memory_pool()
    demonstrate_memory_snapshot()
    demonstrate_distributed_memory()

# Architecture-specific optimizations
if cuda_supported():
    device_props = torch.cuda.get_device_properties(0)

    inductor = getattr(torch, "_inductor", None)
    triton_cfg = getattr(getattr(inductor, "config", None), "triton", None) if inductor else None

    if _ARCH_CFG.arch in {"blackwell", "grace_blackwell"} and triton_cfg is not None:
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
