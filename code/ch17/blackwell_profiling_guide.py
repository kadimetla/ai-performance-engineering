"""
Profiling Guide for Blackwell B200/B300 Optimizations
=======================================================

This module provides comprehensive profiling workflows for Blackwell GPUs
using Nsight Systems, Nsight Compute, and PyTorch Profiler.

Tools Covered:
1. Nsight Systems - Timeline analysis
2. Nsight Compute - Kernel profiling
3. PyTorch Profiler - Python-level profiling
4. HTA (Holistic Tracing Analysis) - Distributed profiling

Requirements:
- CUDA 13.0+
- Nsight Systems 2024.1+
- Nsight Compute 2024.1+
- PyTorch 2.9+

Author: Blackwell Optimization Project
"""

import torch
import torch.profiler as profiler
import os
import subprocess
from pathlib import Path
from typing import Optional, List

# ============================================================================
# 1. Nsight Systems Profiling
# ============================================================================

class NsightSystemsProfiler:
    """
    Nsight Systems profiling for Blackwell
    
    Features:
    - Timeline visualization
    - CUDA API trace
    - Kernel execution timeline
    - Memory transfers
    - Multi-GPU traces
    
    Usage:
        profiler = NsightSystemsProfiler("my_trace")
        with profiler:
            # Your code here
            model(input)
    """
    
    def __init__(
        self,
        output_name: str,
        trace_cuda: bool = True,
        trace_nvtx: bool = True,
        trace_cudnn: bool = True,
        trace_cublas: bool = True,
    ):
        self.output_name = output_name
        self.trace_cuda = trace_cuda
        self.trace_nvtx = trace_nvtx
        self.trace_cudnn = trace_cudnn
        self.trace_cublas = trace_cublas
    
    def __enter__(self):
        """Start profiling"""
        # PyTorch NVTX markers
        torch.cuda.nvtx.range_push(f"Profile: {self.output_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling"""
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    
    @staticmethod
    def profile_command(
        script_path: str,
        output_name: str,
        duration: int = 30,
    ) -> str:
        """
        Generate nsys profile command
        
        Args:
            script_path: Path to Python script
            output_name: Output filename
            duration: Max duration in seconds
            
        Returns:
            Command string to execute
        """
        cmd = [
            "nsys", "profile",
            "--trace=cuda,nvtx,osrt,cudnn,cublas",
            "--cuda-memory-usage=true",
            f"--duration={duration}",
            "--force-overwrite=true",
            f"--output={output_name}",
            "--export=sqlite",
            "python", script_path
        ]
        return " ".join(cmd)
    
    @staticmethod
    def analyze_blackwell_metrics(report_path: str):
        """
        Analyze Blackwell-specific metrics from nsys report
        
        Key metrics:
        - SM utilization (target: >80% on 148 SMs)
        - Memory bandwidth (target: >7 TB/s for HBM3e)
        - Tensor Core utilization
        - NVLink bandwidth (if multi-GPU)
        """
        print(f"=== Nsight Systems Analysis: {report_path} ===\n")
        
        print("Key Metrics to Check:")
        print("1. GPU Utilization:")
        print("   - Target: >80% on Blackwell's 148 SMs")
        print("   - Look for: GPU Utilization timeline")
        
        print("\n2. Memory Bandwidth:")
        print("   - Target: >7 TB/s (HBM3e peak: 7.8 TB/s)")
        print("   - Look for: Memory (HBM) Read/Write Throughput")
        
        print("\n3. Tensor Core Utilization:")
        print("   - Target: >70% for compute-bound workloads")
        print("   - Look for: Tensor Core Active cycles")
        print("   - Blackwell: Uses tcgen05.mma (not WGMMA)")
        
        print("\n4. Kernel Launch Overhead:")
        print("   - With CUDA graphs: <100 μs")
        print("   - Without: ~10-50 μs per launch")
        
        print("\n5. Multi-GPU (if applicable):")
        print("   - NVLink bandwidth: Target ~900 GB/s per link")
        print("   - NCCL operations: Check for overlapping")
        
        print("\nVisualization:")
        print(f"  Open in Nsight Systems: nsys-ui {report_path}")


# ============================================================================
# 2. Nsight Compute Profiling
# ============================================================================

class NsightComputeProfiler:
    """
    Nsight Compute kernel-level profiling
    
    Features:
    - Detailed kernel metrics
    - Roofline analysis
    - Memory access patterns
    - Warp efficiency
    - Tensor Core utilization
    """
    
    @staticmethod
    def profile_kernel_command(
        script_path: str,
        output_name: str,
        kernel_filter: Optional[str] = None,
    ) -> str:
        """
        Generate ncu profile command
        
        Args:
            script_path: Path to Python script
            output_name: Output filename
            kernel_filter: Kernel name filter (e.g., "matmul")
            
        Returns:
            Command string
        """
        metrics = [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",  # SM utilization
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",  # Memory BW
            "gpu__time_duration.sum",  # Kernel duration
            "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",  # Tensor Core ops
            "smsp__sass_thread_inst_executed_op_dmma_pred_on.sum",  # MMA ops (Blackwell)
        ]
        
        cmd = [
            "ncu",
            "--metrics", ",".join(metrics),
            "--kernel-name-base", "demangled",
            f"--export={output_name}",
            "--force-overwrite",
        ]
        
        if kernel_filter:
            cmd.extend(["--kernel-name", kernel_filter])
        
        cmd.extend(["python", script_path])
        
        return " ".join(cmd)
    
    @staticmethod
    def analyze_blackwell_kernel(report_path: str):
        """
        Analyze Blackwell-specific kernel metrics
        """
        print(f"=== Nsight Compute Analysis: {report_path} ===\n")
        
        print("Critical Metrics for Blackwell:")
        print("\n1. Compute Throughput:")
        print("   - FP8 Tensor Cores: Target >1000 TFLOPS")
        print("   - FP16: Target >600 TFLOPS")
        print("   - tcgen05.mma utilization: Target >70%")
        
        print("\n2. Memory Throughput:")
        print("   - HBM3e: Target >7 TB/s (>90% of 7.8 TB/s peak)")
        print("   - L2 Cache hit rate: Target >80%")
        print("   - Check for 256-byte burst access patterns")
        
        print("\n3. Warp Efficiency:")
        print("   - Active warps: Target >80%")
        print("   - Branch divergence: <10%")
        print("   - Memory coalescing: >90%")
        
        print("\n4. Occupancy:")
        print("   - Theoretical: Depends on kernel")
        print("   - Achieved: Target >75% of theoretical")
        
        print("\n5. Blackwell-Specific:")
        print("   - Thread Block Clusters: Check if used (up to 8 CTAs)")
        print("   - Distributed Shared Memory: Check DSMEM usage")
        print("   - TMA: Check async copy efficiency")
        
        print("\nVisualization:")
        print(f"  Open in Nsight Compute: ncu-ui {report_path}")


# ============================================================================
# 3. PyTorch Profiler Integration
# ============================================================================

def profile_with_pytorch_profiler(
    fn: callable,
    output_dir: str = "./profiling_results",
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = True,
):
    """
    Profile with PyTorch's built-in profiler
    
    Features:
    - CPU and GPU time
    - Memory usage
    - Operator-level breakdown
    - TensorBoard integration
    
    Args:
        fn: Function to profile
        output_dir: Output directory
        record_shapes: Record tensor shapes
        profile_memory: Track memory usage
        with_stack: Include Python stack traces
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"=== PyTorch Profiler ===")
    print(f"Output directory: {output_dir}\n")
    
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        on_trace_ready=profiler.tensorboard_trace_handler(output_dir),
        schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    ) as prof:
        for _ in range(5):  # warmup + active steps
            fn()
            prof.step()
    
    # Print summary
    print("\n" + "=" * 80)
    print("Top 10 GPU operators:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10,
    ))
    
    print("\n" + "=" * 80)
    print("Top 10 memory consumers:")
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage",
        row_limit=10,
    ))
    
    print("\n" + "=" * 80)
    print(f"TensorBoard trace saved to: {output_dir}")
    print(f"View with: tensorboard --logdir={output_dir}")


# ============================================================================
# 4. Complete Profiling Workflow
# ============================================================================

def complete_profiling_workflow(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    output_dir: str = "./profiling_blackwell",
):
    """
    Complete profiling workflow for Blackwell
    
    Steps:
    1. PyTorch profiler for high-level overview
    2. Nsight Systems for timeline
    3. Nsight Compute for kernel details
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        output_dir: Output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Complete Profiling Workflow for Blackwell")
    print("=" * 80)
    
    # 1. PyTorch Profiler
    print("\n Step 1: PyTorch Profiler (high-level overview)")
    def run_model():
        with torch.no_grad():
            _ = model(input_tensor)
    
    profile_with_pytorch_profiler(
        run_model,
        output_dir=f"{output_dir}/pytorch_profiler",
    )
    
    # 2. Nsight Systems command
    print("\n\nStep 2: Nsight Systems (timeline analysis)")
    print("Run this command manually:")
    print("-" * 80)
    nsys_cmd = NsightSystemsProfiler.profile_command(
        "your_script.py",
        f"{output_dir}/nsys_trace",
    )
    print(nsys_cmd)
    print("-" * 80)
    
    # 3. Nsight Compute command
    print("\n\nStep 3: Nsight Compute (kernel-level profiling)")
    print("Run this command manually:")
    print("-" * 80)
    ncu_cmd = NsightComputeProfiler.profile_kernel_command(
        "your_script.py",
        f"{output_dir}/ncu_report",
    )
    print(ncu_cmd)
    print("-" * 80)
    
    print("\n\n" + "=" * 80)
    print("Profiling Workflow Complete!")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Review PyTorch Profiler in TensorBoard")
    print("2. Run Nsight Systems command for timeline")
    print("3. Run Nsight Compute command for kernel details")
    print("4. Compare against Blackwell targets:")
    print("   - SM utilization: >80%")
    print("   - Memory BW: >7 TB/s")
    print("   - Tensor Core util: >70%")
    print("   - FP8 TFLOPS: >1000")


# ============================================================================
# 5. Quick Reference Commands
# ============================================================================

def print_quick_reference():
    """Print quick reference for profiling commands"""
    print("=" * 80)
    print("Blackwell Profiling Quick Reference")
    print("=" * 80)
    
    print("\n1. Nsight Systems (Timeline):")
    print("   nsys profile --trace=cuda,nvtx,cudnn,cublas \\")
    print("     --cuda-memory-usage=true \\")
    print("     --output=trace python script.py")
    
    print("\n2. Nsight Compute (Kernel Details):")
    print("   ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\\")
    print("     dram__throughput.avg.pct_of_peak_sustained_elapsed \\")
    print("     --export=report python script.py")
    
    print("\n3. PyTorch Profiler (Python-level):")
    print("   # Use profile_with_pytorch_profiler() from this module")
    
    print("\n4. Key Blackwell Metrics:")
    print("   - SM utilization: >80% (148 SMs)")
    print("   - HBM3e bandwidth: >7 TB/s (>90% of 7.8 TB/s)")
    print("   - Tensor Core (tcgen05): >70% utilization")
    print("   - FP8: >1000 TFLOPS, FP16: >600 TFLOPS")
    
    print("\n5. Analysis Tools:")
    print("   - nsys-ui: Nsight Systems GUI")
    print("   - ncu-ui: Nsight Compute GUI")
    print("   - tensorboard: PyTorch Profiler visualization")
    
    print("=" * 80)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=== Blackwell Profiling Guide ===\n")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("  CUDA not available")
        exit(1)
    
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    
    # Print quick reference
    print_quick_reference()
    
    print("\n\n" + "=" * 80)
    print("Example: Profile a simple model")
    print("=" * 80)
    
    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 1024),
    ).cuda()
    
    # Test input
    x = torch.randn(32, 1024, device="cuda")
    
    # Profile with PyTorch profiler
    def run_model():
        with torch.no_grad():
            _ = model(x)
    
    profile_with_pytorch_profiler(
        run_model,
        output_dir="./example_profiling",
    )
    
    print("\n Profiling example complete!")
    print("Check ./example_profiling for results")

