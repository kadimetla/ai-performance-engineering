#!/usr/bin/env python3
"""Blackwell-only architecture helpers for AI Performance Engineering."""

from typing import Any, Dict
import os
import torch

BLACKWELL_CC = "10.0"

class ArchitectureConfig:
    """Provide configuration details for NVIDIA Blackwell GPUs."""

    def __init__(self) -> None:
        self.arch = self._detect_architecture()
        self.config = self._get_architecture_config()

    def _detect_architecture(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        props = torch.cuda.get_device_properties(0)
        compute_capability = f"{props.major}.{props.minor}"
        return "blackwell" if compute_capability == BLACKWELL_CC else "other"

    def _get_architecture_config(self) -> Dict[str, Any]:
        blackwell = {
            "name": "Blackwell B200/B300",
            "compute_capability": BLACKWELL_CC,
            "sm_version": "sm_100",
            "memory_bandwidth": "7.8 TB/s",
            "tensor_cores": "5th Gen",
            "features": ["HBM3e", "TMA", "NVLink-C2C", "Stream-ordered Memory"],
            "cuda_features": ["Stream-ordered Memory", "TMA", "HBM3e optimisations", "NVLink-C2C"],
            "pytorch_optimizations": [
                "torch.compile with max-autotune",
                "TMA-aware kernels",
                "HBM3e-aware allocation",
                "Stream-ordered memory APIs",
                "NVLink-C2C communication"
            ],
            "triton_features": [
                "Triton 3.5 Blackwell optimisations",
                "HBM3e access patterns",
                "TMA intrinsic support",
                "Stream-ordered memory",
                "Blackwell-tuned kernels"
            ],
            "profiling_tools": [
                "Nsight Systems 2025.x",
                "Nsight Compute 2025.x",
                "HTA",
                "PyTorch Profiler",
                "perf"
            ],
        }
        generic = {
            "name": "Generic CUDA GPU",
            "compute_capability": "Unknown",
            "sm_version": "sm_unknown",
            "memory_bandwidth": "Unknown",
            "tensor_cores": "Unknown",
            "features": [],
            "cuda_features": [],
            "pytorch_optimizations": [],
            "triton_features": [],
            "profiling_tools": [],
        }
        return blackwell if self.arch == "blackwell" else generic

    def get_sm_version(self) -> str:
        return self.config["sm_version"]

    def get_architecture_name(self) -> str:
        return self.config["name"]

    def get_features(self) -> list:
        return self.config["features"]

    def get_cuda_features(self) -> list:
        return self.config["cuda_features"]

    def get_pytorch_optimizations(self) -> list:
        return self.config["pytorch_optimizations"]

    def get_triton_features(self) -> list:
        return self.config["triton_features"]

    def get_profiling_tools(self) -> list:
        return self.config["profiling_tools"]

    def configure_pytorch_optimizations(self) -> None:
        if not torch.cuda.is_available():
            return
        
        # PyTorch Inductor configuration
        inductor = getattr(torch, "_inductor", None)
        if inductor and hasattr(inductor, "config"):
            cfg = inductor.config
            # Enable PyTorch 2.9 features
            if hasattr(cfg, "triton"):
                triton_cfg = cfg.triton
                if hasattr(triton_cfg, "unique_kernel_names"):
                    triton_cfg.unique_kernel_names = True
                # NEW in PyTorch 2.9: CUDA graph trees for better performance
                if hasattr(triton_cfg, "cudagraph_trees"):
                    triton_cfg.cudagraph_trees = True
                if hasattr(triton_cfg, "cudagraphs"):
                    triton_cfg.cudagraphs = True
            
            # Enable max-autotune GEMM backends (PyTorch 2.9)
            if hasattr(cfg, "max_autotune_gemm_backends"):
                cfg.max_autotune_gemm_backends = "TRITON,CUTLASS,ATen"
        
        # Triton 3.5 configuration for Blackwell
        if self.arch == "blackwell":
            try:
                import triton
                # Configure Triton 3.5 for Blackwell (SM 10.0)
                if hasattr(triton.runtime, "driver"):
                    triton.runtime.driver.set_active_device_capability(10, 0)
            except (ImportError, AttributeError):
                pass
            
            # Blackwell-specific environment variables
            os.environ.setdefault("TRITON_CUDNN_ALGOS", "1")
            os.environ.setdefault("TRITON_TMA_ENABLE", "1")
            os.environ.setdefault("TRITON_ALWAYS_COMPILE", "0")  # Use kernel cache
        
        # PyTorch 2.9: Enable FlashAttention-3 for Blackwell
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)  # Disable slow fallback
        
        # Standard CUDA configurations
        os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
        os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "0")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
        
        # PyTorch 2.9: Enable TF32 for Blackwell (improves FP32 matmul performance)
# NEW PyTorch 2.9 API (no warnings!)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'high'

    def print_info(self) -> None:
        cfg = self.config
        print(f"Architecture: {cfg['name']}")
        print(f"Compute Capability: {cfg['compute_capability']}")
        print(f"SM Version: {cfg['sm_version']}")
        print(f"Memory Bandwidth: {cfg['memory_bandwidth']}")
        print(f"Tensor Cores: {cfg['tensor_cores']}")
        if cfg['features']:
            print(f"Features: {', '.join(cfg['features'])}")
        if cfg['cuda_features']:
            print(f"CUDA Features: {', '.join(cfg['cuda_features'])}")
        if cfg['pytorch_optimizations']:
            print(f"PyTorch Optimisations: {', '.join(cfg['pytorch_optimizations'])}")
        if cfg['triton_features']:
            print(f"Triton Features: {', '.join(cfg['triton_features'])}")
        if cfg['profiling_tools']:
            print(f"Profiling Tools: {', '.join(cfg['profiling_tools'])}")

def configure_optimizations() -> None:
    ArchitectureConfig().configure_pytorch_optimizations()

arch_config = ArchitectureConfig()
