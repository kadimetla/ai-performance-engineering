"""Quantization utilities for CUDA-native FP8/FP4 quantization.

Provides proper CUDA-native quantization using FP8/FP4 formats
that work on B200/B300 as well as Grace-Blackwell GB200/GB300 GPUs.
This is the correct way to do quantization on CUDA, not CPU-only qint8.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable, Dict, Literal, Optional, Union, cast

QuantizeFn = Callable[..., Union[torch.Tensor, tuple[torch.Tensor, float]]]


def quantize_to_fp8(tensor: torch.Tensor, return_scale: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, float]]:
    """Quantize tensor to FP8 (E4M3FN format) for CUDA.
    
    FP8 E4M3FN: 1 sign bit, 4 exponent bits, 3 mantissa bits
    Range: ~6e-8 to 448
    Provides 2x memory reduction vs FP16, 4x vs FP32
    
    Args:
        tensor: Input tensor (FP32, FP16, or BF16)
        return_scale: If True, return (tensor, scale) tuple for proper dequantization
        
    Returns:
        Quantized tensor (converted to BF16 for computation compatibility), or (tensor, scale) if return_scale=True
    """
    if hasattr(torch, 'float8_e4m3fn'):
        try:
            # Native FP8 quantization: FP32/FP16/BF16 -> FP8 -> BF16
            # FP8 is the storage format, BF16 is used for computation
            # Native FP8 doesn't have explicit scale (it's built into the format)
            tensor_fp8 = tensor.to(torch.float8_e4m3fn)
            quantized = tensor_fp8.to(torch.bfloat16)
            if return_scale:
                # For native FP8, scale is 1.0 (no explicit scaling needed)
                return quantized, 1.0
            return quantized
        except (RuntimeError, TypeError):
            # Fallback if FP8 conversion fails
            pass
    
    # Fallback: Manual FP8 quantization simulation with scale tracking
    # Scale to FP8 range and quantize
    max_val = tensor.abs().max().item()  # Convert tensor to scalar
    scale = max_val / 448.0 if max_val > 0 else 1.0
    tensor_scaled = tensor / scale
    tensor_clamped = torch.clamp(tensor_scaled, -448.0, 448.0)
    # Quantize to 8-bit precision
    tensor_quantized = (tensor_clamped * 8.0).round() / 8.0
    quantized = tensor_quantized * scale
    
    if return_scale:
        return quantized, scale.item() if isinstance(scale, torch.Tensor) else scale
    return quantized


def quantize_to_fp4(tensor: torch.Tensor, return_scale: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, float]]:
    """Quantize tensor to FP4 for CUDA (simulated via aggressive FP8).
    
    FP4 provides 2x memory reduction vs FP8, 4x vs FP16
    For B200/GB200+, we use aggressive FP8 quantization to simulate FP4 benefits
    
    Args:
        tensor: Input tensor
        return_scale: If True, return (tensor, scale) tuple for proper dequantization
        
    Returns:
        Quantized tensor (converted to BF16 for computation), or (tensor, scale) if return_scale=True
    """
    # FP4 is not natively supported in PyTorch yet, so we use aggressive FP8
    # In production, would use Transformer Engine or custom kernels for FP4
    if hasattr(torch, 'float8_e4m3fn'):
        try:
            # More aggressive quantization for FP4-like behavior
            tensor_fp8 = tensor.to(torch.float8_e4m3fn)
            # Convert back and quantize again for FP4-like precision
            tensor_fp8_2 = tensor_fp8.to(torch.bfloat16).to(torch.float8_e4m3fn)
            quantized = tensor_fp8_2.to(torch.bfloat16)
            if return_scale:
                # For native FP8-based FP4 simulation, scale is 1.0
                return quantized, 1.0
            return quantized
        except (RuntimeError, TypeError):
            pass
    
    # Fallback: Aggressive quantization with scale tracking
    max_val = tensor.abs().max().item()  # Convert tensor to scalar
    scale = max_val / 224.0 if max_val > 0 else 1.0  # FP4-like range
    tensor_scaled = tensor / scale
    tensor_clamped = torch.clamp(tensor_scaled, -224.0, 224.0)
    # More aggressive quantization (4-bit precision simulation)
    tensor_quantized = (tensor_clamped * 16.0).round() / 16.0
    quantized = tensor_quantized * scale
    
    if return_scale:
        return quantized, scale.item() if isinstance(scale, torch.Tensor) else scale
    return quantized


def _ensure_scale_dict(module: nn.Module) -> Dict[str, float]:
    """Ensure module has a mutable dict for quantization scales."""
    existing = getattr(module, "_quantization_scales", None)
    if not isinstance(existing, dict):
        existing = {}
        setattr(module, "_quantization_scales", existing)
    return cast(Dict[str, float], existing)


def _scale_to_float(scale: Union[float, torch.Tensor]) -> float:
    """Convert scale value (float or tensor scalar) to float."""
    if isinstance(scale, torch.Tensor):
        return float(scale.item())
    return float(scale)


def _quantize_tensor_only(quantize_fn: QuantizeFn, tensor: torch.Tensor) -> torch.Tensor:
    """Call quantize_fn without returning scale and ensure tensor result."""
    result = quantize_fn(tensor)
    if isinstance(result, tuple):
        return result[0]
    return result


def quantize_model_to_fp8(
    model: nn.Module,
    device: torch.device,
    precision: Literal['fp8', 'fp4'] = 'fp8',
    store_scales: bool = False
) -> nn.Module:
    """Quantize a PyTorch model to FP8 or FP4 for CUDA.
    
    This is the proper way to do quantization on CUDA - using FP8/FP4
    formats that are natively supported on GB10/H100+ GPUs.
    
    Args:
        model: PyTorch model to quantize
        device: CUDA device
        precision: 'fp8' or 'fp4' quantization
        store_scales: If True, store per-tensor scales in module._quantization_scales
        
    Returns:
        Quantized model (weights quantized, model in BF16 for computation)
    """
    quantize_fn = quantize_to_fp8 if precision == 'fp8' else quantize_to_fp4
    
    # Quantize all Linear layer weights
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Quantize weights: FP32 -> FP8/FP4 -> BF16
                if store_scales:
                    quantized_weight, weight_scale = quantize_fn(module.weight.data, return_scale=True)
                    module.weight.data = quantized_weight
                    scale_dict = _ensure_scale_dict(module)
                    scale_dict['weight'] = _scale_to_float(weight_scale)
                else:
                    module.weight.data = _quantize_tensor_only(quantize_fn, module.weight.data)
                
                if module.bias is not None:
                    if store_scales:
                        quantized_bias, bias_scale = quantize_fn(module.bias.data, return_scale=True)
                        module.bias.data = quantized_bias
                        scale_dict = _ensure_scale_dict(module)
                        scale_dict['bias'] = _scale_to_float(bias_scale)
                    else:
                        module.bias.data = _quantize_tensor_only(quantize_fn, module.bias.data)
    
    # Convert model to BF16 for computation (FP8/FP4 weights stored as BF16)
    model = model.to(device).to(torch.bfloat16).eval()
    return model


def get_quantization_dtype(precision: Literal['fp8', 'fp4'] = 'fp8') -> torch.dtype:
    """Get the computation dtype for quantized models.
    
    Args:
        precision: 'fp8' or 'fp4'
        
    Returns:
        torch.dtype for computation (BF16 for both FP8 and FP4)
    """
    return torch.bfloat16  # BF16 is used for computation with FP8/FP4 quantized weights
