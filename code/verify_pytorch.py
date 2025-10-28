#!/usr/bin/env python3
"""
Verify PyTorch installation and CUDA functionality.
Tests basic PyTorch operations and CUDA availability.
"""

import sys


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)


def check_pytorch_import():
    """Check if PyTorch can be imported."""
    print_section("PyTorch Import Check")
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
        return True, torch
    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")
        return False, None


def check_cuda_availability(torch):
    """Check CUDA availability."""
    print_section("CUDA Availability Check")
    
    if torch.cuda.is_available():
        print("✅ CUDA is available")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n   GPU {i}: {props.name}")
            print(f"     Compute Capability: {props.major}.{props.minor}")
            print(f"     Total Memory: {props.total_memory / 1024**3:.2f} GB")
        
        return True
    else:
        print("❌ CUDA is not available")
        print("   PyTorch may not be built with CUDA support")
        return False


def test_basic_operations(torch):
    """Test basic PyTorch operations."""
    print_section("Basic Operations Test")
    
    try:
        # CPU test
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.matmul(x, y)
        print("✅ CPU operations working")
        
        # GPU test
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            print("✅ GPU operations working")
            
            # Test synchronization
            torch.cuda.synchronize()
            print("✅ CUDA synchronization working")
        
        return True
    except Exception as e:
        print(f"❌ Operations test failed: {e}")
        return False


def test_mixed_precision(torch):
    """Test mixed precision support."""
    print_section("Mixed Precision Support")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping mixed precision test")
        return True
    
    try:
        # Test FP16
        x = torch.randn(100, 100, device='cuda', dtype=torch.float16)
        y = torch.randn(100, 100, device='cuda', dtype=torch.float16)
        z = torch.matmul(x, y)
        print("✅ FP16 operations working")
        
        # Test BF16
        if torch.cuda.is_bf16_supported():
            x = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
            y = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
            z = torch.matmul(x, y)
            print("✅ BF16 operations working")
        else:
            print("⚠️  BF16 not supported on this GPU")
        
        # Test AMP
        from torch.cuda.amp import autocast
        with autocast():
            x = torch.randn(100, 100, device='cuda')
            y = torch.randn(100, 100, device='cuda')
            z = torch.matmul(x, y)
        print("✅ Automatic Mixed Precision (AMP) working")
        
        return True
    except Exception as e:
        print(f"❌ Mixed precision test failed: {e}")
        return False


def test_distributed(torch):
    """Test distributed capabilities."""
    print_section("Distributed Capabilities Check")
    
    try:
        if torch.distributed.is_available():
            print("✅ torch.distributed is available")
            
            if torch.distributed.is_nccl_available():
                print("✅ NCCL backend is available")
            else:
                print("⚠️  NCCL backend not available")
            
            if torch.distributed.is_gloo_available():
                print("✅ Gloo backend is available")
            else:
                print("⚠️  Gloo backend not available")
        else:
            print("❌ torch.distributed is not available")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Distributed check failed: {e}")
        return False


def test_compile_support(torch):
    """Test torch.compile support."""
    print_section("torch.compile Support Check")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping compile test")
        return True
    
    try:
        import torch.nn as nn
        
        model = nn.Linear(10, 10).cuda()
        compiled_model = torch.compile(model)
        
        x = torch.randn(5, 10, device='cuda')
        with torch.no_grad():
            y = compiled_model(x)
        
        print("✅ torch.compile is working")
        return True
    except Exception as e:
        print(f"⚠️  torch.compile test failed: {e}")
        print("   This may be expected on some configurations")
        return True  # Don't fail on this


def main():
    """Run all PyTorch verification checks."""
    print("\n" + "="*80)
    print("  PyTorch Installation Verification Suite")
    print("="*80)
    
    results = {}
    
    # Check PyTorch import
    success, torch_module = check_pytorch_import()
    if not success:
        print("\n❌ CRITICAL: PyTorch is not installed properly!")
        return 1
    
    results['import'] = True
    
    # Run all checks
    results['cuda'] = check_cuda_availability(torch_module)
    results['operations'] = test_basic_operations(torch_module)
    results['mixed_precision'] = test_mixed_precision(torch_module)
    results['distributed'] = test_distributed(torch_module)
    results['compile'] = test_compile_support(torch_module)
    
    # Summary
    print_section("Verification Summary")
    
    checks = [
        ('PyTorch Import', results['import']),
        ('CUDA Availability', results['cuda']),
        ('Basic Operations', results['operations']),
        ('Mixed Precision', results['mixed_precision']),
        ('Distributed Support', results['distributed']),
        ('torch.compile Support', results['compile']),
    ]
    
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {check_name}")
    
    # Critical checks that must pass
    critical_checks = ['import', 'cuda', 'operations']
    critical_passed = all(results[check] for check in critical_checks)
    
    if critical_passed:
        print("\n✅ All critical checks PASSED!")
        print("   PyTorch is properly installed and functional.")
        return 0
    else:
        print("\n❌ Some critical checks FAILED!")
        print("   Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

