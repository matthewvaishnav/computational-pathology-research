#!/usr/bin/env python3
"""
GPU Check Script

This script checks GPU availability and provides detailed information.
"""

import torch
import sys

def check_gpu_detailed():
    """Check GPU availability with detailed information."""
    print("🔍 GPU AVAILABILITY CHECK")
    print("=" * 50)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            
            # Test GPU memory allocation
            try:
                test_tensor = torch.randn(1000, 1000).cuda(i)
                print(f"  Memory Test: ✅ PASSED")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Memory Test: ❌ FAILED - {e}")
    else:
        print("\n❌ No GPU detected!")
        print("Possible reasons:")
        print("1. CUDA not installed")
        print("2. PyTorch CPU-only version installed")
        print("3. GPU drivers not installed")
        print("4. GPU not compatible")
        
        # Check if this is a CPU-only PyTorch installation
        try:
            import torch.version
            if '+cpu' in torch.__version__:
                print("\n⚠️  CPU-only PyTorch detected!")
                print("Install GPU version with:")
                print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        except:
            pass
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_gpu_detailed()