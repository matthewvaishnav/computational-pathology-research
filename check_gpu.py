"""Quick GPU diagnostic without hanging."""
import sys

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA compiled: {torch.version.cuda}")
    
    # Try to check CUDA with timeout protection
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    
    if has_cuda:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available - possible issues:")
        print("1. PyTorch CPU-only version installed")
        print("2. CUDA drivers not installed/updated")
        print("3. CUDA version mismatch")
        
except Exception as e:
    print(f"Error checking GPU: {e}")
    sys.exit(1)
