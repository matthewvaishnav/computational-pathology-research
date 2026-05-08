#!/usr/bin/env python3
"""Debug torch import issues."""

import sys
print(f"Python path: {sys.path[:3]}")

try:
    import torch
    print(f"Torch imported successfully: {torch.__version__}")
    print(f"Has randn: {'randn' in dir(torch)}")
    print(f"Has Tensor: {'Tensor' in dir(torch)}")
    
    # Test basic functionality
    t = torch.randn(2, 3)
    print(f"Created tensor: {t.shape}")
    print(f"Is tensor: {isinstance(t, torch.Tensor)}")
    
except Exception as e:
    print(f"Error importing torch: {e}")
    import traceback
    traceback.print_exc()

# Test validation import
try:
    from src.utils.validation import validate_tensor_shape, ValidationError
    print("Validation module imported successfully")
    
    # Test validation
    t = torch.randn(2, 3)
    validate_tensor_shape(t, (2, 3), "test")
    print("Validation test passed")
    
except Exception as e:
    print(f"Error with validation: {e}")
    import traceback
    traceback.print_exc()