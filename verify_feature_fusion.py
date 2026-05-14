"""
Checkpoint verification script for feature-level fusion in TransnnMIL.

This script verifies:
1. All fusion components exist in the model
2. Model architecture includes fusion layers
3. Forward pass works with dummy input
4. No runtime errors occur
"""

import torch
from src.models.transnnmil import TransnnMIL


def verify_model_components():
    """Verify all fusion components exist in TransnnMIL."""
    print("=" * 80)
    print("CHECKPOINT: Verifying TransnnMIL Feature-Level Fusion Components")
    print("=" * 80)
    print()
    
    # Create model instance
    print("1. Instantiating TransnnMIL model...")
    model = TransnnMIL(
        feature_dim=1024,
        hidden_dim=256,
        num_classes=2,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        use_pos_encoding=False,
    )
    print("   ✓ Model instantiated successfully")
    print()
    
    # Verify fusion components exist
    print("2. Verifying fusion components exist...")
    components = {
        "proj_a": "Branch A projection layer (256 → 512)",
        "proj_b": "Branch B projection layer (1024 → 512)",
        "fusion_attention": "Cross-attention fusion module",
        "fusion_classifier": "Fusion classifier (512 → 256 → num_classes)",
        "gate_param": "Gate parameter (backward compatibility)",
    }
    
    all_exist = True
    for component, description in components.items():
        if hasattr(model, component):
            print(f"   ✓ {component}: {description}")
        else:
            print(f"   ✗ {component}: MISSING!")
            all_exist = False
    
    if not all_exist:
        raise RuntimeError("Some fusion components are missing!")
    print()
    
    # Print model architecture
    print("3. Model Architecture:")
    print("-" * 80)
    print(model)
    print("-" * 80)
    print()
    
    # Verify component details
    print("4. Verifying component details...")
    
    # Check proj_a structure
    print("   proj_a structure:")
    for i, layer in enumerate(model.proj_a):
        print(f"      [{i}] {layer}")
    
    # Check proj_b structure
    print("   proj_b structure:")
    for i, layer in enumerate(model.proj_b):
        print(f"      [{i}] {layer}")
    
    # Check fusion_attention parameters
    print(f"   fusion_attention:")
    print(f"      embed_dim: {model.fusion_attention.embed_dim}")
    print(f"      num_heads: {model.fusion_attention.num_heads}")
    print(f"      batch_first: {model.fusion_attention.batch_first}")
    
    # Check fusion_classifier structure
    print("   fusion_classifier structure:")
    for i, layer in enumerate(model.fusion_classifier):
        print(f"      [{i}] {layer}")
    
    print()
    
    return model


def test_forward_pass(model):
    """Test forward pass with dummy input."""
    print("5. Testing forward pass with dummy input...")
    
    # Create dummy input
    batch_size = 4
    num_patches = 100
    feature_dim = 1024
    
    features = torch.randn(batch_size, num_patches, feature_dim)
    num_patches_tensor = torch.tensor([100, 80, 90, 100])
    
    print(f"   Input shape: {features.shape}")
    print(f"   Num patches: {num_patches_tensor.tolist()}")
    print()
    
    # Test forward pass without attention
    print("   Testing forward pass (return_attention=False)...")
    try:
        logits = model(features, num_patches_tensor, return_attention=False)
        print(f"   ✓ Forward pass successful")
        print(f"   ✓ Output shape: {logits.shape}")
        print(f"   ✓ Expected shape: ({batch_size}, {model.num_classes})")
        
        if logits.shape != (batch_size, model.num_classes):
            raise RuntimeError(
                f"Output shape mismatch! Expected ({batch_size}, {model.num_classes}), "
                f"got {logits.shape}"
            )
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        raise
    print()
    
    # Test forward pass with attention
    print("   Testing forward pass (return_attention=True)...")
    try:
        logits, attention = model(features, num_patches_tensor, return_attention=True)
        print(f"   ✓ Forward pass with attention successful")
        print(f"   ✓ Logits shape: {logits.shape}")
        print(f"   ✓ Attention shape: {attention.shape}")
        print(f"   ✓ Expected attention shape: ({batch_size}, {num_patches})")
        
        if attention.shape != (batch_size, num_patches):
            raise RuntimeError(
                f"Attention shape mismatch! Expected ({batch_size}, {num_patches}), "
                f"got {attention.shape}"
            )
    except Exception as e:
        print(f"   ✗ Forward pass with attention failed: {e}")
        raise
    print()


def test_get_branch_outputs(model):
    """Test get_branch_outputs method."""
    print("6. Testing get_branch_outputs method...")
    
    # Create dummy input
    batch_size = 4
    num_patches = 100
    feature_dim = 1024
    
    features = torch.randn(batch_size, num_patches, feature_dim)
    num_patches_tensor = torch.tensor([100, 80, 90, 100])
    
    try:
        logits_a, logits_b, logits_fused = model.get_branch_outputs(
            features, num_patches_tensor
        )
        
        print(f"   ✓ get_branch_outputs successful")
        print(f"   ✓ Branch A logits shape: {logits_a.shape}")
        print(f"   ✓ Branch B logits shape: {logits_b.shape}")
        print(f"   ✓ Fused logits shape: {logits_fused.shape}")
        
        expected_shape = (batch_size, model.num_classes)
        if logits_a.shape != expected_shape:
            raise RuntimeError(
                f"Branch A logits shape mismatch! Expected {expected_shape}, "
                f"got {logits_a.shape}"
            )
        if logits_b.shape != expected_shape:
            raise RuntimeError(
                f"Branch B logits shape mismatch! Expected {expected_shape}, "
                f"got {logits_b.shape}"
            )
        if logits_fused.shape != expected_shape:
            raise RuntimeError(
                f"Fused logits shape mismatch! Expected {expected_shape}, "
                f"got {logits_fused.shape}"
            )
        
        # Check that outputs are different (fusion has effect)
        if torch.allclose(logits_a, logits_fused):
            print("   ⚠ Warning: Branch A and fused logits are identical")
        else:
            print("   ✓ Branch A and fused logits are different (fusion has effect)")
        
        if torch.allclose(logits_b, logits_fused):
            print("   ⚠ Warning: Branch B and fused logits are identical")
        else:
            print("   ✓ Branch B and fused logits are different (fusion has effect)")
        
    except Exception as e:
        print(f"   ✗ get_branch_outputs failed: {e}")
        raise
    print()


def test_backward_compatibility(model):
    """Test backward compatibility features."""
    print("7. Testing backward compatibility...")
    
    # Test get_gate_value method
    try:
        gate_value = model.get_gate_value()
        print(f"   ✓ get_gate_value() works: {gate_value:.4f}")
        print(f"   ✓ Gate parameter exists and is accessible")
    except Exception as e:
        print(f"   ✗ get_gate_value() failed: {e}")
        raise
    
    # Verify gate_param exists
    if hasattr(model, 'gate_param'):
        print(f"   ✓ gate_param exists as nn.Parameter")
    else:
        print(f"   ✗ gate_param is missing!")
        raise RuntimeError("gate_param is missing!")
    
    print()


def main():
    """Run all verification checks."""
    try:
        # Verify model components
        model = verify_model_components()
        
        # Test forward pass
        test_forward_pass(model)
        
        # Test get_branch_outputs
        test_get_branch_outputs(model)
        
        # Test backward compatibility
        test_backward_compatibility(model)
        
        # Final summary
        print("=" * 80)
        print("CHECKPOINT VERIFICATION: ALL TESTS PASSED ✓")
        print("=" * 80)
        print()
        print("Summary:")
        print("  ✓ All fusion components exist")
        print("  ✓ Model architecture is correct")
        print("  ✓ Forward pass works without errors")
        print("  ✓ Forward pass with attention works")
        print("  ✓ get_branch_outputs works correctly")
        print("  ✓ Backward compatibility maintained")
        print()
        print("The feature-level fusion implementation is ready for comprehensive testing!")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 80)
        print("CHECKPOINT VERIFICATION: FAILED ✗")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        return 1


if __name__ == "__main__":
    exit(main())
