"""Comprehensive test for task 2.1 implementation."""
import torch
from src.models.transnnmil import TransnnMIL


def test_feature_extraction_basic():
    """Test basic feature extraction."""
    print("=" * 60)
    print("Test 1: Basic Feature Extraction")
    print("=" * 60)
    
    model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    model.eval()
    
    # Create dummy input
    batch_size = 4
    num_patches_val = 100
    features = torch.randn(batch_size, num_patches_val, 1024)
    num_patches = torch.tensor([100, 80, 90, 100])
    
    # Test without return_attention
    logits = model(features, num_patches, return_attention=False)
    assert logits.shape == (batch_size, 2), f"Expected shape (4, 2), got {logits.shape}"
    print(f"✓ Forward pass without attention: logits shape = {logits.shape}")
    
    # Test with return_attention
    logits, attention = model(features, num_patches, return_attention=True)
    assert logits.shape == (batch_size, 2), f"Expected logits shape (4, 2), got {logits.shape}"
    assert attention.shape == (batch_size, num_patches_val), f"Expected attention shape (4, 100), got {attention.shape}"
    print(f"✓ Forward pass with attention: logits shape = {logits.shape}, attention shape = {attention.shape}")
    
    print()


def test_get_features_directly():
    """Test calling get_features methods directly."""
    print("=" * 60)
    print("Test 2: Direct get_features() Calls")
    print("=" * 60)
    
    model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    model.eval()
    
    batch_size = 4
    num_patches_val = 100
    features = torch.randn(batch_size, num_patches_val, 1024)
    num_patches = torch.tensor([100, 80, 90, 100])
    
    # Test get_features methods
    features_a = model.branch_a.get_features(features, num_patches)
    features_b = model.branch_b.get_features(features, num_patches)
    
    assert features_a.shape == (batch_size, 256), f"Expected shape (4, 256), got {features_a.shape}"
    assert features_b.shape == (batch_size, 1024), f"Expected shape (4, 1024), got {features_b.shape}"
    
    print(f"✓ Branch A features shape: {features_a.shape} (expected: [4, 256])")
    print(f"✓ Branch B features shape: {features_b.shape} (expected: [4, 1024])")
    print()


def test_variable_batch_sizes():
    """Test with different batch sizes."""
    print("=" * 60)
    print("Test 3: Variable Batch Sizes")
    print("=" * 60)
    
    model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    model.eval()
    
    for batch_size in [1, 2, 8, 16]:
        num_patches_val = 100
        features = torch.randn(batch_size, num_patches_val, 1024)
        num_patches = torch.tensor([num_patches_val] * batch_size)
        
        # Test forward pass
        logits = model(features, num_patches, return_attention=False)
        assert logits.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {logits.shape}"
        
        # Test get_features
        features_a = model.branch_a.get_features(features, num_patches)
        features_b = model.branch_b.get_features(features, num_patches)
        
        assert features_a.shape == (batch_size, 256), f"Expected shape ({batch_size}, 256), got {features_a.shape}"
        assert features_b.shape == (batch_size, 1024), f"Expected shape ({batch_size}, 1024), got {features_b.shape}"
        
        print(f"✓ Batch size {batch_size}: All shapes correct")
    
    print()


def test_variable_num_patches():
    """Test with variable number of patches per bag."""
    print("=" * 60)
    print("Test 4: Variable Number of Patches")
    print("=" * 60)
    
    model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    model.eval()
    
    batch_size = 4
    max_patches = 200
    features = torch.randn(batch_size, max_patches, 1024)
    
    # Different actual patch counts
    num_patches = torch.tensor([200, 150, 100, 50])
    
    # Test forward pass
    logits = model(features, num_patches, return_attention=False)
    assert logits.shape == (batch_size, 2), f"Expected shape (4, 2), got {logits.shape}"
    print(f"✓ Variable patches: logits shape = {logits.shape}")
    
    # Test with return_attention
    logits, attention = model(features, num_patches, return_attention=True)
    assert logits.shape == (batch_size, 2), f"Expected logits shape (4, 2), got {logits.shape}"
    assert attention.shape == (batch_size, max_patches), f"Expected attention shape (4, 200), got {attention.shape}"
    print(f"✓ Variable patches with attention: attention shape = {attention.shape}")
    
    # Test get_features
    features_a = model.branch_a.get_features(features, num_patches)
    features_b = model.branch_b.get_features(features, num_patches)
    
    assert features_a.shape == (batch_size, 256), f"Expected shape (4, 256), got {features_a.shape}"
    assert features_b.shape == (batch_size, 1024), f"Expected shape (4, 1024), got {features_b.shape}"
    print(f"✓ Variable patches: features extracted correctly")
    
    print()


def test_without_num_patches():
    """Test with num_patches=None (fixed-length bags)."""
    print("=" * 60)
    print("Test 5: Fixed-Length Bags (num_patches=None)")
    print("=" * 60)
    
    model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    model.eval()
    
    batch_size = 4
    num_patches_val = 100
    features = torch.randn(batch_size, num_patches_val, 1024)
    
    # Test without num_patches
    logits = model(features, num_patches=None, return_attention=False)
    assert logits.shape == (batch_size, 2), f"Expected shape (4, 2), got {logits.shape}"
    print(f"✓ Fixed-length bags: logits shape = {logits.shape}")
    
    # Test get_features without num_patches
    features_a = model.branch_a.get_features(features, num_patches=None)
    features_b = model.branch_b.get_features(features, num_patches=None)
    
    assert features_a.shape == (batch_size, 256), f"Expected shape (4, 256), got {features_a.shape}"
    assert features_b.shape == (batch_size, 1024), f"Expected shape (4, 1024), got {features_b.shape}"
    print(f"✓ Fixed-length bags: features extracted correctly")
    
    print()


def test_attention_weights_validity():
    """Test that attention weights are valid probabilities."""
    print("=" * 60)
    print("Test 6: Attention Weights Validity")
    print("=" * 60)
    
    model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    model.eval()
    
    batch_size = 4
    num_patches_val = 100
    features = torch.randn(batch_size, num_patches_val, 1024)
    num_patches = torch.tensor([100, 80, 90, 100])
    
    # Get attention weights
    logits, attention = model(features, num_patches, return_attention=True)
    
    # Check attention weights are non-negative
    assert (attention >= 0).all(), "Attention weights should be non-negative"
    print(f"✓ Attention weights are non-negative")
    
    # Check attention weights sum to approximately 1 for each bag
    # (accounting for masking of padded patches)
    for i in range(batch_size):
        valid_patches = num_patches[i].item()
        attention_sum = attention[i, :valid_patches].sum().item()
        assert abs(attention_sum - 1.0) < 1e-5, f"Attention weights should sum to 1, got {attention_sum}"
    
    print(f"✓ Attention weights sum to 1 for valid patches")
    print()


def test_features_are_different():
    """Test that Branch A and Branch B produce different features."""
    print("=" * 60)
    print("Test 7: Branch Features Are Different")
    print("=" * 60)
    
    model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    model.eval()
    
    batch_size = 4
    num_patches_val = 100
    features = torch.randn(batch_size, num_patches_val, 1024)
    num_patches = torch.tensor([100, 80, 90, 100])
    
    # Get features from both branches
    features_a = model.branch_a.get_features(features, num_patches)
    features_b = model.branch_b.get_features(features, num_patches)
    
    # Check that features have different dimensions
    assert features_a.shape[1] != features_b.shape[1], "Branch features should have different dimensions"
    print(f"✓ Branch A: {features_a.shape[1]}-dim, Branch B: {features_b.shape[1]}-dim")
    
    # Check that features are not all zeros
    assert not torch.allclose(features_a, torch.zeros_like(features_a)), "Branch A features should not be all zeros"
    assert not torch.allclose(features_b, torch.zeros_like(features_b)), "Branch B features should not be all zeros"
    print(f"✓ Both branches produce non-zero features")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TASK 2.1 VERIFICATION")
    print("=" * 60 + "\n")
    
    try:
        test_feature_extraction_basic()
        test_get_features_directly()
        test_variable_batch_sizes()
        test_variable_num_patches()
        test_without_num_patches()
        test_attention_weights_validity()
        test_features_are_different()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nTask 2.1 implementation is fully verified:")
        print("  ✓ Feature extraction from both branches")
        print("  ✓ Attention weight handling")
        print("  ✓ Variable batch sizes")
        print("  ✓ Variable number of patches")
        print("  ✓ Fixed-length bags")
        print("  ✓ Attention weight validity")
        print("  ✓ Branch feature differentiation")
        print()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
