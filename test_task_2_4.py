"""
Quick test to verify task 2.4 implementation.
"""
import torch
from src.models.transnnmil import TransnnMIL


def test_task_2_4():
    """Test that task 2.4 is correctly implemented."""
    print("Testing task 2.4: Apply fusion classifier and return results")
    
    # Create model
    model = TransnnMIL(
        feature_dim=1024,
        hidden_dim=256,
        num_classes=2,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
    )
    model.eval()
    
    # Create dummy input
    batch_size = 4
    num_patches_val = 100
    features = torch.randn(batch_size, num_patches_val, 1024)
    num_patches = torch.tensor([100, 80, 90, 100])
    
    print(f"\nInput shape: {features.shape}")
    print(f"Num patches: {num_patches}")
    
    # Test 1: Forward pass with return_attention=False
    print("\n--- Test 1: return_attention=False ---")
    logits = model(features, num_patches, return_attention=False)
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {logits.shape}"
    assert isinstance(logits, torch.Tensor), "Expected logits to be a tensor"
    print("✓ Logits shape is correct")
    print("✓ Returns only logits (not tuple)")
    
    # Test 2: Forward pass with return_attention=True
    print("\n--- Test 2: return_attention=True ---")
    result = model(features, num_patches, return_attention=True)
    assert isinstance(result, tuple), "Expected tuple when return_attention=True"
    assert len(result) == 2, f"Expected tuple of length 2, got {len(result)}"
    
    logits, attention = result
    print(f"Output logits shape: {logits.shape}")
    print(f"Attention weights shape: {attention.shape}")
    
    assert logits.shape == (batch_size, 2), f"Expected logits shape ({batch_size}, 2), got {logits.shape}"
    assert attention.shape == (batch_size, num_patches_val), \
        f"Expected attention shape ({batch_size}, {num_patches_val}), got {attention.shape}"
    print("✓ Logits shape is correct")
    print("✓ Attention weights shape is correct")
    print("✓ Returns tuple (logits, attention_weights)")
    
    # Test 3: Different batch sizes
    print("\n--- Test 3: Different batch sizes ---")
    for bs in [1, 8, 16]:
        features_test = torch.randn(bs, 50, 1024)
        num_patches_test = torch.full((bs,), 50)
        
        logits_test = model(features_test, num_patches_test, return_attention=False)
        assert logits_test.shape == (bs, 2), \
            f"Batch size {bs}: Expected shape ({bs}, 2), got {logits_test.shape}"
        print(f"✓ Batch size {bs}: shape {logits_test.shape} is correct")
    
    # Test 4: Different num_classes
    print("\n--- Test 4: Different num_classes ---")
    for num_classes in [2, 3, 5]:
        model_test = TransnnMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=num_classes,
        )
        model_test.eval()
        
        features_test = torch.randn(4, 50, 1024)
        logits_test = model_test(features_test, return_attention=False)
        assert logits_test.shape == (4, num_classes), \
            f"num_classes={num_classes}: Expected shape (4, {num_classes}), got {logits_test.shape}"
        print(f"✓ num_classes={num_classes}: shape {logits_test.shape} is correct")
    
    print("\n" + "="*60)
    print("✓ All tests passed! Task 2.4 is correctly implemented.")
    print("="*60)


if __name__ == "__main__":
    test_task_2_4()
