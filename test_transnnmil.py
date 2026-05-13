"""
Quick test script for TransnnMIL model.

This script verifies that TransnnMIL:
1. Can be instantiated correctly
2. Processes input bags and returns correct output shapes
3. Works with the factory pattern
4. Returns attention weights when requested
5. Provides gate value inspection
"""

import torch
from src.models.transnnmil import TransnnMIL
from src.models.factory import create_attention_model


def test_direct_instantiation():
    """Test creating TransnnMIL directly."""
    print("=" * 60)
    print("Test 1: Direct Instantiation")
    print("=" * 60)
    
    model = TransnnMIL(
        feature_dim=1024,
        hidden_dim=256,
        num_classes=2,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        use_pos_encoding=False
    )
    
    print(f"✓ Model created successfully")
    print(f"  - Feature dim: {model.feature_dim}")
    print(f"  - Hidden dim: {model.hidden_dim}")
    print(f"  - Num classes: {model.num_classes}")
    print(f"  - Initial gate value: {model.get_gate_value():.4f}")
    print()


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("=" * 60)
    print("Test 2: Forward Pass")
    print("=" * 60)
    
    model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    model.eval()
    
    # Create dummy batch
    batch_size = 4
    num_patches = 100
    feature_dim = 1024
    
    features = torch.randn(batch_size, num_patches, feature_dim)
    patch_counts = torch.tensor([100, 80, 90, 100])
    
    # Forward pass without attention
    with torch.no_grad():
        logits = model(features, patch_counts, return_attention=False)
    
    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {features.shape}")
    print(f"  - Output shape: {logits.shape}")
    print(f"  - Expected shape: ({batch_size}, {model.num_classes})")
    assert logits.shape == (batch_size, model.num_classes), "Output shape mismatch!"
    print()


def test_attention_return():
    """Test returning attention weights."""
    print("=" * 60)
    print("Test 3: Attention Weights")
    print("=" * 60)
    
    model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    model.eval()
    
    batch_size = 4
    num_patches = 100
    features = torch.randn(batch_size, num_patches, 1024)
    patch_counts = torch.tensor([100, 80, 90, 100])
    
    # Forward pass with attention
    with torch.no_grad():
        logits, attention = model(features, patch_counts, return_attention=True)
    
    print(f"✓ Attention weights returned successfully")
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - Attention shape: {attention.shape}")
    print(f"  - Expected attention shape: ({batch_size}, {num_patches})")
    assert attention.shape == (batch_size, num_patches), "Attention shape mismatch!"
    
    # Check attention weights sum to 1 (approximately)
    attention_sums = attention.sum(dim=1)
    print(f"  - Attention sums: {attention_sums.tolist()}")
    print()


def test_factory_creation():
    """Test creating TransnnMIL via factory."""
    print("=" * 60)
    print("Test 4: Factory Creation")
    print("=" * 60)
    
    config = {
        'model_type': 'transnnmil',
        'hidden_dim': 256,
        'num_classes': 2,
        'dropout': 0.1,
        'transnnmil': {
            'num_layers': 2,
            'num_heads': 8,
            'use_pos_encoding': False
        }
    }
    
    model = create_attention_model(config, feature_dim=1024)
    
    print(f"✓ Model created via factory")
    print(f"  - Model type: {type(model).__name__}")
    assert isinstance(model, TransnnMIL), "Factory didn't create TransnnMIL!"
    
    # Test forward pass
    features = torch.randn(2, 50, 1024)
    patch_counts = torch.tensor([50, 40])
    
    with torch.no_grad():
        logits = model(features, patch_counts)
    
    print(f"  - Forward pass successful: {logits.shape}")
    print()


def test_branch_outputs():
    """Test getting individual branch outputs."""
    print("=" * 60)
    print("Test 5: Branch Outputs")
    print("=" * 60)
    
    model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    model.eval()
    
    features = torch.randn(2, 50, 1024)
    patch_counts = torch.tensor([50, 40])
    
    with torch.no_grad():
        logits_a, logits_b, logits_fused = model.get_branch_outputs(features, patch_counts)
    
    print(f"✓ Branch outputs retrieved successfully")
    print(f"  - Branch A (TransMIL) shape: {logits_a.shape}")
    print(f"  - Branch B (nnMIL) shape: {logits_b.shape}")
    print(f"  - Fused output shape: {logits_fused.shape}")
    
    # Check predictions
    preds_a = logits_a.argmax(dim=1)
    preds_b = logits_b.argmax(dim=1)
    preds_fused = logits_fused.argmax(dim=1)
    
    print(f"  - Branch A predictions: {preds_a.tolist()}")
    print(f"  - Branch B predictions: {preds_b.tolist()}")
    print(f"  - Fused predictions: {preds_fused.tolist()}")
    
    gate = model.get_gate_value()
    print(f"  - Gate value: {gate:.4f} (TransMIL weight)")
    print(f"  - Complement: {1-gate:.4f} (nnMIL weight)")
    print()


def test_gate_learning():
    """Test that gate parameter is learnable."""
    print("=" * 60)
    print("Test 6: Gate Parameter Learning")
    print("=" * 60)
    
    model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
    
    # Check gate is a parameter
    print(f"✓ Gate parameter exists")
    print(f"  - Initial value (raw): {model.gate_param.item():.4f}")
    print(f"  - Initial value (sigmoid): {model.get_gate_value():.4f}")
    print(f"  - Requires grad: {model.gate_param.requires_grad}")
    
    # Simulate one gradient update
    features = torch.randn(2, 50, 1024)
    patch_counts = torch.tensor([50, 40])
    target = torch.tensor([0, 1])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Forward pass
    logits = model(features, patch_counts)
    loss = criterion(logits, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gate has gradients
    print(f"  - Gate gradient: {model.gate_param.grad.item():.6f}")
    assert model.gate_param.grad is not None, "Gate parameter has no gradient!"
    
    # Update
    old_gate = model.get_gate_value()
    optimizer.step()
    new_gate = model.get_gate_value()
    
    print(f"  - Gate after update: {new_gate:.4f}")
    print(f"  - Change: {new_gate - old_gate:.6f}")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TransnnMIL Model Tests")
    print("=" * 60 + "\n")
    
    try:
        test_direct_instantiation()
        test_forward_pass()
        test_attention_return()
        test_factory_creation()
        test_branch_outputs()
        test_gate_learning()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nTransnnMIL is ready for training!")
        print("\nUsage example:")
        print("  python train.py --model transnnmil --dataset pcam")
        print()
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

