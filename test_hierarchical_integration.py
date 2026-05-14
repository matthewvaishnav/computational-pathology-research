"""
Test hierarchical pooling integration with TransnnMIL.
"""

import torch
from src.models.transnnmil import TransnnMIL


def test_hierarchical_disabled():
    """Test TransnnMIL without hierarchical pooling (backward compatibility)."""
    model = TransnnMIL(
        feature_dim=1024,
        hidden_dim=256,
        num_classes=2,
        enable_hierarchical=False,
    )
    
    # Forward pass (no coordinates needed)
    features = torch.randn(4, 100, 1024)
    num_patches = torch.tensor([100, 80, 90, 100])
    
    logits = model(features, num_patches)
    assert logits.shape == (4, 2), f"Expected (4, 2), got {logits.shape}"
    
    print("✓ Hierarchical disabled test passed")


def test_hierarchical_enabled():
    """Test TransnnMIL with hierarchical pooling."""
    model = TransnnMIL(
        feature_dim=1024,
        hidden_dim=256,
        num_classes=2,
        enable_hierarchical=True,
        num_regions=16,
        region_hidden_dim=512,
        clustering_method='learnable',
        pooling_method='attention',
    )
    
    # Forward pass (coordinates required)
    features = torch.randn(4, 100, 1024)
    coordinates = torch.rand(4, 100, 2)  # Normalized [0, 1]
    num_patches = torch.tensor([100, 80, 90, 100])
    
    logits = model(features, num_patches, coordinates=coordinates)
    assert logits.shape == (4, 2), f"Expected (4, 2), got {logits.shape}"
    
    print("✓ Hierarchical enabled test passed")


def test_hierarchical_with_attention():
    """Test hierarchical pooling with attention return."""
    model = TransnnMIL(
        feature_dim=1024,
        hidden_dim=256,
        num_classes=2,
        enable_hierarchical=True,
        num_regions=16,
    )
    
    features = torch.randn(4, 100, 1024)
    coordinates = torch.rand(4, 100, 2)
    num_patches = torch.tensor([100, 80, 90, 100])
    
    logits, attention = model(
        features, num_patches, return_attention=True, coordinates=coordinates
    )
    
    assert logits.shape == (4, 2), f"Expected logits (4, 2), got {logits.shape}"
    # Attention should be over regions (16), not patches (100)
    assert attention.shape == (4, 16), f"Expected attention (4, 16), got {attention.shape}"
    
    print("✓ Hierarchical with attention test passed")


def test_hierarchical_pooling_methods():
    """Test different pooling methods."""
    for pooling_method in ['attention', 'mean', 'max']:
        model = TransnnMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            enable_hierarchical=True,
            num_regions=16,
            pooling_method=pooling_method,
        )
        
        features = torch.randn(4, 100, 1024)
        coordinates = torch.rand(4, 100, 2)
        
        logits = model(features, coordinates=coordinates)
        assert logits.shape == (4, 2), f"Pooling {pooling_method} failed"
        
        print(f"✓ Pooling method '{pooling_method}' test passed")


def test_get_branch_outputs_hierarchical():
    """Test get_branch_outputs with hierarchical pooling."""
    model = TransnnMIL(
        feature_dim=1024,
        hidden_dim=256,
        num_classes=2,
        enable_hierarchical=True,
        num_regions=16,
    )
    
    features = torch.randn(4, 100, 1024)
    coordinates = torch.rand(4, 100, 2)
    
    logits_a, logits_b, logits_fused = model.get_branch_outputs(
        features, coordinates=coordinates
    )
    
    assert logits_a.shape == (4, 2), f"Expected (4, 2), got {logits_a.shape}"
    assert logits_b.shape == (4, 2), f"Expected (4, 2), got {logits_b.shape}"
    assert logits_fused.shape == (4, 2), f"Expected (4, 2), got {logits_fused.shape}"
    
    print("✓ get_branch_outputs with hierarchical test passed")


def test_coordinates_required():
    """Test that coordinates are required when hierarchical is enabled."""
    model = TransnnMIL(
        feature_dim=1024,
        enable_hierarchical=True,
    )
    
    features = torch.randn(4, 100, 1024)
    
    try:
        logits = model(features)  # Missing coordinates
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "coordinates required" in str(e)
        print("✓ Coordinates required test passed")


if __name__ == "__main__":
    print("Testing hierarchical pooling integration with TransnnMIL...\n")
    
    test_hierarchical_disabled()
    test_hierarchical_enabled()
    test_hierarchical_with_attention()
    test_hierarchical_pooling_methods()
    test_get_branch_outputs_hierarchical()
    test_coordinates_required()
    
    print("\n✅ All tests passed!")
