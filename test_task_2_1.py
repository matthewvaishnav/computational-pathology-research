"""Quick test to verify task 2.1 implementation."""
import torch
from src.models.transnnmil import TransnnMIL

# Create model
model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
model.eval()

# Create dummy input
batch_size = 4
num_patches_val = 100
features = torch.randn(batch_size, num_patches_val, 1024)
num_patches = torch.tensor([100, 80, 90, 100])

print("Testing feature extraction in forward pass...")

# Test without return_attention
logits = model(features, num_patches, return_attention=False)
print(f"✓ Forward pass without attention: logits shape = {logits.shape}")
assert logits.shape == (batch_size, 2), f"Expected shape (4, 2), got {logits.shape}"

# Test with return_attention
logits, attention = model(features, num_patches, return_attention=True)
print(f"✓ Forward pass with attention: logits shape = {logits.shape}, attention shape = {attention.shape}")
assert logits.shape == (batch_size, 2), f"Expected logits shape (4, 2), got {logits.shape}"
assert attention.shape == (batch_size, num_patches_val), f"Expected attention shape (4, 100), got {attention.shape}"

# Test that get_features methods are being called correctly
print("\nTesting get_features methods directly...")
features_a = model.branch_a.get_features(features, num_patches)
features_b = model.branch_b.get_features(features, num_patches)
print(f"✓ Branch A features shape: {features_a.shape} (expected: [4, 256])")
print(f"✓ Branch B features shape: {features_b.shape} (expected: [4, 1024])")
assert features_a.shape == (batch_size, 256), f"Expected shape (4, 256), got {features_a.shape}"
assert features_b.shape == (batch_size, 1024), f"Expected shape (4, 1024), got {features_b.shape}"

print("\n✅ All tests passed! Task 2.1 implementation is correct.")
