"""
Debug script to check actual feature dimensions
"""

import torch
import timm

# Create ResNet50 model like in the foundation model
encoder = timm.create_model(
    "resnet50",
    pretrained=True,
    num_classes=0,  # Remove classification head
    global_pool=""  # Remove global pooling
)

# Test with sample input
sample_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    features = encoder(sample_input)

print(f"Input shape: {sample_input.shape}")
print(f"Output feature shape: {features.shape}")
print(f"Feature dimension: {features.numel() // features.shape[0]}")

# Test with multiple patches like in the foundation model
batch_size = 1
num_patches = 10
patches = torch.randn(batch_size, num_patches, 3, 224, 224)

# Reshape for batch processing
patches_flat = patches.view(-1, 3, 224, 224)
print(f"Flattened patches shape: {patches_flat.shape}")

with torch.no_grad():
    features_flat = encoder(patches_flat)

print(f"Features flat shape: {features_flat.shape}")

# Reshape back
features_reshaped = features_flat.view(batch_size, num_patches, -1)
print(f"Features reshaped: {features_reshaped.shape}")
print(f"Actual feature dimension per patch: {features_reshaped.shape[-1]}")