"""
Example usage of FeatureGenerator for WSI feature extraction.

This script demonstrates how to use the FeatureGenerator component to extract
feature embeddings from WSI patches using pretrained CNN encoders.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from src.data.wsi_pipeline import FeatureGenerator, PatchExtractor, WSIReader

# Example 1: Basic feature extraction
print("Example 1: Basic feature extraction")
print("-" * 50)

# Initialize generator with ResNet-50
generator = FeatureGenerator(
    encoder_name="resnet50",
    pretrained=True,  # Use ImageNet pretrained weights
    device="auto",  # Automatically select GPU if available
    batch_size=32,
)

print(f"Encoder: {generator.encoder_name}")
print(f"Device: {generator.device}")
print(f"Feature dimension: {generator.feature_dim}")

# Create dummy patches (in practice, these come from WSI)
patches = np.random.randint(0, 255, (32, 224, 224, 3), dtype=np.uint8)

# Extract features
features = generator.extract_features(patches)
print(f"Input shape: {patches.shape}")
print(f"Output shape: {features.shape}")
print(f"Output dtype: {features.dtype}")

# Example 2: Different encoder architectures
print("\n\nExample 2: Different encoder architectures")
print("-" * 50)

encoders = ["resnet50", "densenet121", "efficientnet_b0"]

for encoder_name in encoders:
    gen = FeatureGenerator(encoder_name=encoder_name, pretrained=False, device="cpu")
    print(f"{encoder_name}: feature_dim={gen.feature_dim}")

# Example 3: Streaming extraction for memory efficiency
print("\n\nExample 3: Streaming extraction")
print("-" * 50)

generator = FeatureGenerator(
    encoder_name="resnet50", pretrained=False, device="cpu", batch_size=8
)


# Simulate patch iterator (in practice, use PatchExtractor)
def patch_iterator():
    for i in range(20):
        yield np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


# Stream feature extraction
features_list = []
for features in generator.extract_features_streaming(patch_iterator()):
    features_list.append(features)

print(f"Extracted {len(features_list)} features using streaming")
print(f"Each feature shape: {features_list[0].shape}")

# Example 4: GPU memory management
print("\n\nExample 4: GPU memory management")
print("-" * 50)

generator = FeatureGenerator(
    encoder_name="resnet50", pretrained=False, device="cpu", batch_size=64
)

print(f"Initial batch size: {generator.batch_size}")

# Simulate GPU OOM by reducing batch size
generator.reduce_batch_size(0.5)
print(f"After reduction: {generator.batch_size}")

# Fallback to CPU if GPU fails
generator.fallback_to_cpu()
print(f"Device after fallback: {generator.device}")

# Clear GPU cache after processing
generator.clear_gpu_cache()
print("GPU cache cleared")

# Example 5: Integration with WSI pipeline
print("\n\nExample 5: Integration with WSI pipeline")
print("-" * 50)

# This example shows how FeatureGenerator integrates with other components
# (requires actual WSI file to run)

"""
# Open WSI file
with WSIReader("path/to/slide.svs") as reader:
    # Initialize patch extractor
    extractor = PatchExtractor(patch_size=224, stride=224, level=0)
    
    # Generate coordinates
    coords = extractor.generate_coordinates(reader.dimensions)
    
    # Initialize feature generator
    generator = FeatureGenerator(
        encoder_name="resnet50",
        pretrained=True,
        device="auto",
        batch_size=32
    )
    
    # Extract features in streaming fashion
    all_features = []
    all_coords = []
    
    for patch, coord in extractor.extract_patches_streaming(reader, coords):
        # Extract features for single patch
        features = generator.extract_features(patch[np.newaxis, ...])
        all_features.append(features[0])
        all_coords.append(coord)
        
        # Clear GPU cache periodically
        if len(all_features) % 100 == 0:
            generator.clear_gpu_cache()
    
    # Stack features
    features_array = torch.stack(all_features).cpu().numpy()
    coords_array = np.array(all_coords)
    
    print(f"Extracted {len(all_features)} features")
    print(f"Features shape: {features_array.shape}")
    print(f"Coordinates shape: {coords_array.shape}")
"""

print("\nAll examples completed!")
