"""
Demo script for attention visualization with explainability features.

This script demonstrates the enhanced attention visualization capabilities
including zoom functionality, multi-disease-state heatmaps, and feature
importance explanations for clinical explainability.
"""

import numpy as np
from pathlib import Path
from src.visualization.attention_heatmap import AttentionHeatmapGenerator

# Initialize generator
generator = AttentionHeatmapGenerator(
    attention_dir=Path("outputs/attention_weights"),
    output_dir=Path("outputs/visualizations"),
    colormap="jet",
    thumbnail_size=(1000, 1000),
)

# Example 1: Generate heatmap with automatic zoom to high-attention regions
print("Generating heatmap with zoom functionality...")
zoom_path = generator.generate_heatmap_with_zoom(
    slide_id="slide_001",
    thumbnail_path=Path("data/thumbnails/slide_001.png"),
    top_k_patches=5,  # Zoom into top 5 high-attention patches
    alpha=0.5,
)
print(f"Saved zoom heatmap to: {zoom_path}")

# Example 2: Generate multi-disease-state attention heatmaps
print("\nGenerating multi-disease attention heatmaps...")

# Simulate disease-specific attention weights (normalized to sum to 1.0)
num_patches = 100
disease_attention_weights = {
    "benign": np.random.dirichlet(np.ones(num_patches)),
    "grade_1": np.random.dirichlet(np.ones(num_patches)),
    "grade_2": np.random.dirichlet(np.ones(num_patches)),
    "grade_3": np.random.dirichlet(np.ones(num_patches)),
}

# Simulate patch coordinates
coordinates = np.random.randint(0, 5000, size=(num_patches, 2))

multi_disease_path = generator.generate_multi_disease_heatmaps(
    slide_id="slide_001",
    disease_attention_weights=disease_attention_weights,
    coordinates=coordinates,
    thumbnail_path=Path("data/thumbnails/slide_001.png"),
    min_probability=0.1,  # Only show diseases with max attention >= 0.1
)
print(f"Saved multi-disease heatmap to: {multi_disease_path}")

# Example 3: Generate feature importance explanation
print("\nGenerating feature importance explanation...")

# Load attention weights for a slide
result = generator.load_attention_weights("slide_001")
if result is not None:
    attention_weights, coordinates = result
    
    # Normalize to sum to 1.0 (invariant property)
    attention_weights = attention_weights / attention_weights.sum()
    
    # Generate explanation
    explanation = generator.generate_feature_importance_explanation(
        slide_id="slide_001",
        attention_weights=attention_weights,
        feature_names=["texture", "color", "shape", "intensity", "gradient"],
        top_k=10,
    )
    
    print(f"\nFeature Importance Explanation:")
    print(f"  Slide: {explanation['slide_id']}")
    print(f"  Number of patches: {explanation['num_patches']}")
    print(f"  Top patches: {explanation['top_patches'][:5]}")
    print(f"  Top attention values: {[f'{v:.4f}' for v in explanation['top_attention_values'][:5]]}")
    print(f"\n  Attention Statistics:")
    for key, value in explanation['attention_statistics'].items():
        print(f"    {key}: {value:.6f}")
    
    # Verify invariant property: attention weights sum to 1.0
    assert np.isclose(explanation['attention_statistics']['sum'], 1.0, atol=1e-6), \
        "Attention weights must sum to 1.0 (invariant property)"
    print("\n  ✓ Invariant property verified: attention weights sum to 1.0")

print("\nDemo complete!")
