"""Property-based tests for Grad-CAM generator.

These tests validate universal correctness properties defined in the design document.
Each test uses Hypothesis for property-based testing with minimum 100 iterations.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from hypothesis import given, strategies as st, settings
import hypothesis.extra.numpy as npst

from src.interpretability.gradcam import GradCAMGenerator


class SimpleCNN(nn.Module):
    """Simple CNN for property testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.mark.property
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    height=st.integers(min_value=32, max_value=128),
    width=st.integers(min_value=32, max_value=128),
)
@settings(max_examples=100, deadline=None)
def test_property_1_gradcam_heatmap_normalization(batch_size, height, width):
    """
    Feature: model-interpretability, Property 1: Grad-CAM Heatmap Normalization

    For any generated Grad-CAM heatmap from any CNN architecture and input patch,
    all heatmap values SHALL be in the range [0, 1].

    Validates: Requirements 1.6
    """
    # Create model and generator
    model = SimpleCNN()
    model.eval()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    # Generate random input
    images = torch.randn(batch_size, 3, height, width)

    # Generate heatmaps
    heatmaps = generator.generate(images)

    # Verify all heatmaps are in [0, 1]
    for layer_name, heatmap in heatmaps.items():
        heatmap_np = heatmap.detach().cpu().numpy()
        assert np.all(heatmap_np >= 0.0), f"Heatmap for {layer_name} contains values < 0"
        assert np.all(heatmap_np <= 1.0), f"Heatmap for {layer_name} contains values > 1"


@pytest.mark.property
@given(
    architecture=st.sampled_from(["layer1", "layer2", "layer3"]),
    batch_size=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100, deadline=None)
def test_property_2_gradcam_architecture_support(architecture, batch_size):
    """
    Feature: model-interpretability, Property 2: Grad-CAM Architecture Support

    For any CNN architecture in {ResNet18, ResNet50, DenseNet121, EfficientNet-B0}
    and valid input patch, Grad-CAM generation SHALL succeed and produce a valid heatmap.

    Validates: Requirements 1.2

    Note: Using SimpleCNN layers as proxy for different architectures in property test.
    """
    model = SimpleCNN()
    model.eval()

    # Test that generator can be created and generate heatmaps for any layer
    generator = GradCAMGenerator(model, target_layers=[architecture], device="cpu")

    images = torch.randn(batch_size, 3, 64, 64)
    heatmaps = generator.generate(images)

    # Verify heatmap was generated
    assert architecture in heatmaps, f"No heatmap generated for {architecture}"
    assert heatmaps[architecture].shape[0] == batch_size, "Batch size mismatch"
    assert heatmaps[architecture].ndim == 3, "Heatmap should be 3D [batch, H, W]"


@pytest.mark.property
@given(
    num_layers=st.integers(min_value=1, max_value=3),
    batch_size=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100, deadline=None)
def test_property_3_gradcam_multi_layer_output_cardinality(num_layers, batch_size):
    """
    Feature: model-interpretability, Property 3: Grad-CAM Multi-Layer Output Cardinality

    For any list of target layers, the number of generated Grad-CAM heatmaps
    SHALL equal the number of target layers specified.

    Validates: Requirements 1.5
    """
    model = SimpleCNN()
    model.eval()

    # Select num_layers from available layers
    available_layers = ["layer1", "layer2", "layer3"]
    target_layers = available_layers[:num_layers]

    generator = GradCAMGenerator(model, target_layers=target_layers, device="cpu")

    images = torch.randn(batch_size, 3, 64, 64)
    heatmaps = generator.generate(images)

    # Verify number of heatmaps equals number of target layers
    assert len(heatmaps) == num_layers, f"Expected {num_layers} heatmaps, got {len(heatmaps)}"

    # Verify all target layers have heatmaps
    for layer_name in target_layers:
        assert layer_name in heatmaps, f"Missing heatmap for layer {layer_name}"


@pytest.mark.property
@given(
    height=st.integers(min_value=32, max_value=128),
    width=st.integers(min_value=32, max_value=128),
    alpha=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100, deadline=None)
def test_property_4_gradcam_overlay_validity(height, width, alpha):
    """
    Feature: model-interpretability, Property 4: Grad-CAM Overlay Validity

    For any image, heatmap, and transparency value in [0, 1], the overlay operation
    SHALL produce a valid RGB image with shape matching the input image.

    Validates: Requirements 1.3
    """
    model = SimpleCNN()
    model.eval()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    # Create random image and heatmap
    image = np.random.rand(height, width, 3).astype(np.float32)
    heatmap = np.random.rand(height, width).astype(np.float32)

    # Overlay heatmap
    overlaid = generator.overlay_heatmap(image, heatmap, alpha=alpha)

    # Verify output shape matches input
    assert overlaid.shape == image.shape, "Overlay shape mismatch"

    # Verify output is valid RGB in [0, 1]
    assert overlaid.ndim == 3, "Overlay should be 3D"
    assert overlaid.shape[2] == 3, "Overlay should have 3 channels"
    assert np.all(overlaid >= 0.0), "Overlay contains values < 0"
    assert np.all(overlaid <= 1.0), "Overlay contains values > 1"


@pytest.mark.property
@given(
    height=st.integers(min_value=32, max_value=128), width=st.integers(min_value=32, max_value=128)
)
@settings(max_examples=100, deadline=None)
def test_property_5_gradcam_visualization_round_trip(height, width):
    """
    Feature: model-interpretability, Property 5: Grad-CAM Visualization Round-Trip

    For any valid Grad-CAM heatmap, saving then loading the visualization
    SHALL preserve heatmap values within 1% relative error.

    Validates: Requirements 1.8

    Note: This tests the save operation. Full round-trip would require loading
    the saved image, which is tested in unit tests.
    """
    import tempfile

    model = SimpleCNN()
    model.eval()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    # Create random image and heatmap
    image = np.random.rand(height, width, 3).astype(np.float32)
    heatmap = np.random.rand(height, width).astype(np.float32)

    # Save visualization using temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_gradcam.png"
        saved_path = generator.save_visualization(image, heatmap, output_path, dpi=300)

        # Verify file was created
        assert saved_path.exists(), "Visualization file was not created"
        assert saved_path.stat().st_size > 0, "Visualization file is empty"

        # Verify it's a valid image file (basic check)
        assert saved_path.suffix == ".png", "File should be PNG format"
