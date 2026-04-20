"""Unit tests for Grad-CAM generator."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from src.interpretability.gradcam import GradCAMGenerator


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""

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


def test_gradcam_initialization():
    """Test GradCAMGenerator initialization."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    assert generator.model is model
    assert generator.target_layers == ["layer3"]
    assert len(generator.hooks) == 2  # Forward and backward hooks


def test_gradcam_invalid_layer():
    """Test GradCAMGenerator with invalid layer name."""
    model = SimpleCNN()

    with pytest.raises(ValueError, match="Layer 'invalid_layer' not found"):
        GradCAMGenerator(model, target_layers=["invalid_layer"], device="cpu")


def test_gradcam_empty_layers():
    """Test GradCAMGenerator with empty target_layers."""
    model = SimpleCNN()

    with pytest.raises(ValueError, match="target_layers cannot be empty"):
        GradCAMGenerator(model, target_layers=[], device="cpu")


def test_gradcam_generate_single_image():
    """Test Grad-CAM generation for single image."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    images = torch.randn(1, 3, 64, 64)
    heatmaps = generator.generate(images)

    assert "layer3" in heatmaps
    assert heatmaps["layer3"].shape == (1, 64, 64)
    assert torch.all(heatmaps["layer3"] >= 0)
    assert torch.all(heatmaps["layer3"] <= 1)


def test_gradcam_generate_batch():
    """Test Grad-CAM generation for batch of images."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    batch_size = 4
    images = torch.randn(batch_size, 3, 64, 64)
    heatmaps = generator.generate(images)

    assert "layer3" in heatmaps
    assert heatmaps["layer3"].shape == (batch_size, 64, 64)


def test_gradcam_generate_multiple_layers():
    """Test Grad-CAM generation for multiple layers."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer1", "layer2", "layer3"], device="cpu")

    images = torch.randn(2, 3, 64, 64)
    heatmaps = generator.generate(images)

    assert len(heatmaps) == 3
    assert "layer1" in heatmaps
    assert "layer2" in heatmaps
    assert "layer3" in heatmaps

    for layer_name, heatmap in heatmaps.items():
        assert heatmap.shape == (2, 64, 64)


def test_gradcam_target_class():
    """Test Grad-CAM with specific target class."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    images = torch.randn(2, 3, 64, 64)

    # Generate for class 0
    heatmaps_class0 = generator.generate(images, class_idx=0)

    # Generate for class 1
    heatmaps_class1 = generator.generate(images, class_idx=1)

    # Heatmaps should be different for different classes
    assert not torch.allclose(heatmaps_class0["layer3"], heatmaps_class1["layer3"])


def test_gradcam_overlay_basic():
    """Test heatmap overlay on image."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    image = np.random.rand(64, 64, 3).astype(np.float32)
    heatmap = np.random.rand(64, 64).astype(np.float32)

    overlaid = generator.overlay_heatmap(image, heatmap, alpha=0.5)

    assert overlaid.shape == image.shape
    assert overlaid.dtype == np.float32
    assert np.all(overlaid >= 0)
    assert np.all(overlaid <= 1)


def test_gradcam_overlay_alpha_extremes():
    """Test overlay with alpha=0 and alpha=1."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    image = np.random.rand(64, 64, 3).astype(np.float32)
    heatmap = np.random.rand(64, 64).astype(np.float32)

    # Alpha=0 should show mostly original image
    overlaid_0 = generator.overlay_heatmap(image, heatmap, alpha=0.0)
    assert np.allclose(overlaid_0, image, atol=0.01)

    # Alpha=1 should show mostly heatmap
    overlaid_1 = generator.overlay_heatmap(image, heatmap, alpha=1.0)
    # Should be different from original image
    assert not np.allclose(overlaid_1, image)


def test_gradcam_overlay_image_255_range():
    """Test overlay with image in [0, 255] range."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    image_255 = (np.random.rand(64, 64, 3) * 255).astype(np.float32)
    heatmap = np.random.rand(64, 64).astype(np.float32)

    overlaid = generator.overlay_heatmap(image_255, heatmap, alpha=0.5)

    # Output should still be in [0, 1]
    assert np.all(overlaid >= 0)
    assert np.all(overlaid <= 1)


def test_gradcam_overlay_shape_mismatch():
    """Test overlay with mismatched image and heatmap shapes."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    image = np.random.rand(64, 64, 3).astype(np.float32)
    heatmap = np.random.rand(32, 32).astype(np.float32)  # Different size

    with pytest.raises(ValueError, match="does not match heatmap shape"):
        generator.overlay_heatmap(image, heatmap)


def test_gradcam_overlay_invalid_heatmap_dims():
    """Test overlay with invalid heatmap dimensions."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    image = np.random.rand(64, 64, 3).astype(np.float32)
    heatmap_3d = np.random.rand(64, 64, 1).astype(np.float32)  # 3D instead of 2D

    with pytest.raises(ValueError, match="Heatmap must be 2D"):
        generator.overlay_heatmap(image, heatmap_3d)


def test_gradcam_save_visualization(tmp_path):
    """Test saving Grad-CAM visualization."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    image = np.random.rand(64, 64, 3).astype(np.float32)
    heatmap = np.random.rand(64, 64).astype(np.float32)

    output_path = tmp_path / "test_gradcam.png"
    saved_path = generator.save_visualization(image, heatmap, output_path, dpi=300)

    assert saved_path.exists()
    assert saved_path.suffix == ".png"
    assert saved_path.stat().st_size > 0


def test_gradcam_save_creates_directory(tmp_path):
    """Test that save_visualization creates parent directories."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    image = np.random.rand(64, 64, 3).astype(np.float32)
    heatmap = np.random.rand(64, 64).astype(np.float32)

    # Path with non-existent parent directory
    output_path = tmp_path / "subdir" / "test_gradcam.png"
    saved_path = generator.save_visualization(image, heatmap, output_path)

    assert saved_path.exists()
    assert saved_path.parent.exists()


def test_gradcam_edge_case_all_zero_gradients():
    """Test Grad-CAM with all-zero gradients (edge case)."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    # Create constant image (should produce near-zero gradients)
    images = torch.ones(1, 3, 64, 64) * 0.5
    heatmaps = generator.generate(images)

    # Should still produce valid heatmap in [0, 1]
    assert "layer3" in heatmaps
    assert torch.all(heatmaps["layer3"] >= 0)
    assert torch.all(heatmaps["layer3"] <= 1)


def test_gradcam_edge_case_single_pixel():
    """Test Grad-CAM with very small image (edge case)."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    # Very small image
    images = torch.randn(1, 3, 8, 8)
    heatmaps = generator.generate(images)

    assert "layer3" in heatmaps
    assert heatmaps["layer3"].shape == (1, 8, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gradcam_gpu_execution():
    """Test Grad-CAM execution on GPU."""
    model = SimpleCNN()
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cuda")

    images = torch.randn(2, 3, 64, 64)
    heatmaps = generator.generate(images)

    assert "layer3" in heatmaps
    assert heatmaps["layer3"].device.type == "cuda"


def test_gradcam_cpu_fallback():
    """Test automatic CPU fallback when CUDA requested but unavailable."""
    model = SimpleCNN()

    # This should not raise an error even if CUDA is unavailable
    generator = GradCAMGenerator(model, target_layers=["layer3"], device="cpu")

    images = torch.randn(1, 3, 64, 64)
    heatmaps = generator.generate(images)

    assert "layer3" in heatmaps
