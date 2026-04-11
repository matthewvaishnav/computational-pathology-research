"""
Unit tests for attention heatmap visualization.

Tests the AttentionHeatmapGenerator class for creating attention heatmaps
from saved attention weights.
"""

import tempfile
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest

# Use non-interactive backend for testing
matplotlib.use('Agg')

from src.visualization.attention_heatmap import AttentionHeatmapGenerator


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as attention_dir, \
         tempfile.TemporaryDirectory() as output_dir:
        yield Path(attention_dir), Path(output_dir)


@pytest.fixture
def sample_attention_data(temp_dirs):
    """Create sample attention weight HDF5 file."""
    attention_dir, _ = temp_dirs
    
    # Create sample data
    slide_id = "test_slide_001"
    attention_weights = np.array([0.1, 0.3, 0.6, 0.2, 0.4])
    coordinates = np.array([
        [0, 0],
        [256, 0],
        [512, 0],
        [0, 256],
        [256, 256]
    ])
    
    # Save to HDF5
    h5_path = attention_dir / f"{slide_id}.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("attention_weights", data=attention_weights)
        f.create_dataset("coordinates", data=coordinates)
        f.attrs["slide_id"] = slide_id
    
    return slide_id, attention_weights, coordinates


def test_init(temp_dirs):
    """Test AttentionHeatmapGenerator initialization."""
    attention_dir, output_dir = temp_dirs
    
    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir,
        output_dir=output_dir,
        colormap="jet",
        thumbnail_size=(1000, 1000)
    )
    
    assert generator.attention_dir == attention_dir
    assert generator.output_dir == output_dir
    assert generator.thumbnail_size == (1000, 1000)
    assert output_dir.exists()


def test_load_attention_weights(temp_dirs, sample_attention_data):
    """Test loading attention weights from HDF5."""
    attention_dir, output_dir = temp_dirs
    slide_id, expected_weights, expected_coords = sample_attention_data
    
    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir,
        output_dir=output_dir
    )
    
    result = generator.load_attention_weights(slide_id)
    assert result is not None
    
    weights, coords = result
    np.testing.assert_array_almost_equal(weights, expected_weights)
    np.testing.assert_array_equal(coords, expected_coords)


def test_load_attention_weights_missing_file(temp_dirs):
    """Test loading attention weights when file doesn't exist."""
    attention_dir, output_dir = temp_dirs
    
    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir,
        output_dir=output_dir
    )
    
    result = generator.load_attention_weights("nonexistent_slide")
    assert result is None


def test_create_heatmap_array(temp_dirs, sample_attention_data):
    """Test creating heatmap array from attention weights."""
    attention_dir, output_dir = temp_dirs
    _, attention_weights, coordinates = sample_attention_data
    
    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir,
        output_dir=output_dir,
        thumbnail_size=(100, 100)
    )
    
    heatmap = generator.create_heatmap_array(
        attention_weights,
        coordinates,
        canvas_size=(100, 100),
        patch_size=256
    )
    
    # Check shape
    assert heatmap.shape == (100, 100)
    
    # Check values are in [0, 1] range
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    
    # Check that non-zero values exist (patches were placed)
    assert heatmap.sum() > 0


def test_create_heatmap_array_normalization(temp_dirs):
    """Test that attention weights are normalized to [0, 1]."""
    attention_dir, output_dir = temp_dirs
    
    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir,
        output_dir=output_dir,
        thumbnail_size=(100, 100)
    )
    
    # Test with different ranges of attention weights
    attention_weights = np.array([10.0, 20.0, 30.0])
    coordinates = np.array([[0, 0], [256, 0], [512, 0]])
    
    heatmap = generator.create_heatmap_array(
        attention_weights,
        coordinates,
        canvas_size=(100, 100),
        patch_size=256
    )
    
    # Values should be normalized
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_generate_heatmap(temp_dirs, sample_attention_data):
    """Test generating a complete heatmap."""
    attention_dir, output_dir = temp_dirs
    slide_id, _, _ = sample_attention_data
    
    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir,
        output_dir=output_dir,
        thumbnail_size=(100, 100)
    )
    
    output_path = generator.generate_heatmap(slide_id)
    
    # Check that output file was created
    assert output_path is not None
    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert slide_id in output_path.name


def test_generate_heatmap_missing_weights(temp_dirs):
    """Test generating heatmap when attention weights don't exist."""
    attention_dir, output_dir = temp_dirs
    
    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir,
        output_dir=output_dir
    )
    
    output_path = generator.generate_heatmap("nonexistent_slide")
    assert output_path is None


def test_generate_batch(temp_dirs, sample_attention_data):
    """Test batch generation of heatmaps."""
    attention_dir, output_dir = temp_dirs
    slide_id, attention_weights, coordinates = sample_attention_data
    
    # Create additional test slides
    slide_ids = [slide_id, "test_slide_002", "test_slide_003"]
    
    for sid in slide_ids[1:]:
        h5_path = attention_dir / f"{sid}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("attention_weights", data=attention_weights)
            f.create_dataset("coordinates", data=coordinates)
            f.attrs["slide_id"] = sid
    
    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir,
        output_dir=output_dir,
        thumbnail_size=(100, 100)
    )
    
    heatmap_paths = generator.generate_batch(slide_ids)
    
    # Check that all heatmaps were generated
    assert len(heatmap_paths) == 3
    for path in heatmap_paths:
        assert path.exists()
        assert path.suffix == ".png"


def test_generate_batch_with_missing_slides(temp_dirs, sample_attention_data):
    """Test batch generation when some slides are missing."""
    attention_dir, output_dir = temp_dirs
    slide_id, _, _ = sample_attention_data
    
    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir,
        output_dir=output_dir
    )
    
    # Mix of existing and non-existing slides
    slide_ids = [slide_id, "nonexistent_slide_001", "nonexistent_slide_002"]
    
    heatmap_paths = generator.generate_batch(slide_ids)
    
    # Only the existing slide should have a heatmap
    assert len(heatmap_paths) == 1
    assert heatmap_paths[0].exists()


def test_colormap_application(temp_dirs, sample_attention_data):
    """Test that different colormaps can be used."""
    attention_dir, output_dir = temp_dirs
    slide_id, _, _ = sample_attention_data
    
    # Test with different colormaps
    for colormap in ["jet", "viridis", "hot", "cool"]:
        generator = AttentionHeatmapGenerator(
            attention_dir=attention_dir,
            output_dir=output_dir,
            colormap=colormap,
            thumbnail_size=(100, 100)
        )
        
        output_path = generator.generate_heatmap(slide_id)
        assert output_path is not None
        assert output_path.exists()


def test_variable_patch_counts(temp_dirs):
    """Test handling of variable numbers of patches."""
    attention_dir, output_dir = temp_dirs
    
    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir,
        output_dir=output_dir,
        thumbnail_size=(100, 100)
    )
    
    # Test with different numbers of patches
    for num_patches in [5, 50, 500]:
        slide_id = f"test_slide_{num_patches}"
        
        # Create random attention data
        attention_weights = np.random.rand(num_patches)
        coordinates = np.random.randint(0, 5000, size=(num_patches, 2))
        
        # Save to HDF5
        h5_path = attention_dir / f"{slide_id}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("attention_weights", data=attention_weights)
            f.create_dataset("coordinates", data=coordinates)
        
        # Generate heatmap
        heatmap = generator.create_heatmap_array(
            attention_weights,
            coordinates,
            canvas_size=(100, 100)
        )
        
        assert heatmap.shape == (100, 100)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
