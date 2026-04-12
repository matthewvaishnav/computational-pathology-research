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
matplotlib.use("Agg")

from src.visualization.attention_heatmap import AttentionHeatmapGenerator


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with (
        tempfile.TemporaryDirectory() as attention_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        yield Path(attention_dir), Path(output_dir)


@pytest.fixture
def sample_attention_data(temp_dirs):
    """Create sample attention weight HDF5 file."""
    attention_dir, _ = temp_dirs

    # Create sample data
    slide_id = "test_slide_001"
    attention_weights = np.array([0.1, 0.3, 0.6, 0.2, 0.4])
    coordinates = np.array([[0, 0], [256, 0], [512, 0], [0, 256], [256, 256]])

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
        thumbnail_size=(1000, 1000),
    )

    assert generator.attention_dir == attention_dir
    assert generator.output_dir == output_dir
    assert generator.thumbnail_size == (1000, 1000)
    assert output_dir.exists()


def test_load_attention_weights(temp_dirs, sample_attention_data):
    """Test loading attention weights from HDF5."""
    attention_dir, output_dir = temp_dirs
    slide_id, expected_weights, expected_coords = sample_attention_data

    generator = AttentionHeatmapGenerator(attention_dir=attention_dir, output_dir=output_dir)

    result = generator.load_attention_weights(slide_id)
    assert result is not None

    weights, coords = result
    np.testing.assert_array_almost_equal(weights, expected_weights)
    np.testing.assert_array_equal(coords, expected_coords)


def test_load_attention_weights_missing_file(temp_dirs):
    """Test loading attention weights when file doesn't exist."""
    attention_dir, output_dir = temp_dirs

    generator = AttentionHeatmapGenerator(attention_dir=attention_dir, output_dir=output_dir)

    result = generator.load_attention_weights("nonexistent_slide")
    assert result is None


def test_create_heatmap_array(temp_dirs, sample_attention_data):
    """Test creating heatmap array from attention weights."""
    attention_dir, output_dir = temp_dirs
    _, attention_weights, coordinates = sample_attention_data

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(100, 100)
    )

    heatmap = generator.create_heatmap_array(
        attention_weights, coordinates, canvas_size=(100, 100), patch_size=256
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
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(100, 100)
    )

    # Test with different ranges of attention weights
    attention_weights = np.array([10.0, 20.0, 30.0])
    coordinates = np.array([[0, 0], [256, 0], [512, 0]])

    heatmap = generator.create_heatmap_array(
        attention_weights, coordinates, canvas_size=(100, 100), patch_size=256
    )

    # Values should be normalized
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_generate_heatmap(temp_dirs, sample_attention_data):
    """Test generating a complete heatmap."""
    attention_dir, output_dir = temp_dirs
    slide_id, _, _ = sample_attention_data

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(100, 100)
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

    generator = AttentionHeatmapGenerator(attention_dir=attention_dir, output_dir=output_dir)

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
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(100, 100)
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

    generator = AttentionHeatmapGenerator(attention_dir=attention_dir, output_dir=output_dir)

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
            thumbnail_size=(100, 100),
        )

        output_path = generator.generate_heatmap(slide_id)
        assert output_path is not None
        assert output_path.exists()


def test_variable_patch_counts(temp_dirs):
    """Test handling of variable numbers of patches."""
    attention_dir, output_dir = temp_dirs

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(100, 100)
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
            attention_weights, coordinates, canvas_size=(100, 100)
        )

        assert heatmap.shape == (100, 100)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0


def test_generate_heatmap_with_zoom(temp_dirs, sample_attention_data):
    """Test generating heatmap with zoom functionality."""
    attention_dir, output_dir = temp_dirs
    slide_id, _, _ = sample_attention_data

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(1000, 1000)
    )

    # Test with auto-detected zoom regions
    output_path = generator.generate_heatmap_with_zoom(slide_id, top_k_patches=3)

    assert output_path is not None
    assert output_path.exists()
    assert "zoom" in output_path.name


def test_generate_heatmap_with_manual_zoom_regions(temp_dirs, sample_attention_data):
    """Test generating heatmap with manually specified zoom regions."""
    attention_dir, output_dir = temp_dirs
    slide_id, _, _ = sample_attention_data

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(1000, 1000)
    )

    # Specify manual zoom regions
    zoom_regions = [(100, 100, 200, 200), (500, 500, 200, 200)]
    output_path = generator.generate_heatmap_with_zoom(slide_id, zoom_regions=zoom_regions)

    assert output_path is not None
    assert output_path.exists()


def test_identify_high_attention_regions(temp_dirs, sample_attention_data):
    """Test identification of high-attention regions."""
    attention_dir, output_dir = temp_dirs
    _, attention_weights, coordinates = sample_attention_data

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(1000, 1000)
    )

    zoom_regions = generator._identify_high_attention_regions(
        attention_weights, coordinates, top_k=3, region_size=200
    )

    # Should return 3 regions
    assert len(zoom_regions) == 3

    # Each region should be a tuple of (x, y, width, height)
    for region in zoom_regions:
        assert len(region) == 4
        x, y, w, h = region
        assert w == 200
        assert h == 200
        assert 0 <= x < 1000
        assert 0 <= y < 1000


def test_generate_multi_disease_heatmaps(temp_dirs, sample_attention_data):
    """Test generating multi-disease attention heatmaps."""
    attention_dir, output_dir = temp_dirs
    slide_id, attention_weights, coordinates = sample_attention_data

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(1000, 1000)
    )

    # Create disease-specific attention weights (normalized to sum to 1.0)
    disease_weights = {
        "grade_1": attention_weights / attention_weights.sum(),
        "grade_2": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        "grade_3": np.array([0.1, 0.1, 0.3, 0.3, 0.2]),
    }

    output_path = generator.generate_multi_disease_heatmaps(slide_id, disease_weights, coordinates)

    assert output_path is not None
    assert output_path.exists()
    assert "multi_disease" in output_path.name


def test_multi_disease_attention_weight_normalization(temp_dirs, sample_attention_data):
    """Test that attention weights are validated to sum to 1.0 for each disease."""
    attention_dir, output_dir = temp_dirs
    slide_id, _, coordinates = sample_attention_data

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(1000, 1000)
    )

    # Create unnormalized weights (should be auto-normalized)
    disease_weights = {
        "grade_1": np.array([1.0, 2.0, 3.0, 1.0, 2.0]),  # Sum = 9.0
        "grade_2": np.array([0.5, 0.5, 1.0, 1.0, 0.5]),  # Sum = 3.5
    }

    output_path = generator.generate_multi_disease_heatmaps(slide_id, disease_weights, coordinates)

    # Should succeed with normalization
    assert output_path is not None
    assert output_path.exists()


def test_multi_disease_min_probability_filter(temp_dirs, sample_attention_data):
    """Test filtering diseases by minimum probability threshold."""
    attention_dir, output_dir = temp_dirs
    slide_id, _, coordinates = sample_attention_data

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(1000, 1000)
    )

    # Create disease weights with one very low probability disease
    disease_weights = {
        "grade_1": np.array([0.3, 0.3, 0.2, 0.1, 0.1]),  # Max = 0.3
        "grade_2": np.array([0.05, 0.05, 0.05, 0.05, 0.8]),  # Max = 0.8
        "grade_3": np.array([0.02, 0.02, 0.02, 0.02, 0.92]),  # Max = 0.92
    }

    # Filter out diseases with max attention < 0.15
    output_path = generator.generate_multi_disease_heatmaps(
        slide_id, disease_weights, coordinates, min_probability=0.15
    )

    # Should only include grade_1, grade_2, grade_3 (all have max > 0.15)
    assert output_path is not None


def test_generate_feature_importance_explanation(temp_dirs, sample_attention_data):
    """Test generating feature importance explanations."""
    attention_dir, output_dir = temp_dirs
    slide_id, attention_weights, _ = sample_attention_data

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(1000, 1000)
    )

    # Normalize attention weights to sum to 1.0
    attention_weights_norm = attention_weights / attention_weights.sum()

    explanation = generator.generate_feature_importance_explanation(
        slide_id, attention_weights_norm, top_k=3
    )

    # Check structure
    assert "slide_id" in explanation
    assert explanation["slide_id"] == slide_id
    assert "num_patches" in explanation
    assert explanation["num_patches"] == len(attention_weights)
    assert "top_patches" in explanation
    assert len(explanation["top_patches"]) == 3
    assert "top_attention_values" in explanation
    assert len(explanation["top_attention_values"]) == 3
    assert "attention_statistics" in explanation

    # Check statistics
    stats = explanation["attention_statistics"]
    assert "mean" in stats
    assert "std" in stats
    assert "max" in stats
    assert "min" in stats
    assert "median" in stats
    assert "sum" in stats

    # Verify sum is close to 1.0
    assert np.isclose(stats["sum"], 1.0, atol=1e-6)


def test_feature_importance_with_feature_names(temp_dirs, sample_attention_data):
    """Test feature importance explanation with feature names."""
    attention_dir, output_dir = temp_dirs
    slide_id, attention_weights, _ = sample_attention_data

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(1000, 1000)
    )

    # Normalize attention weights
    attention_weights_norm = attention_weights / attention_weights.sum()

    feature_names = ["texture", "color", "shape", "intensity", "gradient"]

    explanation = generator.generate_feature_importance_explanation(
        slide_id, attention_weights_norm, feature_names=feature_names, top_k=3
    )

    # Should include feature importance
    assert "feature_importance" in explanation
    assert isinstance(explanation["feature_importance"], dict)
    assert len(explanation["feature_importance"]) == 3


def test_attention_weight_sum_invariant(temp_dirs):
    """Test that attention weights sum to 1.0 (invariant property)."""
    attention_dir, output_dir = temp_dirs

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(1000, 1000)
    )

    # Create normalized attention weights
    num_patches = 100
    attention_weights = np.random.rand(num_patches)
    attention_weights = attention_weights / attention_weights.sum()

    # Verify sum is 1.0
    assert np.isclose(attention_weights.sum(), 1.0, atol=1e-6)

    # Test with feature importance explanation
    explanation = generator.generate_feature_importance_explanation(
        "test_slide", attention_weights, top_k=10
    )

    # Verify sum in statistics
    assert np.isclose(explanation["attention_statistics"]["sum"], 1.0, atol=1e-6)


def test_multi_disease_single_disease(temp_dirs, sample_attention_data):
    """Test multi-disease visualization with only one disease."""
    attention_dir, output_dir = temp_dirs
    slide_id, attention_weights, coordinates = sample_attention_data

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(1000, 1000)
    )

    # Single disease
    disease_weights = {"grade_1": attention_weights / attention_weights.sum()}

    output_path = generator.generate_multi_disease_heatmaps(slide_id, disease_weights, coordinates)

    assert output_path is not None
    assert output_path.exists()


def test_multi_disease_many_diseases(temp_dirs, sample_attention_data):
    """Test multi-disease visualization with many diseases."""
    attention_dir, output_dir = temp_dirs
    slide_id, attention_weights, coordinates = sample_attention_data

    generator = AttentionHeatmapGenerator(
        attention_dir=attention_dir, output_dir=output_dir, thumbnail_size=(1000, 1000)
    )

    # Create 6 diseases to test multi-row layout
    disease_weights = {}
    for i in range(6):
        weights = np.random.rand(len(attention_weights))
        disease_weights[f"disease_{i}"] = weights / weights.sum()

    output_path = generator.generate_multi_disease_heatmaps(slide_id, disease_weights, coordinates)

    assert output_path is not None
    assert output_path.exists()
