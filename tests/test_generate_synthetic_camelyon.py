"""
Tests for synthetic CAMELYON data generation.
"""

import json
import pytest
import tempfile
from pathlib import Path

import h5py
import numpy as np


def test_generate_synthetic_camelyon_script_exists():
    """Test that synthetic CAMELYON generator script exists."""
    script_path = Path("scripts/generate_synthetic_camelyon.py")
    assert script_path.exists(), f"Generator script not found: {script_path}"


def test_generate_synthetic_camelyon_creates_index(tmp_path):
    """Test that generator creates slide index."""
    import subprocess

    output_dir = tmp_path / "camelyon_test"

    result = subprocess.run(
        [
            "python",
            "scripts/generate_synthetic_camelyon.py",
            "--output-dir",
            str(output_dir),
            "--num-train",
            "2",
            "--num-val",
            "1",
            "--num-test",
            "1",
            "--num-patches",
            "10",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Generator failed: {result.stderr}"

    # Check slide index exists
    index_path = output_dir / "slide_index.json"
    assert index_path.exists(), "Slide index not created"

    # Load and verify index
    with open(index_path, "r") as f:
        data = json.load(f)

    assert data["dataset"] == "CAMELYON"
    assert data["num_slides"] == 4
    assert len(data["slides"]) == 4

    # Check splits
    train_slides = [s for s in data["slides"] if s["split"] == "train"]
    val_slides = [s for s in data["slides"] if s["split"] == "val"]
    test_slides = [s for s in data["slides"] if s["split"] == "test"]

    assert len(train_slides) == 2
    assert len(val_slides) == 1
    assert len(test_slides) == 1


def test_generate_synthetic_camelyon_creates_features(tmp_path):
    """Test that generator creates HDF5 feature files."""
    import subprocess

    output_dir = tmp_path / "camelyon_test"

    result = subprocess.run(
        [
            "python",
            "scripts/generate_synthetic_camelyon.py",
            "--output-dir",
            str(output_dir),
            "--num-train",
            "2",
            "--num-val",
            "1",
            "--num-test",
            "1",
            "--num-patches",
            "10",
            "--feature-dim",
            "512",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    # Check features directory exists
    features_dir = output_dir / "features"
    assert features_dir.exists(), "Features directory not created"

    # Check that HDF5 files were created
    h5_files = list(features_dir.glob("*.h5"))
    assert len(h5_files) == 4, f"Expected 4 HDF5 files, got {len(h5_files)}"

    # Verify one HDF5 file structure
    with h5py.File(h5_files[0], "r") as f:
        assert "features" in f, "Missing 'features' dataset"
        assert "coordinates" in f, "Missing 'coordinates' dataset"

        features = f["features"]
        coordinates = f["coordinates"]

        assert features.shape == (10, 512), f"Wrong features shape: {features.shape}"
        assert coordinates.shape == (10, 2), f"Wrong coordinates shape: {coordinates.shape}"


def test_generate_synthetic_camelyon_labels_alternate():
    """Test that labels alternate between normal and tumor."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "camelyon_test"

        result = subprocess.run(
            [
                "python",
                "scripts/generate_synthetic_camelyon.py",
                "--output-dir",
                str(output_dir),
                "--num-train",
                "4",
                "--num-val",
                "0",
                "--num-test",
                "0",
                "--num-patches",
                "5",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Load index
        index_path = output_dir / "slide_index.json"
        with open(index_path, "r") as f:
            data = json.load(f)

        # Check labels alternate
        labels = [s["label"] for s in data["slides"]]
        assert labels == [0, 1, 0, 1], f"Labels should alternate: {labels}"


def test_existing_synthetic_camelyon_data():
    """Test that existing synthetic CAMELYON data is valid."""
    data_dir = Path("data/camelyon")

    # Skip if data doesn't exist
    if not data_dir.exists():
        pytest.skip("Synthetic CAMELYON data not generated yet")

    # Check slide index
    index_path = data_dir / "slide_index.json"
    if not index_path.exists():
        pytest.skip("Slide index not found")

    with open(index_path, "r") as f:
        data = json.load(f)

    assert data["dataset"] == "CAMELYON"
    assert data["num_slides"] > 0
    assert len(data["slides"]) == data["num_slides"]

    # Check features directory
    features_dir = data_dir / "features"
    if not features_dir.exists():
        pytest.skip("Features directory not found")

    # Verify at least one HDF5 file
    h5_files = list(features_dir.glob("*.h5"))
    assert len(h5_files) > 0, "No HDF5 feature files found"

    # Verify structure of first file
    with h5py.File(h5_files[0], "r") as f:
        assert "features" in f
        assert "coordinates" in f
        assert f["features"].ndim == 2
        assert f["coordinates"].ndim == 2
        assert f["coordinates"].shape[1] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
