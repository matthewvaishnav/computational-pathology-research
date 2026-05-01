"""
Tests for src/utils/attention_utils.py

Tests cover:
- save_attention_weights
- load_attention_weights
- Error handling
- Edge cases
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.exceptions import DataLoadError, DataSaveError
from src.utils.attention_utils import load_attention_weights, save_attention_weights


# ============================================================================
# save_attention_weights Tests
# ============================================================================


class TestSaveAttentionWeights:
    """Tests for save_attention_weights."""

    def test_save_basic(self):
        """Test basic save functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            attention = torch.tensor([0.1, 0.3, 0.6])
            coords = torch.tensor([[0, 0], [256, 0], [0, 256]])

            save_attention_weights(attention, coords, "slide_001", Path(tmpdir))

            # Verify file exists
            output_path = Path(tmpdir) / "slide_001.h5"
            assert output_path.exists()

            # Verify contents
            with h5py.File(output_path, "r") as f:
                assert "attention_weights" in f
                assert "coordinates" in f
                assert f.attrs["slide_id"] == "slide_001"

                saved_attention = f["attention_weights"][:]
                saved_coords = f["coordinates"][:]

                np.testing.assert_array_almost_equal(saved_attention, attention.numpy())
                np.testing.assert_array_equal(saved_coords, coords.numpy())

    def test_save_creates_directory(self):
        """Test that save creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "dir"
            assert not output_dir.exists()

            attention = torch.tensor([0.5])
            coords = torch.tensor([[0, 0]])

            save_attention_weights(attention, coords, "slide_001", output_dir)

            assert output_dir.exists()
            assert (output_dir / "slide_001.h5").exists()

    def test_save_with_cuda_tensors(self):
        """Test save with CUDA tensors (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            attention = torch.tensor([0.1, 0.3, 0.6]).cuda()
            coords = torch.tensor([[0, 0], [256, 0], [0, 256]]).cuda()

            save_attention_weights(attention, coords, "slide_001", Path(tmpdir))

            output_path = Path(tmpdir) / "slide_001.h5"
            assert output_path.exists()

    def test_save_large_attention_weights(self):
        """Test save with large number of patches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            num_patches = 10000
            attention = torch.rand(num_patches)
            coords = torch.randint(0, 1000, (num_patches, 2))

            save_attention_weights(attention, coords, "slide_large", Path(tmpdir))

            output_path = Path(tmpdir) / "slide_large.h5"
            assert output_path.exists()

            # Verify compression worked
            file_size = output_path.stat().st_size
            # Compressed size should be less than uncompressed
            # (rough estimate: 10000 floats * 4 bytes + 10000 * 2 ints * 4 bytes)
            uncompressed_estimate = 10000 * 4 + 10000 * 2 * 4
            assert file_size < uncompressed_estimate

    def test_save_mismatched_lengths_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            attention = torch.tensor([0.1, 0.3, 0.6])
            coords = torch.tensor([[0, 0], [256, 0]])  # Only 2 coords

            with pytest.raises(ValueError, match="must have same length"):
                save_attention_weights(attention, coords, "slide_001", Path(tmpdir))

    def test_save_overwrites_existing_file(self):
        """Test that save overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            attention1 = torch.tensor([0.1, 0.3])
            coords1 = torch.tensor([[0, 0], [256, 0]])

            save_attention_weights(attention1, coords1, "slide_001", Path(tmpdir))

            # Save again with different data
            attention2 = torch.tensor([0.5, 0.7, 0.9])
            coords2 = torch.tensor([[0, 0], [256, 0], [0, 256]])

            save_attention_weights(attention2, coords2, "slide_001", Path(tmpdir))

            # Verify new data
            output_path = Path(tmpdir) / "slide_001.h5"
            with h5py.File(output_path, "r") as f:
                saved_attention = f["attention_weights"][:]
                assert len(saved_attention) == 3

    def test_save_with_special_characters_in_slide_id(self):
        """Test save with special characters in slide ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            attention = torch.tensor([0.5])
            coords = torch.tensor([[0, 0]])

            # Use slide ID with special characters (but valid for filename)
            slide_id = "slide_001-test_v2"

            save_attention_weights(attention, coords, slide_id, Path(tmpdir))

            output_path = Path(tmpdir) / f"{slide_id}.h5"
            assert output_path.exists()

    @pytest.mark.skip(reason="Windows handles invalid paths differently")
    def test_save_io_error_raises_datasaveerror(self):
        """Test that I/O errors raise DataSaveError."""
        # Use invalid path to trigger I/O error
        invalid_path = Path("/invalid/path/that/does/not/exist")

        attention = torch.tensor([0.5])
        coords = torch.tensor([[0, 0]])

        with pytest.raises(DataSaveError, match="Failed to save attention weights"):
            save_attention_weights(attention, coords, "slide_001", invalid_path)


# ============================================================================
# load_attention_weights Tests
# ============================================================================


class TestLoadAttentionWeights:
    """Tests for load_attention_weights."""

    def test_load_basic(self):
        """Test basic load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First save some data
            attention = torch.tensor([0.1, 0.3, 0.6])
            coords = torch.tensor([[0, 0], [256, 0], [0, 256]])
            save_attention_weights(attention, coords, "slide_001", Path(tmpdir))

            # Now load it
            loaded_attention, loaded_coords = load_attention_weights("slide_001", Path(tmpdir))

            assert loaded_attention is not None
            assert loaded_coords is not None

            np.testing.assert_array_almost_equal(loaded_attention, attention.numpy())
            np.testing.assert_array_equal(loaded_coords, coords.numpy())

    def test_load_nonexistent_file_returns_none(self):
        """Test that loading nonexistent file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_attention_weights("nonexistent_slide", Path(tmpdir))

            assert result is None

    def test_load_large_attention_weights(self):
        """Test load with large number of patches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            num_patches = 10000
            attention = torch.rand(num_patches)
            coords = torch.randint(0, 1000, (num_patches, 2))

            save_attention_weights(attention, coords, "slide_large", Path(tmpdir))

            loaded_attention, loaded_coords = load_attention_weights("slide_large", Path(tmpdir))

            assert len(loaded_attention) == num_patches
            assert loaded_coords.shape == (num_patches, 2)

    def test_load_verifies_data_integrity(self):
        """Test that loaded data matches saved data exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use specific values to verify integrity
            attention = torch.tensor([0.123456, 0.789012, 0.345678])
            coords = torch.tensor([[10, 20], [30, 40], [50, 60]])

            save_attention_weights(attention, coords, "slide_001", Path(tmpdir))

            loaded_attention, loaded_coords = load_attention_weights("slide_001", Path(tmpdir))

            # Verify exact values
            np.testing.assert_array_almost_equal(loaded_attention, attention.numpy(), decimal=6)
            np.testing.assert_array_equal(loaded_coords, coords.numpy())

    def test_load_corrupted_file_raises_dataload_error(self):
        """Test that corrupted file raises DataLoadError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a corrupted HDF5 file
            corrupted_path = Path(tmpdir) / "slide_001.h5"
            with open(corrupted_path, "w") as f:
                f.write("This is not a valid HDF5 file")

            with pytest.raises(DataLoadError, match="Failed to load attention weights"):
                load_attention_weights("slide_001", Path(tmpdir))

    def test_load_missing_dataset_raises_dataload_error(self):
        """Test that missing dataset raises DataLoadError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create HDF5 file with missing datasets
            h5_path = Path(tmpdir) / "slide_001.h5"
            with h5py.File(h5_path, "w") as f:
                # Only create attention_weights, not coordinates
                f.create_dataset("attention_weights", data=np.array([0.5]))

            with pytest.raises(DataLoadError, match="Failed to load attention weights"):
                load_attention_weights("slide_001", Path(tmpdir))


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for attention utilities."""

    def test_save_and_load_roundtrip(self):
        """Test complete save and load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            attention = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
            coords = torch.tensor([[0, 0], [256, 0], [512, 0], [0, 256], [256, 256]])

            # Save
            save_attention_weights(attention, coords, "test_slide", Path(tmpdir))

            # Load
            loaded_attention, loaded_coords = load_attention_weights("test_slide", Path(tmpdir))

            # Verify
            assert loaded_attention is not None
            assert loaded_coords is not None
            np.testing.assert_array_almost_equal(loaded_attention, attention.numpy())
            np.testing.assert_array_equal(loaded_coords, coords.numpy())

    def test_multiple_slides_in_same_directory(self):
        """Test saving and loading multiple slides in same directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save multiple slides
            for i in range(5):
                attention = torch.rand(10)
                coords = torch.randint(0, 1000, (10, 2))
                save_attention_weights(attention, coords, f"slide_{i:03d}", Path(tmpdir))

            # Verify all files exist
            for i in range(5):
                h5_path = Path(tmpdir) / f"slide_{i:03d}.h5"
                assert h5_path.exists()

            # Load and verify each slide
            for i in range(5):
                result = load_attention_weights(f"slide_{i:03d}", Path(tmpdir))
                assert result is not None
                loaded_attention, loaded_coords = result
                assert len(loaded_attention) == 10
                assert loaded_coords.shape == (10, 2)

    def test_edge_case_single_patch(self):
        """Test with single patch (edge case)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            attention = torch.tensor([0.999])
            coords = torch.tensor([[128, 128]])

            save_attention_weights(attention, coords, "single_patch", Path(tmpdir))

            loaded_attention, loaded_coords = load_attention_weights("single_patch", Path(tmpdir))

            assert len(loaded_attention) == 1
            assert loaded_coords.shape == (1, 2)
            np.testing.assert_almost_equal(loaded_attention[0], 0.999)

    def test_edge_case_zero_attention_weights(self):
        """Test with all zero attention weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            attention = torch.zeros(5)
            coords = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])

            save_attention_weights(attention, coords, "zero_attention", Path(tmpdir))

            loaded_attention, loaded_coords = load_attention_weights(
                "zero_attention", Path(tmpdir)
            )

            np.testing.assert_array_equal(loaded_attention, np.zeros(5))

    def test_edge_case_negative_coordinates(self):
        """Test with negative coordinates (valid use case)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            attention = torch.tensor([0.5, 0.5])
            coords = torch.tensor([[-100, -200], [100, 200]])

            save_attention_weights(attention, coords, "negative_coords", Path(tmpdir))

            loaded_attention, loaded_coords = load_attention_weights(
                "negative_coords", Path(tmpdir)
            )

            np.testing.assert_array_equal(loaded_coords, coords.numpy())
