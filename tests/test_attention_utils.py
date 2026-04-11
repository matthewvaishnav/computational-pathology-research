"""
Unit tests for attention weight storage utilities.

Tests cover:
- Saving attention weights to HDF5
- Loading attention weights from HDF5
- Round-trip preservation of values
- Handling of missing files
- Dimension mismatch error handling
"""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from src.utils.attention_utils import load_attention_weights, save_attention_weights


class TestSaveAttentionWeights(unittest.TestCase):
    """Test save_attention_weights function."""

    def test_save_creates_file(self):
        """Test that saving creates an HDF5 file."""
        attention_weights = torch.rand(100)
        coordinates = torch.randint(0, 1000, (100, 2))
        slide_id = "test_slide_001"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_attention_weights(attention_weights, coordinates, slide_id, output_dir)

            # Verify file exists
            output_path = output_dir / f"{slide_id}.h5"
            self.assertTrue(output_path.exists(), "HDF5 file not created")

    def test_save_preserves_values(self):
        """Test that saved values match input values."""
        attention_weights = torch.rand(100)
        coordinates = torch.randint(0, 1000, (100, 2))
        slide_id = "test_slide_002"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_attention_weights(attention_weights, coordinates, slide_id, output_dir)

            # Load and verify
            output_path = output_dir / f"{slide_id}.h5"
            with h5py.File(output_path, "r") as f:
                saved_weights = f["attention_weights"][:]
                saved_coords = f["coordinates"][:]

                np.testing.assert_allclose(
                    saved_weights,
                    attention_weights.numpy(),
                    err_msg="Attention weights not preserved",
                )
                np.testing.assert_array_equal(
                    saved_coords,
                    coordinates.numpy(),
                    err_msg="Coordinates not preserved",
                )

    def test_save_includes_slide_id_attribute(self):
        """Test that slide_id is saved as HDF5 attribute."""
        attention_weights = torch.rand(50)
        coordinates = torch.randint(0, 1000, (50, 2))
        slide_id = "test_slide_003"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_attention_weights(attention_weights, coordinates, slide_id, output_dir)

            # Verify attribute
            output_path = output_dir / f"{slide_id}.h5"
            with h5py.File(output_path, "r") as f:
                self.assertEqual(
                    f.attrs["slide_id"],
                    slide_id,
                    "slide_id attribute not saved correctly",
                )

    def test_save_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        attention_weights = torch.rand(50)
        coordinates = torch.randint(0, 1000, (50, 2))
        slide_id = "test_slide_004"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "directory"
            self.assertFalse(output_dir.exists(), "Directory should not exist yet")

            save_attention_weights(attention_weights, coordinates, slide_id, output_dir)

            self.assertTrue(output_dir.exists(), "Output directory not created")
            output_path = output_dir / f"{slide_id}.h5"
            self.assertTrue(output_path.exists(), "HDF5 file not created")

    def test_save_dimension_mismatch_raises_error(self):
        """Test that dimension mismatch raises ValueError."""
        attention_weights = torch.rand(100)
        coordinates = torch.randint(0, 1000, (80, 2))  # Different length
        slide_id = "test_slide_005"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with self.assertRaises(ValueError) as context:
                save_attention_weights(attention_weights, coordinates, slide_id, output_dir)

            self.assertIn("must have same length", str(context.exception))

    def test_save_with_different_patch_counts(self):
        """Test saving with various patch counts."""
        test_cases = [10, 50, 100, 500, 1000]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            for num_patches in test_cases:
                attention_weights = torch.rand(num_patches)
                coordinates = torch.randint(0, 1000, (num_patches, 2))
                slide_id = f"test_slide_{num_patches}"

                save_attention_weights(attention_weights, coordinates, slide_id, output_dir)

                # Verify file exists and has correct size
                output_path = output_dir / f"{slide_id}.h5"
                with h5py.File(output_path, "r") as f:
                    self.assertEqual(
                        len(f["attention_weights"]),
                        num_patches,
                        f"Wrong number of patches for {num_patches}",
                    )


class TestLoadAttentionWeights(unittest.TestCase):
    """Test load_attention_weights function."""

    def test_load_existing_file(self):
        """Test loading from an existing HDF5 file."""
        attention_weights = np.random.rand(100)
        coordinates = np.random.randint(0, 1000, (100, 2))
        slide_id = "test_slide_006"

        with tempfile.TemporaryDirectory() as tmpdir:
            attention_dir = Path(tmpdir)
            output_path = attention_dir / f"{slide_id}.h5"

            # Create test file
            with h5py.File(output_path, "w") as f:
                f.create_dataset("attention_weights", data=attention_weights)
                f.create_dataset("coordinates", data=coordinates)
                f.attrs["slide_id"] = slide_id

            # Load and verify
            loaded_weights, loaded_coords = load_attention_weights(slide_id, attention_dir)

            self.assertIsNotNone(loaded_weights, "Failed to load attention weights")
            self.assertIsNotNone(loaded_coords, "Failed to load coordinates")
            np.testing.assert_allclose(
                loaded_weights, attention_weights, err_msg="Loaded weights don't match"
            )
            np.testing.assert_array_equal(
                loaded_coords, coordinates, err_msg="Loaded coordinates don't match"
            )

    def test_load_missing_file_returns_none(self):
        """Test that loading missing file returns None."""
        slide_id = "nonexistent_slide"

        with tempfile.TemporaryDirectory() as tmpdir:
            attention_dir = Path(tmpdir)

            result = load_attention_weights(slide_id, attention_dir)

            self.assertIsNone(result, "Should return None for missing file")

    def test_load_with_different_patch_counts(self):
        """Test loading files with various patch counts."""
        test_cases = [10, 50, 100, 500]

        with tempfile.TemporaryDirectory() as tmpdir:
            attention_dir = Path(tmpdir)

            for num_patches in test_cases:
                attention_weights = np.random.rand(num_patches)
                coordinates = np.random.randint(0, 1000, (num_patches, 2))
                slide_id = f"test_slide_{num_patches}"

                # Create test file
                output_path = attention_dir / f"{slide_id}.h5"
                with h5py.File(output_path, "w") as f:
                    f.create_dataset("attention_weights", data=attention_weights)
                    f.create_dataset("coordinates", data=coordinates)

                # Load and verify
                loaded_weights, loaded_coords = load_attention_weights(slide_id, attention_dir)

                self.assertEqual(
                    len(loaded_weights),
                    num_patches,
                    f"Wrong number of patches loaded for {num_patches}",
                )
                self.assertEqual(
                    loaded_coords.shape[0],
                    num_patches,
                    f"Wrong coordinate count for {num_patches}",
                )


class TestRoundTripPreservation(unittest.TestCase):
    """Test that save followed by load preserves values."""

    def test_round_trip_float32(self):
        """Test round-trip with float32 tensors."""
        attention_weights = torch.rand(100, dtype=torch.float32)
        coordinates = torch.randint(0, 1000, (100, 2), dtype=torch.int32)
        slide_id = "test_slide_007"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Save
            save_attention_weights(attention_weights, coordinates, slide_id, output_dir)

            # Load
            loaded_weights, loaded_coords = load_attention_weights(slide_id, output_dir)

            # Verify
            np.testing.assert_allclose(
                loaded_weights,
                attention_weights.numpy(),
                rtol=1e-6,
                err_msg="Round-trip failed for float32",
            )
            np.testing.assert_array_equal(
                loaded_coords,
                coordinates.numpy(),
                err_msg="Round-trip failed for coordinates",
            )

    def test_round_trip_float64(self):
        """Test round-trip with float64 tensors."""
        attention_weights = torch.rand(100, dtype=torch.float64)
        coordinates = torch.randint(0, 1000, (100, 2), dtype=torch.int64)
        slide_id = "test_slide_008"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Save
            save_attention_weights(attention_weights, coordinates, slide_id, output_dir)

            # Load
            loaded_weights, loaded_coords = load_attention_weights(slide_id, output_dir)

            # Verify
            np.testing.assert_allclose(
                loaded_weights,
                attention_weights.numpy(),
                rtol=1e-10,
                err_msg="Round-trip failed for float64",
            )
            np.testing.assert_array_equal(
                loaded_coords,
                coordinates.numpy(),
                err_msg="Round-trip failed for coordinates",
            )

    def test_round_trip_preserves_extreme_values(self):
        """Test that extreme values are preserved in round-trip."""
        # Create attention weights with extreme values
        attention_weights = torch.tensor([0.0, 1e-10, 0.5, 0.999999, 1.0])
        coordinates = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        slide_id = "test_slide_009"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Save
            save_attention_weights(attention_weights, coordinates, slide_id, output_dir)

            # Load
            loaded_weights, loaded_coords = load_attention_weights(slide_id, output_dir)

            # Verify extreme values preserved
            np.testing.assert_allclose(
                loaded_weights,
                attention_weights.numpy(),
                rtol=1e-6,
                atol=1e-10,
                err_msg="Extreme values not preserved",
            )

    def test_round_trip_multiple_slides(self):
        """Test round-trip for multiple slides."""
        num_slides = 5

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Save multiple slides
            original_data = {}
            for i in range(num_slides):
                attention_weights = torch.rand(100)
                coordinates = torch.randint(0, 1000, (100, 2))
                slide_id = f"test_slide_{i:03d}"

                save_attention_weights(attention_weights, coordinates, slide_id, output_dir)
                original_data[slide_id] = (
                    attention_weights.numpy(),
                    coordinates.numpy(),
                )

            # Load and verify all slides
            for slide_id, (orig_weights, orig_coords) in original_data.items():
                loaded_weights, loaded_coords = load_attention_weights(slide_id, output_dir)

                np.testing.assert_allclose(
                    loaded_weights,
                    orig_weights,
                    err_msg=f"Round-trip failed for {slide_id}",
                )
                np.testing.assert_array_equal(
                    loaded_coords,
                    orig_coords,
                    err_msg=f"Coordinates failed for {slide_id}",
                )


if __name__ == "__main__":
    unittest.main()
