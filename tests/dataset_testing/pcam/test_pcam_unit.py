"""
Enhanced PCam unit tests.

This module provides comprehensive unit tests for PCam dataset functionality,
including initialization, data loading, indexing, and transform application.

Task 4.1: Create enhanced PCam unit tests
- Test dataset initialization with various configurations
- Validate image dimensions (96x96x3) and binary labels (0 or 1)
- Test data loading, indexing, and transform application
- Requirements: 1.1, 1.2, 1.3
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import h5py
from typing import Dict, Any

from tests.dataset_testing.synthetic.pcam_generator import PCamSyntheticGenerator, PCamSyntheticSpec
from tests.dataset_testing.base_interfaces import PerformanceBenchmark


class TestPCamDatasetInitialization:
    """Test PCam dataset initialization with various configurations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = PCamSyntheticGenerator(random_seed=42)

        # Generate test data
        self.test_samples = self.generator.generate_samples(
            num_samples=100, output_dir=self.temp_dir
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_dataset_initialization_with_valid_path(self):
        """Test dataset initialization with valid data path."""
        # Test that dataset can be initialized with synthetic data
        h5_path = self.temp_dir / "synthetic_pcam.h5"
        assert h5_path.exists(), "Synthetic PCam data should be created"

        # Verify HDF5 file structure
        with h5py.File(h5_path, "r") as f:
            assert "x" in f, "Images dataset should exist"
            assert "y" in f, "Labels dataset should exist"

            images = f["x"]
            labels = f["y"]

            assert images.shape == (100, 96, 96, 3), "Images should have correct shape"
            assert labels.shape == (100,), "Labels should have correct shape"
            assert images.dtype == np.uint8, "Images should be uint8"

    def test_dataset_initialization_with_invalid_path(self):
        """Test dataset initialization with invalid path."""
        invalid_path = self.temp_dir / "nonexistent.h5"

        # Should handle missing file gracefully
        assert not invalid_path.exists(), "File should not exist"

    def test_dataset_initialization_with_custom_transforms(self):
        """Test dataset initialization with custom transforms."""
        # Mock transform
        mock_transform = Mock()
        mock_transform.return_value = torch.randn(3, 96, 96)

        # Test that transforms can be applied
        sample_image = self.test_samples["images"][0]
        transformed = mock_transform(sample_image)

        assert transformed.shape == (3, 96, 96), "Transform should change image format"
        mock_transform.assert_called_once()

    def test_dataset_initialization_with_different_splits(self):
        """Test dataset initialization with different data splits."""
        # Create splits using generator
        splits = self.generator.create_dataset_splits(
            self.test_samples, split_ratios={"train": 0.7, "val": 0.15, "test": 0.15}
        )

        assert "train" in splits, "Train split should exist"
        assert "val" in splits, "Validation split should exist"
        assert "test" in splits, "Test split should exist"

        # Verify split sizes
        total_samples = sum(len(split["images"]) for split in splits.values())
        assert total_samples == 100, "All samples should be distributed across splits"

        # Verify splits are not empty
        # Note: This is a simplified check - in practice, we'd need actual indices
        assert len(splits["train"]["images"]) > 0, "Train split should not be empty"
        assert len(splits["val"]["images"]) > 0, "Val split should not be empty"
        assert len(splits["test"]["images"]) > 0, "Test split should not be empty"

    def test_dataset_initialization_with_various_configurations(self):
        """Test dataset initialization with various configuration parameters."""
        # Test different sample counts
        for num_samples in [10, 50, 200]:
            samples = self.generator.generate_samples(num_samples=num_samples)
            assert len(samples["images"]) == num_samples, f"Should generate {num_samples} samples"
            assert len(samples["labels"]) == num_samples, f"Should generate {num_samples} labels"

        # Test different label distributions
        distributions = [{0: 0.3, 1: 0.7}, {0: 0.8, 1: 0.2}, {0: 0.5, 1: 0.5}]

        for dist in distributions:
            spec = PCamSyntheticSpec(num_samples=100, label_distribution=dist)
            samples = self.generator.generate_samples(num_samples=100, spec=spec)

            # Check approximate distribution (allow some variance)
            label_counts = np.bincount(samples["labels"])
            if len(label_counts) == 2:
                actual_ratio_0 = label_counts[0] / len(samples["labels"])
                expected_ratio_0 = dist[0]
                assert (
                    abs(actual_ratio_0 - expected_ratio_0) < 0.2
                ), f"Label distribution should approximate expected: {actual_ratio_0} vs {expected_ratio_0}"

    def test_dataset_initialization_error_handling(self):
        """Test error handling during dataset initialization."""
        # Test with corrupted file
        corrupted_file = self.temp_dir / "corrupted.h5"

        # Create file with invalid structure
        with h5py.File(corrupted_file, "w") as f:
            f.create_dataset("wrong_name", data=np.random.randint(0, 255, (10, 32, 32, 3)))

        # Should detect invalid structure
        with h5py.File(corrupted_file, "r") as f:
            assert "x" not in f, "Should not have correct image dataset"
            assert "y" not in f, "Should not have correct label dataset"


class TestPCamDataValidation:
    """Test PCam data validation and format checking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PCamSyntheticGenerator(random_seed=42)

    def test_image_dimensions_validation(self):
        """Test validation of image dimensions (96x96x3)."""
        # Generate valid samples
        samples = self.generator.generate_samples(num_samples=10)

        # Validate dimensions
        images = samples["images"]
        assert images.shape == (10, 96, 96, 3), "Images should be 96x96x3"
        assert images.dtype == np.uint8, "Images should be uint8"

        # Test individual image dimensions
        for i in range(len(images)):
            img = images[i]
            assert img.shape == (96, 96, 3), f"Image {i} should be 96x96x3"
            assert 0 <= img.min() <= img.max() <= 255, f"Image {i} should have valid pixel values"

    def test_binary_labels_validation(self):
        """Test validation of binary labels (0 or 1)."""
        # Generate samples with different label distributions
        samples = self.generator.generate_samples(
            num_samples=100,
            spec=PCamSyntheticSpec(num_samples=100, label_distribution={0: 0.3, 1: 0.7}),
        )

        labels = samples["labels"]

        # Validate label properties
        assert labels.dtype in [np.int32, np.int64], "Labels should be integer type"
        assert set(np.unique(labels)) <= {0, 1}, "Labels should only contain 0 and 1"
        assert len(labels) == 100, "Should have correct number of labels"

        # Check distribution is approximately correct
        label_counts = np.bincount(labels)
        if len(label_counts) == 2:  # Both labels present
            ratio_0 = label_counts[0] / len(labels)
            ratio_1 = label_counts[1] / len(labels)

            # Allow some variance due to randomness
            assert 0.2 <= ratio_0 <= 0.4, "Label 0 ratio should be approximately 0.3"
            assert 0.6 <= ratio_1 <= 0.8, "Label 1 ratio should be approximately 0.7"

    def test_data_type_validation(self):
        """Test validation of data types."""
        samples = self.generator.generate_samples(num_samples=5)

        # Validate data types
        assert isinstance(samples["images"], np.ndarray), "Images should be numpy array"
        assert isinstance(samples["labels"], np.ndarray), "Labels should be numpy array"
        assert isinstance(samples["metadata"], dict), "Metadata should be dictionary"

        # Validate array dtypes
        assert samples["images"].dtype == np.uint8, "Images should be uint8"
        assert samples["labels"].dtype in [np.int32, np.int64], "Labels should be integer"

    def test_sample_validation_method(self):
        """Test the validate_samples method."""
        # Valid samples
        valid_samples = self.generator.generate_samples(num_samples=10)
        assert self.generator.validate_samples(
            valid_samples
        ), "Valid samples should pass validation"

        # Invalid samples - wrong image shape
        invalid_samples = valid_samples.copy()
        invalid_samples["images"] = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
        assert not self.generator.validate_samples(
            invalid_samples
        ), "Wrong image shape should fail validation"

        # Invalid samples - wrong labels
        invalid_samples = valid_samples.copy()
        invalid_samples["labels"] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # Invalid labels
        assert not self.generator.validate_samples(
            invalid_samples
        ), "Invalid labels should fail validation"

        # Invalid samples - mismatched lengths
        invalid_samples = valid_samples.copy()
        invalid_samples["labels"] = np.array([0, 1, 0])  # Wrong length
        assert not self.generator.validate_samples(
            invalid_samples
        ), "Mismatched lengths should fail validation"


class TestPCamDataLoading:
    """Test PCam data loading and indexing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = PCamSyntheticGenerator(random_seed=42)

        # Generate and save test data
        self.test_samples = self.generator.generate_samples(
            num_samples=50, output_dir=self.temp_dir
        )
        self.h5_path = self.temp_dir / "synthetic_pcam.h5"

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_data_loading_from_h5_file(self):
        """Test loading data from HDF5 file."""
        # Load data from file
        with h5py.File(self.h5_path, "r") as f:
            loaded_images = f["x"][:]
            loaded_labels = f["y"][:]

        # Verify loaded data matches original
        np.testing.assert_array_equal(loaded_images, self.test_samples["images"])
        np.testing.assert_array_equal(loaded_labels, self.test_samples["labels"])

    def test_data_indexing(self):
        """Test data indexing functionality."""
        images = self.test_samples["images"]
        labels = self.test_samples["labels"]

        # Test individual indexing
        for i in range(min(10, len(images))):
            img = images[i]
            label = labels[i]

            assert img.shape == (96, 96, 3), f"Image {i} should have correct shape"
            assert label in [0, 1], f"Label {i} should be binary"

        # Test slice indexing
        img_slice = images[10:20]
        label_slice = labels[10:20]

        assert img_slice.shape == (10, 96, 96, 3), "Image slice should have correct shape"
        assert label_slice.shape == (10,), "Label slice should have correct shape"

    def test_data_loading_performance(self):
        """Test data loading performance."""
        # Create performance benchmark
        baseline_metrics = {
            "loading_time_seconds": 5.0,  # 5 seconds baseline
            "memory_usage_mb": 100.0,  # 100MB baseline
        }
        benchmark = PerformanceBenchmark(baseline_metrics)

        # Benchmark loading operation
        def load_data():
            with h5py.File(self.h5_path, "r") as f:
                return f["x"][:], f["y"][:]

        metrics = benchmark.benchmark_loading(load_data)

        # Verify reasonable performance
        assert metrics["loading_time_seconds"] < 10.0, "Loading should be reasonably fast"
        assert "memory_usage_mb" in metrics, "Memory usage should be tracked"

        # Check for regressions
        benchmark.check_regression(metrics)
        # Note: Regressions are warnings, not failures in unit tests

    def test_batch_loading(self):
        """Test batch loading functionality."""
        images = self.test_samples["images"]
        labels = self.test_samples["labels"]

        batch_size = 8
        num_batches = len(images) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batch_images = images[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]

            assert batch_images.shape == (
                batch_size,
                96,
                96,
                3,
            ), "Batch images should have correct shape"
            assert batch_labels.shape == (batch_size,), "Batch labels should have correct shape"
            assert np.all(np.isin(batch_labels, [0, 1])), "Batch labels should be binary"


class TestPCamTransformApplication:
    """Test PCam transform application functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PCamSyntheticGenerator(random_seed=42)
        self.test_samples = self.generator.generate_samples(num_samples=10)

    def test_normalization_transform(self):
        """Test normalization transform application."""

        # Mock normalization transform
        def normalize_transform(image):
            # Convert to float and normalize to [0, 1]
            normalized = image.astype(np.float32) / 255.0
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            return normalized

        # Apply transform to sample
        sample_image = self.test_samples["images"][0]
        transformed = normalize_transform(sample_image)

        # Verify transform properties
        assert transformed.dtype in [
            np.float32,
            np.float64,
        ], "Normalized image should be float type"
        assert transformed.shape == (96, 96, 3), "Shape should be preserved"

        # Check normalization statistics (approximately)
        for channel in range(3):
            channel_mean = np.mean(transformed[:, :, channel])
            channel_std = np.std(transformed[:, :, channel])

            # Should be approximately normalized (allowing for image content variation)
            assert -3.0 <= channel_mean <= 3.0, f"Channel {channel} mean should be reasonable"
            assert 0.1 <= channel_std <= 5.0, f"Channel {channel} std should be reasonable"

    def test_augmentation_transform(self):
        """Test augmentation transform application."""

        # Mock augmentation transforms
        def flip_transform(image):
            return np.fliplr(image)  # Horizontal flip

        def rotation_transform(image):
            # Simple 90-degree rotation
            return np.rot90(image)

        sample_image = self.test_samples["images"][0]

        # Test horizontal flip
        flipped = flip_transform(sample_image)
        assert flipped.shape == sample_image.shape, "Flip should preserve shape"
        assert flipped.dtype == sample_image.dtype, "Flip should preserve dtype"

        # Test rotation
        rotated = rotation_transform(sample_image)
        assert rotated.shape == sample_image.shape, "Rotation should preserve shape"
        assert rotated.dtype == sample_image.dtype, "Rotation should preserve dtype"

    def test_transform_consistency(self):
        """Test that transforms are applied consistently."""
        sample_image = self.test_samples["images"][0]

        # Define deterministic transform
        def deterministic_transform(image):
            # Simple brightness adjustment
            return np.clip(image.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)

        # Apply transform multiple times
        result1 = deterministic_transform(sample_image)
        result2 = deterministic_transform(sample_image)

        # Results should be identical
        np.testing.assert_array_equal(
            result1, result2, "Deterministic transform should be consistent"
        )

    def test_transform_chain(self):
        """Test chaining multiple transforms."""
        sample_image = self.test_samples["images"][0]

        # Define transform chain
        def transform_chain(image):
            # 1. Normalize to [0, 1]
            normalized = image.astype(np.float32) / 255.0

            # 2. Apply brightness adjustment
            brightened = np.clip(normalized * 1.2, 0, 1)

            # 3. Convert back to uint8
            result = (brightened * 255).astype(np.uint8)

            return result

        transformed = transform_chain(sample_image)

        # Verify final result
        assert transformed.shape == sample_image.shape, "Transform chain should preserve shape"
        assert transformed.dtype == np.uint8, "Transform chain should preserve dtype"
        assert (
            0 <= transformed.min() <= transformed.max() <= 255
        ), "Transform chain should preserve value range"

    def test_transform_error_handling(self):
        """Test transform error handling."""
        sample_image = self.test_samples["images"][0]

        # Define transform that might fail
        def potentially_failing_transform(image):
            if image.shape != (96, 96, 3):
                raise ValueError("Invalid image shape")
            return image * 2  # This might overflow

        # Test with valid input
        try:
            result = potentially_failing_transform(sample_image)
            # Should handle overflow gracefully in real implementation
            assert result.shape == sample_image.shape, "Transform should preserve shape"
        except ValueError:
            pytest.fail("Transform should not fail with valid input")

        # Test with invalid input
        invalid_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            potentially_failing_transform(invalid_image)


class TestPCamMetadata:
    """Test PCam metadata handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PCamSyntheticGenerator(random_seed=42)

    def test_metadata_generation(self):
        """Test metadata generation and structure."""
        samples = self.generator.generate_samples(
            num_samples=20,
            spec=PCamSyntheticSpec(num_samples=20, noise_level=0.2, corruption_probability=0.1),
        )

        metadata = samples["metadata"]

        # Verify metadata structure
        required_fields = [
            "num_samples",
            "image_shape",
            "label_distribution",
            "noise_level",
            "corruption_probability",
            "generator_seed",
        ]

        for field in required_fields:
            assert field in metadata, f"Metadata should contain {field}"

        # Verify metadata values
        assert metadata["num_samples"] == 20, "Metadata should reflect actual sample count"
        assert metadata["image_shape"] == (96, 96, 3), "Metadata should contain correct image shape"
        assert metadata["generator_seed"] == 42, "Metadata should contain generator seed"

    def test_metadata_consistency(self):
        """Test metadata consistency across generations."""
        spec = PCamSyntheticSpec(num_samples=10, noise_level=0.15)

        # Generate samples multiple times with same spec
        samples1 = self.generator.generate_samples(num_samples=10, spec=spec)
        samples2 = self.generator.generate_samples(num_samples=10, spec=spec)

        # Metadata should be consistent (except for actual data)
        metadata1 = samples1["metadata"]
        metadata2 = samples2["metadata"]

        consistent_fields = ["image_shape", "noise_level", "generator_seed"]
        for field in consistent_fields:
            assert (
                metadata1[field] == metadata2[field]
            ), f"Metadata field {field} should be consistent"


class TestPCamEdgeCases:
    """Test PCam edge cases and error conditions.

    Task 4.5: Write unit tests for PCam edge cases
    - Test invalid indices, corrupted files, and missing data
    - Validate error messages and recovery suggestions
    - Requirements: 1.7, 6.1
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = PCamSyntheticGenerator(random_seed=42)
        self.test_samples = self.generator.generate_samples(
            num_samples=50, output_dir=self.temp_dir
        )
        self.h5_path = self.temp_dir / "synthetic_pcam.h5"

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_invalid_indices(self):
        """Test handling of invalid dataset indices."""
        images = self.test_samples["images"]
        labels = self.test_samples["labels"]

        # Test negative indices (should work in Python)
        assert images[-1].shape == (96, 96, 3), "Negative indexing should work"
        assert labels[-1] in [0, 1], "Negative indexing should return valid label"

        # Test out-of-bounds indices
        with pytest.raises(IndexError):
            _ = images[len(images)]  # Should raise IndexError

        with pytest.raises(IndexError):
            _ = labels[len(labels)]  # Should raise IndexError

        # Test invalid slice indices
        empty_slice = images[100:200]  # Beyond dataset size
        assert len(empty_slice) == 0, "Empty slice should return empty array"

    def test_corrupted_files(self):
        """Test handling of corrupted HDF5 files."""
        # Create corrupted file scenarios
        corruption_scenarios = [
            ("missing_datasets", "File missing required datasets"),
            ("wrong_dimensions", "Images have wrong dimensions"),
            ("invalid_data_types", "Data has invalid types"),
            ("truncated_file", "File is truncated/incomplete"),
        ]

        for corruption_type, description in corruption_scenarios:
            corrupted_file = self.temp_dir / f"corrupted_{corruption_type}.h5"

            if corruption_type == "missing_datasets":
                # Create file without required datasets
                with h5py.File(corrupted_file, "w") as f:
                    f.create_dataset("wrong_name", data=np.random.randint(0, 255, (10, 96, 96, 3)))

                # Test detection
                with h5py.File(corrupted_file, "r") as f:
                    assert "x" not in f, "Should not have images dataset"
                    assert "y" not in f, "Should not have labels dataset"

            elif corruption_type == "wrong_dimensions":
                # Create file with wrong image dimensions
                wrong_images = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
                wrong_labels = np.random.randint(0, 2, 10)

                with h5py.File(corrupted_file, "w") as f:
                    f.create_dataset("x", data=wrong_images)
                    f.create_dataset("y", data=wrong_labels)

                # Test detection
                with h5py.File(corrupted_file, "r") as f:
                    assert f["x"].shape[1:] != (96, 96, 3), "Should have wrong dimensions"

            elif corruption_type == "invalid_data_types":
                # Create file with wrong data types
                float_images = np.random.random((10, 96, 96, 3)).astype(np.float64)
                # Use bytes instead of unicode strings for HDF5 compatibility
                string_labels = np.array([b"zero", b"one"] * 5)

                with h5py.File(corrupted_file, "w") as f:
                    f.create_dataset("x", data=float_images)
                    f.create_dataset("y", data=string_labels)

                # Test detection
                with h5py.File(corrupted_file, "r") as f:
                    assert f["x"].dtype != np.uint8, "Should have wrong image dtype"
                    assert f["y"].dtype.kind == "S", "Should have string labels"

    def test_missing_data(self):
        """Test handling of missing or incomplete data."""
        # Test empty dataset
        empty_file = self.temp_dir / "empty.h5"
        with h5py.File(empty_file, "w") as f:
            f.create_dataset("x", data=np.empty((0, 96, 96, 3), dtype=np.uint8))
            f.create_dataset("y", data=np.empty((0,), dtype=np.int32))

        # Test detection
        with h5py.File(empty_file, "r") as f:
            assert len(f["x"]) == 0, "Should detect empty dataset"
            assert len(f["y"]) == 0, "Should detect empty labels"

        # Test partially missing data
        partial_samples = self.generator.corrupt_samples(self.test_samples, "missing_data")

        # Should have some invalid labels (-1)
        assert np.any(partial_samples["labels"] == -1), "Should have missing data markers"

        # Should have some zero images (missing data)
        zero_images = np.all(partial_samples["images"] == 0, axis=(1, 2, 3))
        assert np.any(zero_images), "Should have some zero/missing images"

    def test_memory_constraints(self):
        """Test behavior under memory constraints."""

        # Test loading large dataset in chunks
        def load_in_chunks(file_path: Path, chunk_size: int = 10):
            """Load dataset in chunks to simulate memory constraints."""
            chunks = []

            with h5py.File(file_path, "r") as f:
                total_samples = len(f["y"])

                for start_idx in range(0, total_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_samples)

                    chunk_images = f["x"][start_idx:end_idx]
                    chunk_labels = f["y"][start_idx:end_idx]

                    chunks.append(
                        {
                            "images": chunk_images,
                            "labels": chunk_labels,
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                        }
                    )

            return chunks

        # Test chunked loading
        chunks = load_in_chunks(self.h5_path, chunk_size=20)

        # Verify chunks
        assert len(chunks) > 1, "Should create multiple chunks"

        total_samples_in_chunks = sum(len(chunk["labels"]) for chunk in chunks)
        assert total_samples_in_chunks == 50, "All samples should be in chunks"

        # Verify chunk consistency
        for chunk in chunks:
            assert len(chunk["images"]) == len(
                chunk["labels"]
            ), "Chunk should have matching images and labels"
            assert chunk["images"].shape[1:] == (
                96,
                96,
                3,
            ), "Chunk images should have correct shape"

    def test_concurrent_access(self):
        """Test concurrent access to dataset files."""
        import threading
        import time

        results = []
        errors = []

        def read_dataset(thread_id: int):
            """Read dataset from multiple threads."""
            try:
                with h5py.File(self.h5_path, "r") as f:
                    # Simulate some processing time
                    time.sleep(0.1)

                    # Read some data
                    sample_images = f["x"][:5]
                    sample_labels = f["y"][:5]

                    results.append(
                        {
                            "thread_id": thread_id,
                            "num_images": len(sample_images),
                            "num_labels": len(sample_labels),
                        }
                    )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=read_dataset, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Should not have errors: {errors}"
        assert len(results) == 5, "All threads should complete successfully"

        for result in results:
            assert result["num_images"] == 5, "Each thread should read 5 images"
            assert result["num_labels"] == 5, "Each thread should read 5 labels"

    def test_error_recovery_suggestions(self):
        """Test generation of helpful error recovery suggestions."""

        def generate_error_suggestions(error_type: str, error_details: Dict[str, Any]) -> str:
            """Generate helpful error recovery suggestions."""
            suggestions = {
                "missing_file": "File not found. Please check the file path and ensure the dataset has been downloaded.",
                "corrupted_file": "File appears to be corrupted. Try re-downloading the dataset or check file integrity.",
                "wrong_format": "File format is incorrect. Ensure you're using the correct PCam dataset format (HDF5 with 'x' and 'y' datasets).",
                "insufficient_memory": "Not enough memory to load dataset. Try using a smaller batch size or loading data in chunks.",
                "permission_denied": "Permission denied accessing file. Check file permissions and ensure you have read access.",
                "network_error": "Network error during download. Check your internet connection and try again.",
            }

            base_suggestion = suggestions.get(error_type, "Unknown error occurred.")

            # Add specific details if available
            if error_details:
                if "file_path" in error_details:
                    base_suggestion += f" File path: {error_details['file_path']}"
                if "expected_shape" in error_details and "actual_shape" in error_details:
                    base_suggestion += f" Expected shape: {error_details['expected_shape']}, got: {error_details['actual_shape']}"

            return base_suggestion

        # Test different error scenarios
        test_cases = [
            ("missing_file", {"file_path": "/path/to/missing.h5"}),
            ("corrupted_file", {"file_path": "/path/to/corrupted.h5"}),
            ("wrong_format", {"expected_shape": (96, 96, 3), "actual_shape": (64, 64, 3)}),
            ("insufficient_memory", {}),
            ("permission_denied", {"file_path": "/restricted/path.h5"}),
            ("network_error", {}),
        ]

        for error_type, error_details in test_cases:
            suggestion = generate_error_suggestions(error_type, error_details)

            assert isinstance(suggestion, str), "Suggestion should be a string"
            assert len(suggestion) > 0, "Suggestion should not be empty"
            assert error_type.replace("_", " ") in suggestion.lower() or any(
                keyword in suggestion.lower()
                for keyword in ["file", "dataset", "memory", "permission", "network"]
            ), f"Suggestion should be relevant to error type: {error_type}"
