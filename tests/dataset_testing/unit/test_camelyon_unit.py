"""
Enhanced CAMELYON unit tests for comprehensive dataset testing.

Tests slide metadata loading, HDF5 feature file structure, and coordinate-feature alignment.
"""


import h5py
import numpy as np
import pytest
import torch

from src.data.camelyon_dataset import (
    CAMELYONPatchDataset,
    CAMELYONSlideDataset,
    CAMELYONSlideIndex,
    SlideMetadata,
    validate_feature_file,
)
from tests.dataset_testing.synthetic.camelyon_generator import (
    CAMELYONSyntheticGenerator,
    CAMELYONSyntheticSpec,
)


class TestCAMELYONSlideMetadataLoading:
    """Test slide metadata loading and validation."""

    def test_slide_metadata_creation_with_all_fields(self):
        """Test creating SlideMetadata with all optional fields."""
        slide = SlideMetadata(
            slide_id="patient_042_node_3",
            patient_id="patient_042",
            file_path="/data/slides/patient_042_node_3.tif",
            label=1,
            split="train",
            annotation_path="/data/annotations/patient_042_node_3.xml",
            width=100000,
            height=80000,
            magnification=40.0,
            mpp=0.25,
        )

        assert slide.slide_id == "patient_042_node_3"
        assert slide.patient_id == "patient_042"
        assert slide.file_path == "/data/slides/patient_042_node_3.tif"
        assert slide.label == 1
        assert slide.split == "train"
        assert slide.annotation_path == "/data/annotations/patient_042_node_3.xml"
        assert slide.width == 100000
        assert slide.height == 80000
        assert slide.magnification == 40.0
        assert slide.mpp == 0.25

    def test_slide_metadata_minimal_creation(self):
        """Test creating SlideMetadata with only required fields."""
        slide = SlideMetadata(
            slide_id="test_slide",
            patient_id="test_patient",
            file_path="/path/to/slide.tif",
            label=0,
            split="val",
        )

        assert slide.slide_id == "test_slide"
        assert slide.patient_id == "test_patient"
        assert slide.label == 0
        assert slide.split == "val"
        assert slide.annotation_path is None
        assert slide.width is None
        assert slide.height is None
        assert slide.magnification is None
        assert slide.mpp is None

    def test_slide_index_creation_and_access(self):
        """Test CAMELYONSlideIndex creation and slide access."""
        slides = [
            SlideMetadata("slide_001", "patient_001", "/path/1.tif", 1, "train"),
            SlideMetadata("slide_002", "patient_001", "/path/2.tif", 0, "train"),
            SlideMetadata("slide_003", "patient_002", "/path/3.tif", 1, "val"),
        ]

        index = CAMELYONSlideIndex(slides)

        assert len(index) == 3
        assert index["slide_001"].label == 1
        assert index["slide_002"].label == 0
        assert index["slide_003"].patient_id == "patient_002"

    def test_slide_index_split_filtering(self):
        """Test filtering slides by split."""
        slides = [
            SlideMetadata("slide_001", "patient_001", "/path/1.tif", 1, "train"),
            SlideMetadata("slide_002", "patient_001", "/path/2.tif", 0, "train"),
            SlideMetadata("slide_003", "patient_002", "/path/3.tif", 1, "val"),
            SlideMetadata("slide_004", "patient_003", "/path/4.tif", 0, "test"),
        ]

        index = CAMELYONSlideIndex(slides)

        train_slides = index.get_slides_by_split("train")
        val_slides = index.get_slides_by_split("val")
        test_slides = index.get_slides_by_split("test")

        assert len(train_slides) == 2
        assert len(val_slides) == 1
        assert len(test_slides) == 1
        assert all(s.split == "train" for s in train_slides)
        assert all(s.split == "val" for s in val_slides)
        assert all(s.split == "test" for s in test_slides)

    def test_slide_index_patient_filtering(self):
        """Test filtering slides by patient."""
        slides = [
            SlideMetadata("slide_001", "patient_001", "/path/1.tif", 1, "train"),
            SlideMetadata("slide_002", "patient_001", "/path/2.tif", 0, "train"),
            SlideMetadata("slide_003", "patient_002", "/path/3.tif", 1, "val"),
        ]

        index = CAMELYONSlideIndex(slides)

        patient_001_slides = index.get_slides_by_patient("patient_001")
        patient_002_slides = index.get_slides_by_patient("patient_002")

        assert len(patient_001_slides) == 2
        assert len(patient_002_slides) == 1
        assert all(s.patient_id == "patient_001" for s in patient_001_slides)
        assert all(s.patient_id == "patient_002" for s in patient_002_slides)

    def test_slide_index_annotation_filtering(self):
        """Test filtering slides with annotations."""
        slides = [
            SlideMetadata("slide_001", "patient_001", "/path/1.tif", 1, "train", "/ann/1.xml"),
            SlideMetadata("slide_002", "patient_001", "/path/2.tif", 0, "train"),
            SlideMetadata("slide_003", "patient_002", "/path/3.tif", 1, "val", "/ann/3.xml"),
        ]

        index = CAMELYONSlideIndex(slides)
        annotated_slides = index.get_annotated_slides()

        assert len(annotated_slides) == 2
        assert all(s.annotation_path is not None for s in annotated_slides)
        assert "slide_001" in [s.slide_id for s in annotated_slides]
        assert "slide_003" in [s.slide_id for s in annotated_slides]


class TestHDF5FeatureFileStructure:
    """Test HDF5 feature file structure validation."""

    def test_valid_hdf5_structure(self, tmp_path):
        """Test validation of properly structured HDF5 files."""
        feature_path = tmp_path / "valid_slide.h5"

        # Create valid HDF5 file
        with h5py.File(feature_path, "w") as f:
            features = np.random.randn(100, 2048).astype(np.float32)
            coordinates = np.random.randint(0, 10000, (100, 2)).astype(np.int32)

            f.create_dataset("features", data=features)
            f.create_dataset("coordinates", data=coordinates)

        result = validate_feature_file(str(feature_path))

        assert result["valid"] is True
        assert result["num_patches"] == 100
        assert result["feature_dim"] == 2048
        assert result["error"] is None

    def test_missing_features_dataset(self, tmp_path):
        """Test detection of missing features dataset."""
        feature_path = tmp_path / "missing_features.h5"

        with h5py.File(feature_path, "w") as f:
            coordinates = np.random.randint(0, 10000, (100, 2)).astype(np.int32)
            f.create_dataset("coordinates", data=coordinates)

        result = validate_feature_file(str(feature_path))

        assert result["valid"] is False
        assert "missing 'features'" in result["error"].lower()

    def test_missing_coordinates_dataset(self, tmp_path):
        """Test detection of missing coordinates dataset."""
        feature_path = tmp_path / "missing_coordinates.h5"

        with h5py.File(feature_path, "w") as f:
            features = np.random.randn(100, 2048).astype(np.float32)
            f.create_dataset("features", data=features)

        result = validate_feature_file(str(feature_path))

        assert result["valid"] is False
        assert "missing 'coordinates'" in result["error"].lower()

    def test_mismatched_patch_counts(self, tmp_path):
        """Test detection of mismatched patch counts between features and coordinates."""
        feature_path = tmp_path / "mismatched_counts.h5"

        with h5py.File(feature_path, "w") as f:
            features = np.random.randn(100, 2048).astype(np.float32)
            coordinates = np.random.randint(0, 10000, (80, 2)).astype(np.int32)  # Different count

            f.create_dataset("features", data=features)
            f.create_dataset("coordinates", data=coordinates)

        result = validate_feature_file(str(feature_path))

        assert result["valid"] is False
        assert "mismatched" in result["error"].lower()

    def test_invalid_coordinates_shape(self, tmp_path):
        """Test detection of invalid coordinates shape."""
        feature_path = tmp_path / "invalid_coords.h5"

        with h5py.File(feature_path, "w") as f:
            features = np.random.randn(100, 2048).astype(np.float32)
            coordinates = np.random.randint(0, 10000, (100, 3)).astype(np.int32)  # Wrong shape

            f.create_dataset("features", data=features)
            f.create_dataset("coordinates", data=coordinates)

        result = validate_feature_file(str(feature_path))

        assert result["valid"] is False
        assert "[n, 2]" in result["error"].lower()

    def test_nonexistent_file(self, tmp_path):
        """Test handling of nonexistent files."""
        result = validate_feature_file(str(tmp_path / "nonexistent.h5"))

        assert result["valid"] is False
        assert "not found" in result["error"].lower()

    def test_corrupted_hdf5_file(self, tmp_path):
        """Test handling of corrupted HDF5 files."""
        feature_path = tmp_path / "corrupted.h5"

        # Create a file with invalid HDF5 content
        with open(feature_path, "w") as f:
            f.write("This is not a valid HDF5 file")

        result = validate_feature_file(str(feature_path))

        assert result["valid"] is False
        assert result["error"] is not None


class TestCoordinateFeatureAlignment:
    """Test coordinate-feature alignment for attention models."""

    @pytest.fixture
    def synthetic_generator(self):
        """Create synthetic data generator."""
        return CAMELYONSyntheticGenerator(random_seed=42)

    @pytest.fixture
    def sample_slide_data(self, synthetic_generator, tmp_path):
        """Generate sample slide data for testing."""
        spec = CAMELYONSyntheticSpec(
            num_slides=3,
            patches_per_slide_range=(50, 100),
            feature_dim=2048,
        )

        dataset = synthetic_generator.generate_samples(
            num_slides=3, spec=spec, output_dir=tmp_path / "synthetic_camelyon"
        )

        return dataset, tmp_path / "synthetic_camelyon"

    def test_coordinate_feature_alignment_consistency(self, sample_slide_data):
        """Test that coordinates and features maintain consistent alignment."""
        dataset, data_dir = sample_slide_data

        # Create slide index
        slides = [
            SlideMetadata(
                slide_id=entry["slide_id"],
                patient_id=entry["patient_id"],
                file_path=f"features/{entry['slide_id']}.h5",
                label=entry["label"],
                split="train",
            )
            for entry in dataset["slide_index"]
        ]

        slide_index = CAMELYONSlideIndex(slides)

        # Create patch dataset
        patch_dataset = CAMELYONPatchDataset(
            slide_index=slide_index,
            features_dir=str(data_dir / "features"),
            split="train",
        )

        # Test alignment for each slide
        for slide_entry in dataset["slide_index"]:
            slide_id = slide_entry["slide_id"]
            slide_data = patch_dataset.get_slide_patch_data(slide_id)

            assert slide_data is not None
            assert slide_data["features"].shape[0] == slide_data["coordinates"].shape[0]
            assert slide_data["features"].shape[0] == slide_data["num_patches"]
            assert slide_data["coordinates"].shape[1] == 2  # (x, y) coordinates
            assert slide_data["features"].shape[1] == 2048  # Feature dimension

    def test_patch_indexing_consistency(self, sample_slide_data):
        """Test that patch indexing maintains coordinate-feature alignment."""
        dataset, data_dir = sample_slide_data

        # Create slide index
        slides = [
            SlideMetadata(
                slide_id=entry["slide_id"],
                patient_id=entry["patient_id"],
                file_path=f"features/{entry['slide_id']}.h5",
                label=entry["label"],
                split="train",
            )
            for entry in dataset["slide_index"]
        ]

        slide_index = CAMELYONSlideIndex(slides)

        # Create patch dataset
        patch_dataset = CAMELYONPatchDataset(
            slide_index=slide_index,
            features_dir=str(data_dir / "features"),
            split="train",
        )

        # Test individual patch access
        for i in range(min(10, len(patch_dataset))):  # Test first 10 patches
            sample = patch_dataset[i]

            # Verify sample structure
            assert "features" in sample
            assert "coordinates" in sample
            assert "slide_id" in sample
            assert "patch_idx" in sample

            # Verify dimensions
            assert sample["features"].shape == torch.Size([2048])
            assert sample["coordinates"].shape == torch.Size([2])

            # Verify patch index is valid
            assert 0 <= sample["patch_idx"] < 1000  # Reasonable patch index range

    def test_slide_level_feature_aggregation(self, sample_slide_data):
        """Test slide-level feature aggregation maintains coordinate information."""
        dataset, data_dir = sample_slide_data

        # Create slide index
        slides = [
            SlideMetadata(
                slide_id=entry["slide_id"],
                patient_id=entry["patient_id"],
                file_path=f"features/{entry['slide_id']}.h5",
                label=entry["label"],
                split="train",
            )
            for entry in dataset["slide_index"]
        ]

        slide_index = CAMELYONSlideIndex(slides)

        # Create slide dataset
        slide_dataset = CAMELYONSlideDataset(
            slide_index=slide_index,
            features_dir=str(data_dir / "features"),
            split="train",
        )

        # Test slide-level data
        for i in range(len(slide_dataset)):
            sample = slide_dataset[i]

            # Verify slide structure
            assert "slide_id" in sample
            assert "features" in sample
            assert "coordinates" in sample
            assert "num_patches" in sample

            # Verify alignment
            features = sample["features"]
            coordinates = sample["coordinates"]
            num_patches = sample["num_patches"]

            assert features.shape[0] == coordinates.shape[0] == num_patches
            assert features.shape[1] == 2048
            assert coordinates.shape[1] == 2

    def test_coordinate_range_validation(self, sample_slide_data):
        """Test that coordinates are within expected ranges."""
        dataset, data_dir = sample_slide_data

        # Load HDF5 files directly to check coordinate ranges
        features_dir = data_dir / "features"

        for slide_entry in dataset["slide_index"]:
            slide_id = slide_entry["slide_id"]
            feature_path = features_dir / f"{slide_id}.h5"

            with h5py.File(feature_path, "r") as f:
                coordinates = f["coordinates"][:]

                # Check coordinate ranges (should be within 0-10000 based on generator)
                assert np.all(coordinates >= 0)
                assert np.all(coordinates <= 10000)

                # Check coordinate data type
                assert coordinates.dtype == np.int32

                # Check no NaN or infinite values
                assert np.all(np.isfinite(coordinates))

    def test_feature_normalization_consistency(self, sample_slide_data):
        """Test that feature vectors maintain consistent normalization."""
        dataset, data_dir = sample_slide_data

        features_dir = data_dir / "features"

        for slide_entry in dataset["slide_index"]:
            slide_id = slide_entry["slide_id"]
            feature_path = features_dir / f"{slide_id}.h5"

            with h5py.File(feature_path, "r") as f:
                features = f["features"][:]

                # Check feature data type
                assert features.dtype == np.float32

                # Check no NaN or infinite values
                assert np.all(np.isfinite(features))

                # Check feature normalization (L2 normalized vectors should have norm ~1)
                norms = np.linalg.norm(features, axis=1)
                assert np.allclose(norms, 1.0, atol=1e-5)
