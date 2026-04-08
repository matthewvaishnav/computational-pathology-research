"""
Tests for CAMELYON dataset and index utilities.
"""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.data.camelyon_dataset import (
    CAMELYONPatchDataset,
    CAMELYONSlideIndex,
    SlideAggregator,
    SlideMetadata,
    create_patch_index,
    validate_feature_file,
)


class TestSlideMetadata:
    """Test SlideMetadata dataclass."""

    def test_basic_creation(self):
        """Can create SlideMetadata with required fields."""
        slide = SlideMetadata(
            slide_id="patient_001_node_0",
            patient_id="patient_001",
            file_path="/data/slides/patient_001_node_0.tif",
            label=1,
            split="train",
        )
        assert slide.slide_id == "patient_001_node_0"
        assert slide.label == 1
        assert slide.split == "train"
        assert slide.annotation_path is None

    def test_full_creation(self):
        """Can create SlideMetadata with all optional fields."""
        slide = SlideMetadata(
            slide_id="patient_001_node_0",
            patient_id="patient_001",
            file_path="/data/slides/patient_001_node_0.tif",
            label=1,
            split="train",
            annotation_path="/data/annotations/patient_001_node_0.xml",
            width=100000,
            height=80000,
            magnification=40.0,
            mpp=0.25,
        )
        assert slide.width == 100000
        assert slide.height == 80000
        assert slide.magnification == 40.0
        assert slide.mpp == 0.25


class TestCAMELYONSlideIndex:
    """Test CAMELYONSlideIndex functionality."""

    @pytest.fixture
    def sample_slides(self):
        """Create sample slide metadata."""
        return [
            SlideMetadata(
                "patient_001_node_0", "patient_001", "/path/1.tif", 1, "train", "/ann/1.xml"
            ),
            SlideMetadata("patient_001_node_1", "patient_001", "/path/2.tif", 0, "train"),
            SlideMetadata(
                "patient_002_node_0", "patient_002", "/path/3.tif", 1, "val", "/ann/3.xml"
            ),
            SlideMetadata("patient_003_node_0", "patient_003", "/path/4.tif", 0, "test"),
            SlideMetadata(
                "patient_003_node_1", "patient_003", "/path/5.tif", 1, "test", "/ann/5.xml"
            ),
        ]

    @pytest.fixture
    def index(self, sample_slides):
        """Create a slide index with sample slides."""
        return CAMELYONSlideIndex(sample_slides)

    def test_length(self, index):
        """Index reports correct length."""
        assert len(index) == 5

    def test_getitem(self, index):
        """Can retrieve slide by ID."""
        slide = index["patient_001_node_0"]
        assert slide.slide_id == "patient_001_node_0"
        assert slide.label == 1

    def test_get_slides_by_split(self, index):
        """Can filter slides by split."""
        train_slides = index.get_slides_by_split("train")
        assert len(train_slides) == 2
        assert all(s.split == "train" for s in train_slides)

        val_slides = index.get_slides_by_split("val")
        assert len(val_slides) == 1

        test_slides = index.get_slides_by_split("test")
        assert len(test_slides) == 2

    def test_get_slides_by_patient(self, index):
        """Can filter slides by patient."""
        patient_001_slides = index.get_slides_by_patient("patient_001")
        assert len(patient_001_slides) == 2
        assert all(s.patient_id == "patient_001" for s in patient_001_slides)

        patient_003_slides = index.get_slides_by_patient("patient_003")
        assert len(patient_003_slides) == 2

    def test_get_annotated_slides(self, index):
        """Can get slides with annotations."""
        annotated = index.get_annotated_slides()
        assert len(annotated) == 3
        assert all(s.annotation_path is not None for s in annotated)

    def test_save_and_load(self, index, tmp_path):
        """Can save and reload index."""
        output_path = tmp_path / "slide_index.json"
        index.save(str(output_path))

        assert output_path.exists()

        # Check JSON structure
        with open(output_path) as f:
            data = json.load(f)
        assert data["dataset"] == "CAMELYON"
        assert data["num_slides"] == 5
        assert len(data["slides"]) == 5

        # Load back
        loaded = CAMELYONSlideIndex.load(str(output_path))
        assert len(loaded) == 5
        assert loaded["patient_001_node_0"].label == 1

    def test_load_missing_file(self, tmp_path):
        """Loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            CAMELYONSlideIndex.load(str(tmp_path / "nonexistent.json"))

    def test_from_directory(self, tmp_path):
        """Can create index from directory of mock slide files."""
        slide_dir = tmp_path / "slides"
        slide_dir.mkdir()

        # Create mock slide files
        for i in range(5):
            (slide_dir / f"patient_{i:03d}_node_0.tif").touch()

        index = CAMELYONSlideIndex.from_directory(
            root_dir=str(slide_dir),
            slide_pattern="*.tif",
            split_ratios=(0.6, 0.2, 0.2),
            seed=42,
        )

        assert len(index) == 5
        # All splits should have at least one slide with this seed
        assert len(index.get_slides_by_split("train")) >= 1
        assert len(index.get_slides_by_split("val")) >= 1
        assert len(index.get_slides_by_split("test")) >= 1

    def test_patient_id_extraction(self, tmp_path):
        """Patient ID correctly extracted from filename."""
        slide_dir = tmp_path / "slides"
        slide_dir.mkdir()

        (slide_dir / "patient_042_node_3.tif").touch()
        (slide_dir / "random_name.tif").touch()

        index = CAMELYONSlideIndex.from_directory(str(slide_dir))

        slide1 = index["patient_042_node_3"]
        assert slide1.patient_id == "patient_042"

        slide2 = index["random_name"]
        assert slide2.patient_id == "random_name"


class TestSlideAggregator:
    """Test SlideAggregator helper for MIL aggregation."""

    def test_add_predictions(self):
        """Can add patch predictions grouped by slide."""
        aggregator = SlideAggregator()

        # Add first batch
        slide_ids = ["slide_001", "slide_001", "slide_002"]
        predictions = torch.tensor([[0.1], [0.2], [0.3]])
        aggregator.add_predictions(slide_ids, predictions)

        # Add second batch
        slide_ids2 = ["slide_001", "slide_002"]
        predictions2 = torch.tensor([[0.4], [0.5]])
        aggregator.add_predictions(slide_ids2, predictions2)

        # Check internal state
        assert len(aggregator.slide_data["slide_001"]["predictions"]) == 3
        assert len(aggregator.slide_data["slide_002"]["predictions"]) == 2

    def test_get_slide_predictions_max(self):
        """Can aggregate with max pooling."""
        aggregator = SlideAggregator()

        slide_ids = ["slide_001", "slide_001", "slide_001"]
        predictions = torch.tensor([[0.1], [0.5], [0.3]])
        aggregator.add_predictions(slide_ids, predictions)

        slide_preds = aggregator.get_slide_predictions(aggregation="max")
        assert slide_preds["slide_001"] == pytest.approx(0.5)

    def test_get_slide_predictions_mean(self):
        """Can aggregate with mean pooling."""
        aggregator = SlideAggregator()

        slide_ids = ["slide_001", "slide_001"]
        predictions = torch.tensor([[0.1], [0.3]])
        aggregator.add_predictions(slide_ids, predictions)

        slide_preds = aggregator.get_slide_predictions(aggregation="mean")
        assert slide_preds["slide_001"] == pytest.approx(0.2)

    def test_get_slide_predictions_attention(self):
        """Can aggregate with attention-weighted pooling."""
        aggregator = SlideAggregator()

        slide_ids = ["slide_001", "slide_001", "slide_001"]
        predictions = torch.tensor([[0.1], [0.3], [0.5]])
        attention = torch.tensor([0.1, 0.3, 0.6])  # Third patch most important
        aggregator.add_predictions(slide_ids, predictions, attention)

        slide_preds = aggregator.get_slide_predictions(aggregation="attention")
        # Weighted average: 0.1*0.1 + 0.3*0.3 + 0.6*0.5 = 0.01 + 0.09 + 0.3 = 0.4
        expected = 0.1 * 0.1 + 0.3 * 0.3 + 0.6 * 0.5
        assert slide_preds["slide_001"] == pytest.approx(expected, abs=0.01)

    def test_add_features_with_coordinates(self):
        """Can add features and coordinates."""
        aggregator = SlideAggregator()

        slide_ids = ["slide_001", "slide_001"]
        features = torch.randn(2, 2048)
        coordinates = torch.tensor([[100, 200], [300, 400]])
        aggregator.add_features(slide_ids, features, coordinates)

        slide_data = aggregator.get_slide_features("slide_001")
        assert slide_data is not None
        assert slide_data["features"].shape == (2, 2048)
        assert slide_data["coordinates"].shape == (2, 2)

    def test_get_slide_ids(self):
        """Can get list of all slide IDs."""
        aggregator = SlideAggregator()

        slide_ids = ["slide_001", "slide_002", "slide_003"]
        predictions = torch.randn(3, 1)
        aggregator.add_predictions(slide_ids, predictions)

        ids = aggregator.get_slide_ids()
        assert len(ids) == 3
        assert "slide_001" in ids

    def test_clear(self):
        """Can clear all aggregated data."""
        aggregator = SlideAggregator()

        slide_ids = ["slide_001"]
        predictions = torch.randn(1, 1)
        aggregator.add_predictions(slide_ids, predictions)

        assert len(aggregator.get_slide_ids()) == 1

        aggregator.clear()
        assert len(aggregator.get_slide_ids()) == 0


class TestCAMELYONPatchDataset:
    """Test CAMELYONPatchDataset functionality."""

    @pytest.fixture
    def mock_features_dir(self, tmp_path):
        """Create mock HDF5 feature files."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()

        # Create feature files for 3 slides
        for slide_id, num_patches in [("slide_001", 10), ("slide_002", 15), ("slide_003", 8)]:
            with h5py.File(features_dir / f"{slide_id}.h5", "w") as f:
                # Features: [num_patches, 2048]
                f.create_dataset(
                    "features", data=np.random.randn(num_patches, 2048).astype(np.float32)
                )
                # Coordinates: [num_patches, 2]
                f.create_dataset(
                    "coordinates",
                    data=np.random.randint(0, 10000, (num_patches, 2)).astype(np.int32),
                )

        return features_dir

    @pytest.fixture
    def slide_index(self):
        """Create slide index for testing."""
        slides = [
            SlideMetadata("slide_001", "patient_001", "/path/1.tif", 1, "train"),
            SlideMetadata("slide_002", "patient_001", "/path/2.tif", 0, "train"),
            SlideMetadata("slide_003", "patient_002", "/path/3.tif", 1, "val"),
        ]
        return CAMELYONSlideIndex(slides)

    def test_dataset_length(self, slide_index, mock_features_dir):
        """Dataset reports correct total number of patches."""
        dataset = CAMELYONPatchDataset(
            slide_index=slide_index,
            features_dir=str(mock_features_dir),
            split="train",
        )
        # 10 patches from slide_001 + 15 from slide_002 = 25
        assert len(dataset) == 25

    def test_getitem_returns_correct_structure(self, slide_index, mock_features_dir):
        """__getitem__ returns sample with expected structure."""
        dataset = CAMELYONPatchDataset(
            slide_index=slide_index,
            features_dir=str(mock_features_dir),
            split="train",
        )

        sample = dataset[0]

        assert "features" in sample
        assert "coordinates" in sample
        assert "slide_id" in sample
        assert "patient_id" in sample
        assert "label" in sample
        assert "patch_idx" in sample

        assert sample["features"].shape == torch.Size([2048])
        assert sample["coordinates"].shape == torch.Size([2])
        assert isinstance(sample["slide_id"], str)
        assert isinstance(sample["label"], int)

    def test_val_split_only(self, slide_index, mock_features_dir):
        """Val split only contains val slides."""
        dataset = CAMELYONPatchDataset(
            slide_index=slide_index,
            features_dir=str(mock_features_dir),
            split="val",
        )

        assert len(dataset) == 8  # Only slide_003

        sample = dataset[0]
        assert sample["slide_id"] == "slide_003"
        assert sample["label"] == 1

    def test_get_slide_features(self, slide_index, mock_features_dir):
        """Can retrieve all features for a specific slide."""
        dataset = CAMELYONPatchDataset(
            slide_index=slide_index,
            features_dir=str(mock_features_dir),
            split="train",
        )

        features = dataset.get_slide_features("slide_001")
        assert features is not None
        assert features.shape == torch.Size([10, 2048])

    def test_get_slide_patch_data(self, slide_index, mock_features_dir):
        """Can retrieve slide-level patch features and coordinates together."""
        dataset = CAMELYONPatchDataset(
            slide_index=slide_index,
            features_dir=str(mock_features_dir),
            split="train",
        )

        slide_data = dataset.get_slide_patch_data("slide_001")

        assert slide_data is not None
        assert slide_data["slide_id"] == "slide_001"
        assert slide_data["patient_id"] == "patient_001"
        assert slide_data["label"] == 1
        assert slide_data["features"].shape == torch.Size([10, 2048])
        assert slide_data["coordinates"].shape == torch.Size([10, 2])
        assert slide_data["num_patches"] == 10

    def test_aggregate_slide_features(self, slide_index, mock_features_dir):
        """Can aggregate patch features to slide-level vectors."""
        dataset = CAMELYONPatchDataset(
            slide_index=slide_index,
            features_dir=str(mock_features_dir),
            split="train",
        )

        mean_features = dataset.aggregate_slide_features("slide_001", method="mean")
        max_features = dataset.aggregate_slide_features("slide_001", method="max")

        assert mean_features is not None
        assert max_features is not None
        assert mean_features.shape == torch.Size([2048])
        assert max_features.shape == torch.Size([2048])

    def test_aggregate_slide_features_invalid_method(self, slide_index, mock_features_dir):
        """Invalid aggregation method raises ValueError."""
        dataset = CAMELYONPatchDataset(
            slide_index=slide_index,
            features_dir=str(mock_features_dir),
            split="train",
        )

        with pytest.raises(ValueError, match="Unknown aggregation method"):
            dataset.aggregate_slide_features("slide_001", method="median")

    def test_get_slide_features_missing(self, slide_index, mock_features_dir, tmp_path):
        """Returns None for missing slide."""
        dataset = CAMELYONPatchDataset(
            slide_index=slide_index,
            features_dir=str(mock_features_dir),
            split="train",
        )

        features = dataset.get_slide_features("nonexistent_slide")
        assert features is None

    def test_missing_feature_file_warning(self, slide_index, tmp_path):
        """Warns when feature file is missing."""
        # Create features_dir but don't populate it
        features_dir = tmp_path / "empty_features"
        features_dir.mkdir()

        dataset = CAMELYONPatchDataset(
            slide_index=slide_index,
            features_dir=str(features_dir),
            split="train",
        )

        # Should have 0 patches since no feature files exist
        assert len(dataset) == 0


class TestValidateFeatureFile:
    """Test feature file validation."""

    def test_valid_file(self, tmp_path):
        """Valid file passes validation."""
        feature_path = tmp_path / "slide_001.h5"
        with h5py.File(feature_path, "w") as f:
            f.create_dataset("features", data=np.random.randn(10, 2048).astype(np.float32))
            f.create_dataset(
                "coordinates", data=np.random.randint(0, 1000, (10, 2)).astype(np.int32)
            )

        result = validate_feature_file(str(feature_path))
        assert result["valid"] is True
        assert result["num_patches"] == 10
        assert result["feature_dim"] == 2048
        assert result["error"] is None

    def test_missing_file(self, tmp_path):
        """Missing file reports error."""
        result = validate_feature_file(str(tmp_path / "nonexistent.h5"))
        assert result["valid"] is False
        assert "not found" in result["error"].lower()

    def test_missing_features_dataset(self, tmp_path):
        """Missing features dataset reports error."""
        feature_path = tmp_path / "slide_001.h5"
        with h5py.File(feature_path, "w") as f:
            f.create_dataset("coordinates", data=np.random.randint(0, 1000, (10, 2)))

        result = validate_feature_file(str(feature_path))
        assert result["valid"] is False
        assert "missing 'features'" in result["error"].lower()

    def test_missing_coordinates_dataset(self, tmp_path):
        """Missing coordinates dataset reports error."""
        feature_path = tmp_path / "slide_001.h5"
        with h5py.File(feature_path, "w") as f:
            f.create_dataset("features", data=np.random.randn(10, 2048))

        result = validate_feature_file(str(feature_path))
        assert result["valid"] is False
        assert "missing 'coordinates'" in result["error"].lower()

    def test_mismatched_patch_counts(self, tmp_path):
        """Mismatched patch counts reports error."""
        feature_path = tmp_path / "slide_001.h5"
        with h5py.File(feature_path, "w") as f:
            f.create_dataset("features", data=np.random.randn(10, 2048))
            f.create_dataset("coordinates", data=np.random.randint(0, 1000, (8, 2)))

        result = validate_feature_file(str(feature_path))
        assert result["valid"] is False
        assert "mismatched" in result["error"].lower()

    def test_invalid_coordinates_shape(self, tmp_path):
        """Invalid coordinates shape reports error."""
        feature_path = tmp_path / "slide_001.h5"
        with h5py.File(feature_path, "w") as f:
            f.create_dataset("features", data=np.random.randn(10, 2048))
            f.create_dataset("coordinates", data=np.random.randint(0, 1000, (10, 3)))  # Wrong shape

        result = validate_feature_file(str(feature_path))
        assert result["valid"] is False
        assert "[n, 2]" in result["error"].lower()


class TestCreatePatchIndex:
    """Test create_patch_index utility."""

    @pytest.fixture
    def setup(self, tmp_path):
        """Create mock data for testing."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()

        # Create feature files
        for slide_id, num_patches in [("slide_001", 5), ("slide_002", 3)]:
            with h5py.File(features_dir / f"{slide_id}.h5", "w") as f:
                f.create_dataset(
                    "features", data=np.random.randn(num_patches, 512).astype(np.float32)
                )
                f.create_dataset(
                    "coordinates",
                    data=np.random.randint(0, 1000, (num_patches, 2)).astype(np.int32),
                )

        slides = [
            SlideMetadata("slide_001", "patient_001", "/path/1.tif", 1, "train"),
            SlideMetadata("slide_002", "patient_001", "/path/2.tif", 0, "val"),
        ]
        index = CAMELYONSlideIndex(slides)

        return index, features_dir

    def test_creates_patch_index(self, setup, tmp_path):
        """Creates patch index JSON file."""
        index, features_dir = setup
        output_path = tmp_path / "patch_index.json"

        counts = create_patch_index(
            slide_index=index,
            features_dir=str(features_dir),
            output_path=str(output_path),
            splits=["train", "val"],
        )

        assert output_path.exists()
        assert counts["train"] == 5
        assert counts["val"] == 3

        # Verify JSON structure
        with open(output_path) as f:
            data = json.load(f)

        assert "train" in data
        assert "val" in data
        assert len(data["train"]) == 5
        assert len(data["val"]) == 3

        # Check first entry structure
        entry = data["train"][0]
        assert "global_idx" in entry
        assert "slide_id" in entry
        assert "patch_idx" in entry
        assert "patient_id" in entry
        assert "label" in entry
