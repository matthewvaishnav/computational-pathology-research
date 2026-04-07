"""
Unit tests for the PatchCamelyon dataset helpers.

These tests target the current HDF5-backed dataset API and the
PCamDatasetWithFeatures wrapper used by the PCam experiment scripts.
"""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torchvision import transforms

from src.data.pcam_dataset import (
    PCamDataset,
    PCamDatasetWithFeatures,
    get_pcam_transforms,
)


def _write_split(root_dir: Path, split: str, images: np.ndarray, labels: np.ndarray) -> None:
    """Create one HDF5-backed PCam split."""
    split_dir = root_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(split_dir / "images.h5py", "w") as f:
        f.create_dataset("images", data=images)

    with h5py.File(split_dir / "labels.h5py", "w") as f:
        f.create_dataset("labels", data=labels)


def _write_metadata(root_dir: Path, train_count: int = 0, val_count: int = 0, test_count: int = 0) -> None:
    """Write lightweight metadata.json used by the dataset."""
    metadata = {
        "dataset": "PCam",
        "splits": {
            "train": {"num_samples": train_count},
            "val": {"num_samples": val_count},
            "test": {"num_samples": test_count},
        },
        "image_shape": [96, 96, 3],
        "num_classes": 2,
    }
    with open(root_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for PCam test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_pcam_data(temp_data_dir: Path):
    """Create one valid train split with RGB images."""
    num_samples = 5
    images = np.random.randint(0, 256, size=(num_samples, 96, 96, 3), dtype=np.uint8)
    labels = np.random.randint(0, 2, size=num_samples, dtype=np.int64)
    _write_split(temp_data_dir, "train", images, labels)
    _write_metadata(temp_data_dir, train_count=num_samples)
    return temp_data_dir, images, labels


class TestPCamDataset:
    """Tests for the base PCam dataset implementation."""

    def test_initialization_with_existing_data(self, mock_pcam_data):
        """Dataset initializes from HDF5-backed split data."""
        data_dir, _, labels = mock_pcam_data

        dataset = PCamDataset(root_dir=str(data_dir), split="train", download=False)

        assert len(dataset) == len(labels)
        assert dataset.split == "train"
        assert dataset.metadata["dataset"] == "PCam"

    def test_initialization_invalid_split(self, temp_data_dir):
        """Invalid split names should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid split"):
            PCamDataset(root_dir=str(temp_data_dir), split="invalid", download=False)

    def test_initialization_missing_data_no_download(self, temp_data_dir):
        """Missing data with download disabled should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Dataset not found"):
            PCamDataset(root_dir=str(temp_data_dir), split="train", download=False)

    def test_getitem_returns_expected_fields(self, mock_pcam_data):
        """Samples should contain image, label, and image_id."""
        data_dir, _, labels = mock_pcam_data
        dataset = PCamDataset(root_dir=str(data_dir), split="train", download=False)

        sample = dataset[0]

        assert set(sample.keys()) == {"image", "label", "image_id"}
        assert sample["image"].shape == (3, 96, 96)
        assert sample["image"].dtype == torch.float32
        assert sample["label"].dtype == torch.long
        assert sample["label"].item() in {0, 1}
        assert sample["label"].item() == int(labels[0])
        assert sample["image_id"] == "train_00000000"

    def test_default_transform_returns_unit_range_tensor(self, mock_pcam_data):
        """Default ToTensor transform should scale images to [0, 1]."""
        data_dir, _, _ = mock_pcam_data
        dataset = PCamDataset(root_dir=str(data_dir), split="train", download=False)

        image = dataset[0]["image"]

        assert image.min().item() >= 0.0
        assert image.max().item() <= 1.0

    def test_custom_transform_is_applied(self, mock_pcam_data):
        """Custom transforms should override the default preprocessing."""
        data_dir, _, _ = mock_pcam_data
        zero_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.zeros_like(x)),
            ]
        )
        dataset = PCamDataset(
            root_dir=str(data_dir),
            split="train",
            transform=zero_transform,
            download=False,
        )

        image = dataset[0]["image"]

        assert torch.count_nonzero(image) == 0

    def test_grayscale_images_are_expanded_to_three_channels(self, temp_data_dir):
        """Grayscale images should be repeated across RGB channels."""
        images = np.random.randint(0, 256, size=(3, 96, 96), dtype=np.uint8)
        labels = np.array([0, 1, 0], dtype=np.int64)
        _write_split(temp_data_dir, "train", images, labels)
        _write_metadata(temp_data_dir, train_count=3)

        dataset = PCamDataset(root_dir=str(temp_data_dir), split="train", download=False)
        image = dataset[0]["image"]

        assert image.shape == (3, 96, 96)
        assert torch.allclose(image[0], image[1])
        assert torch.allclose(image[1], image[2])

    def test_invalid_channel_count_returns_placeholder_and_warning(self, temp_data_dir):
        """Invalid image channel counts should fall back to a placeholder sample."""
        images = np.random.randint(0, 256, size=(1, 96, 96, 4), dtype=np.uint8)
        labels = np.array([1], dtype=np.int64)
        _write_split(temp_data_dir, "train", images, labels)
        _write_metadata(temp_data_dir, train_count=1)

        dataset = PCamDataset(root_dir=str(temp_data_dir), split="train", download=False)

        with pytest.warns(RuntimeWarning, match="Returning zeros as placeholder"):
            sample = dataset[0]

        assert sample["image"].shape == (3, 96, 96)
        assert torch.count_nonzero(sample["image"]) == 0
        assert sample["label"].item() == 0
        assert sample["image_id"].endswith("_error")

    def test_mismatched_images_and_labels_raises_runtime_error(self, temp_data_dir):
        """Image/label count mismatches should fail during initialization."""
        images = np.random.randint(0, 256, size=(4, 96, 96, 3), dtype=np.uint8)
        labels = np.array([0, 1], dtype=np.int64)
        _write_split(temp_data_dir, "train", images, labels)
        _write_metadata(temp_data_dir, train_count=4)

        with pytest.raises(RuntimeError, match="does not match"):
            PCamDataset(root_dir=str(temp_data_dir), split="train", download=False)

    def test_out_of_bounds_index_raises_index_error(self, mock_pcam_data):
        """Out-of-range indices should raise IndexError."""
        data_dir, _, labels = mock_pcam_data
        dataset = PCamDataset(root_dir=str(data_dir), split="train", download=False)

        with pytest.raises(IndexError, match=f"dataset with {len(labels)} samples"):
            dataset[len(labels)]

    def test_repr_includes_split_and_num_samples(self, mock_pcam_data):
        """repr should include key dataset details."""
        data_dir, _, labels = mock_pcam_data
        dataset = PCamDataset(root_dir=str(data_dir), split="train", download=False)

        dataset_repr = repr(dataset)

        assert "PCamDataset(" in dataset_repr
        assert "split='train'" in dataset_repr
        assert f"num_samples={len(labels)}" in dataset_repr


class TestPCamDatasetWithFeatures:
    """Tests for the optional features wrapper dataset."""

    def test_returns_preextracted_features_when_available(self, mock_pcam_data):
        """Feature wrapper should attach wsi_features from HDF5."""
        data_dir, _, labels = mock_pcam_data
        features_path = data_dir / "train_features.h5py"
        features = np.random.randn(len(labels), 512).astype(np.float32)
        with h5py.File(features_path, "w") as f:
            f.create_dataset("features", data=features)

        dataset = PCamDatasetWithFeatures(
            root_dir=str(data_dir),
            split="train",
            features_path=str(features_path),
            download=False,
        )

        sample = dataset[0]

        assert sample["image"].shape == (3, 96, 96)
        assert sample["wsi_features"].shape == (512,)
        assert sample["wsi_features"].dtype == torch.float32

    def test_missing_feature_file_returns_raw_images_only(self, mock_pcam_data):
        """Missing features file should not break sample loading."""
        data_dir, _, _ = mock_pcam_data
        dataset = PCamDatasetWithFeatures(
            root_dir=str(data_dir),
            split="train",
            features_path=str(data_dir / "missing_features.h5py"),
            download=False,
        )

        sample = dataset[0]

        assert set(sample.keys()) == {"image", "label", "image_id"}


class TestGetPCamTransforms:
    """Tests for the standard PCam transform helpers."""

    def test_train_transforms_include_augmentation(self):
        """Training transforms should include augmentation before normalization."""
        transform = get_pcam_transforms(split="train", augmentation=True)

        assert isinstance(transform, transforms.Compose)
        transform_types = [type(step) for step in transform.transforms]
        assert transform_types == [
            transforms.RandomHorizontalFlip,
            transforms.RandomVerticalFlip,
            transforms.ColorJitter,
            transforms.ToTensor,
            transforms.Normalize,
        ]

    def test_eval_transforms_are_minimal(self):
        """Validation/test transforms should only tensorize and normalize."""
        transform = get_pcam_transforms(split="val", augmentation=False)

        assert isinstance(transform, transforms.Compose)
        transform_types = [type(step) for step in transform.transforms]
        assert transform_types == [transforms.ToTensor, transforms.Normalize]

    def test_eval_transforms_produce_expected_shape(self):
        """Returned transforms should work with a standard RGB PIL image."""
        transform = get_pcam_transforms(split="test", augmentation=False)
        image = np.random.randint(0, 256, size=(96, 96, 3), dtype=np.uint8)

        tensor = transform(image)

        assert tensor.shape == (3, 96, 96)
        assert tensor.dtype == torch.float32
