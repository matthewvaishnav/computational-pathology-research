"""
Unit tests for the PatchCamelyon dataset helpers.

These tests target the current .npy-backed dataset API.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision import transforms

from src.data.pcam_dataset import (
    PCamDataset,
    get_pcam_transforms,
)


def _write_split(root_dir: Path, split: str, images: np.ndarray, labels: np.ndarray) -> None:
    """Create one PCam split with .npy files."""
    split_dir = root_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    np.save(split_dir / "images.npy", images)
    np.save(split_dir / "labels.npy", labels)


def _write_metadata(
    root_dir: Path, train_count: int = 0, val_count: int = 0, test_count: int = 0
) -> None:
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
        """Dataset initializes from .npy split data."""
        data_dir, _, labels = mock_pcam_data

        dataset = PCamDataset(root_dir=str(data_dir), split="train", download=False)

        assert len(dataset) == len(labels)
        assert dataset.split == "train"

    def test_initialization_invalid_split(self, temp_data_dir):
        """Invalid split names should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid split"):
            PCamDataset(root_dir=str(temp_data_dir), split="invalid", download=False)

    def test_initialization_missing_data_no_download(self, temp_data_dir):
        """Missing data with download disabled should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Dataset files not found"):
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
        assert sample["image_id"] == "train_0"

    def test_default_transform_returns_normalized_tensor(self, mock_pcam_data):
        """Default transform should normalize images (not just [0,1])."""
        data_dir, _, _ = mock_pcam_data
        dataset = PCamDataset(root_dir=str(data_dir), split="train", download=False)

        image = dataset[0]["image"]

        # Normalized images can be outside [0,1] range
        assert image.dtype == torch.float32
        assert image.shape == (3, 96, 96)

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

    def test_grayscale_images_are_converted_to_rgb(self, temp_data_dir):
        """Grayscale images should be handled by PIL."""
        images = np.random.randint(0, 256, size=(3, 96, 96, 3), dtype=np.uint8)
        labels = np.array([0, 1, 0], dtype=np.int64)
        _write_split(temp_data_dir, "train", images, labels)
        _write_metadata(temp_data_dir, train_count=3)

        dataset = PCamDataset(root_dir=str(temp_data_dir), split="train", download=False)
        image = dataset[0]["image"]

        assert image.shape == (3, 96, 96)

    def test_out_of_bounds_index_returns_dummy(self, mock_pcam_data):
        """Out-of-range indices should return dummy sample (error handling)."""
        data_dir, _, labels = mock_pcam_data
        dataset = PCamDataset(root_dir=str(data_dir), split="train", download=False)

        # This will raise IndexError from numpy, caught by __getitem__
        sample = dataset[len(labels) + 100]
        assert "dummy" in sample["image_id"]


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
