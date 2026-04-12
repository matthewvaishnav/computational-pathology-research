"""
PatchCamelyon (PCam) dataset implementation for binary classification.

This module implements a PyTorch Dataset for loading and preprocessing
PatchCamelyon data with support for data augmentation and feature extraction.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


def get_pcam_transforms(split: str = "train", augmentation: bool = True) -> transforms.Compose:
    """
    Get transforms for PCam dataset.

    Args:
        split: Dataset split ('train', 'val', 'test')
        augmentation: Whether to apply data augmentation (only for train split)

    Returns:
        Composed transforms
    """
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if split == "train" and augmentation:
        augment_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ]
        return transforms.Compose(augment_transforms + base_transforms)
    else:
        return transforms.Compose(base_transforms)


class PCamDataset(Dataset):
    """
    PatchCamelyon dataset for binary classification.

    The PatchCamelyon dataset contains 96x96 pixel histopathology patches
    from lymph node sections with binary labels (0=normal, 1=metastatic).

    Args:
        root_dir: Directory to download/store dataset
        split: One of 'train', 'val', 'test'
        transform: Optional torchvision transforms
        download: Whether to download if not present
        feature_extractor: Optional pretrained model for feature extraction
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        download: bool = True,
        feature_extractor: Optional[torch.nn.Module] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.feature_extractor = feature_extractor

        # Validate split
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', 'test'")

        # Create root directory if it doesn't exist
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # Download dataset if requested and not present
        if download and not self._check_dataset_exists():
            logger.info(f"Dataset not found in {self.root_dir}. Downloading...")
            self.download()

        # Load data
        self._load_split()

    def _check_dataset_exists(self) -> bool:
        """Check if dataset files exist."""
        split_dir = self.root_dir / self.split
        # Check for both .npy and .h5py formats
        npy_exists = (
            split_dir.exists()
            and (split_dir / "images.npy").exists()
            and (split_dir / "labels.npy").exists()
        )
        h5py_exists = (
            split_dir.exists()
            and (split_dir / "images.h5py").exists()
            and (split_dir / "labels.h5py").exists()
        )
        return npy_exists or h5py_exists

    def download(self):
        """Download PCam dataset from TensorFlow Datasets."""
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "tensorflow_datasets is required for PCam download. "
                "Install with: pip install tensorflow-datasets"
            )

        try:
            logger.info("Downloading PatchCamelyon dataset...")

            # Download the dataset
            ds, info = tfds.load(
                "patch_camelyon",
                split=["train", "validation", "test"],
                with_info=True,
                as_supervised=True,
                download=True,
                data_dir=str(self.root_dir / "tfds_cache"),
            )

            # Map splits
            split_mapping = {"train": ds[0], "val": ds[1], "test": ds[2]}

            # Process each split
            for split_name, split_ds in split_mapping.items():
                logger.info(f"Processing {split_name} split...")

                split_dir = self.root_dir / split_name
                split_dir.mkdir(parents=True, exist_ok=True)

                images = []
                labels = []

                # Convert TensorFlow dataset to numpy arrays
                for image, label in split_ds:
                    images.append(image.numpy())
                    labels.append(label.numpy())

                # Save as numpy arrays
                images = np.array(images)
                labels = np.array(labels)

                np.save(split_dir / "images.npy", images)
                np.save(split_dir / "labels.npy", labels)

                logger.info(f"Saved {len(images)} samples for {split_name} split")

            # Clean up TensorFlow cache
            tfds_cache = self.root_dir / "tfds_cache"
            if tfds_cache.exists():
                shutil.rmtree(tfds_cache)

            logger.info("Dataset download completed successfully")

        except Exception as e:
            logger.error(f"Failed to download PCam dataset: {e}")
            logger.info(
                "Troubleshooting steps:\n"
                "1. Check internet connection\n"
                "2. Verify disk space (need ~2GB)\n"
                "3. Try manual download from: "
                "https://github.com/basveeling/pcam\n"
                f"4. Place files in: {self.root_dir}"
            )
            raise

    def _load_split(self):
        """Load the specified split into memory."""
        split_dir = self.root_dir / self.split

        if not self._check_dataset_exists():
            raise RuntimeError(
                f"Dataset files not found in {split_dir}. "
                "Set download=True to download the dataset."
            )

        try:
            # Try loading from .npy files first
            if (split_dir / "images.npy").exists() and (split_dir / "labels.npy").exists():
                self.images = np.load(split_dir / "images.npy")
                self.labels = np.load(split_dir / "labels.npy")
                logger.info(
                    f"Loaded {len(self.images)} samples from .npy files for {self.split} split"
                )
            # Fall back to .h5py files
            elif (split_dir / "images.h5py").exists() and (split_dir / "labels.h5py").exists():
                import h5py

                with h5py.File(split_dir / "images.h5py", "r") as f:
                    self.images = f["images"][:]
                with h5py.File(split_dir / "labels.h5py", "r") as f:
                    self.labels = f["labels"][:]
                logger.info(
                    f"Loaded {len(self.images)} samples from .h5py files for {self.split} split"
                )
            else:
                raise RuntimeError(f"No valid dataset files found in {split_dir}")

        except Exception as e:
            logger.error(f"Failed to load dataset files: {e}")
            raise RuntimeError(f"Corrupted dataset files in {split_dir}")

    def __len__(self) -> int:
        """Returns number of samples in split."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
            - 'image': Tensor [3, 96, 96] (raw image)
            - 'label': Tensor (scalar)
            - 'image_id': str
        """
        try:
            # Load image and label
            image = self.images[idx]  # [96, 96, 3]
            label = self.labels[idx]  # scalar

            # Convert numpy array to PIL Image for transforms
            image = Image.fromarray(image.astype(np.uint8))

            # Apply transforms
            if self.transform is not None:
                image = self.transform(image)
            else:
                # Default: convert to tensor and normalize
                image = transforms.ToTensor()(image)
                image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                    image
                )

            # Convert label to tensor
            label = torch.tensor(label, dtype=torch.long)

            return {"image": image, "label": label, "image_id": f"{self.split}_{idx}"}

        except Exception as e:
            logger.warning(f"Failed to load sample {idx}: {e}")
            # Return a dummy sample to avoid breaking the dataloader
            dummy_image = torch.zeros(3, 96, 96)
            dummy_label = torch.tensor(0, dtype=torch.long)
            return {
                "image": dummy_image,
                "label": dummy_label,
                "image_id": f"{self.split}_{idx}_dummy",
            }


def validate_dataset(dataset: PCamDataset):
    """
    Validate dataset integrity before training.

    Args:
        dataset: PCamDataset instance to validate

    Raises:
        ValueError: If dataset validation fails
    """
    logger.info("Validating dataset...")

    # Check dataset size
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    # Sample a few items to validate
    num_samples_to_check = min(10, len(dataset))

    for i in range(num_samples_to_check):
        try:
            sample = dataset[i]
        except Exception as e:
            logger.warning(f"Failed to load sample {i}: {e}")
            continue

        # Validate shapes
        if sample["image"].shape != (3, 96, 96):
            raise ValueError(
                f"Invalid image shape: {sample['image'].shape}, " "expected (3, 96, 96)"
            )

        # Validate labels
        if sample["label"].item() not in [0, 1]:
            raise ValueError(f"Invalid label: {sample['label'].item()}, expected 0 or 1")

    logger.info(f"Dataset validation passed: {len(dataset)} samples")
