"""
PatchCamelyon (PCam) dataset implementation for binary classification.

This module implements a PyTorch Dataset for loading and preprocessing
PatchCamelyon data with support for data augmentation and feature extraction.
"""

import logging
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional

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
            error_msg = (
                "tensorflow_datasets is required for PCam download but is not installed.\n\n"
                "TROUBLESHOOTING STEPS:\n"
                "1. Install tensorflow-datasets:\n"
                "   pip install tensorflow-datasets\n"
                "   OR\n"
                "   conda install -c conda-forge tensorflow-datasets\n\n"
                "2. If installation fails, try installing TensorFlow first:\n"
                "   pip install tensorflow>=2.10.0\n\n"
                "3. For CPU-only systems, use:\n"
                "   pip install tensorflow-cpu tensorflow-datasets\n\n"
                "4. Verify installation:\n"
                "   python -c 'import tensorflow_datasets as tfds; print(tfds.__version__)'\n"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

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
            error_type = type(e).__name__
            error_msg = (
                f"Failed to download PatchCamelyon dataset: {error_type}: {str(e)}\n\n"
                "TROUBLESHOOTING STEPS:\n"
                "1. CHECK INTERNET CONNECTION:\n"
                "   - Verify you can access https://www.tensorflow.org\n"
                "   - Check if you're behind a proxy or firewall\n"
                "   - Try: curl -I https://storage.googleapis.com\n\n"
                "2. VERIFY DISK SPACE:\n"
                f"   - Required: ~2GB free space in {self.root_dir}\n"
                "   - Check available space: df -h (Linux/Mac) or dir (Windows)\n\n"
                "3. CHECK WRITE PERMISSIONS:\n"
                f"   - Ensure you have write access to: {self.root_dir}\n"
                "   - Try: touch {self.root_dir}/test.txt && rm {self.root_dir}/test.txt\n\n"
                "4. MANUAL DOWNLOAD (if automatic download fails):\n"
                "   a. Download from: https://github.com/basveeling/pcam\n"
                "   b. Extract files to the following structure:\n"
                f"      {self.root_dir}/train/images.npy\n"
                f"      {self.root_dir}/train/labels.npy\n"
                f"      {self.root_dir}/val/images.npy\n"
                f"      {self.root_dir}/val/labels.npy\n"
                f"      {self.root_dir}/test/images.npy\n"
                f"      {self.root_dir}/test/labels.npy\n\n"
                "5. RETRY WITH DIFFERENT DATA DIRECTORY:\n"
                "   - Try a different location with more space\n"
                "   - Update config: data.root_dir: /path/to/new/location\n\n"
                "6. CHECK TENSORFLOW DATASETS VERSION:\n"
                "   - Minimum required: tensorflow-datasets>=4.9.0\n"
                "   - Check version: pip show tensorflow-datasets\n"
                "   - Update if needed: pip install --upgrade tensorflow-datasets\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _load_split(self):
        """Load the specified split into memory."""
        split_dir = self.root_dir / self.split

        if not self._check_dataset_exists():
            error_msg = (
                f"Dataset files not found in {split_dir}\n\n"
                "TROUBLESHOOTING STEPS:\n"
                "1. ENABLE AUTOMATIC DOWNLOAD:\n"
                "   - Set download=True when creating PCamDataset\n"
                "   - Example: dataset = PCamDataset(root_dir='./data/pcam', download=True)\n\n"
                "2. VERIFY EXPECTED FILE STRUCTURE:\n"
                f"   {split_dir}/images.npy (or images.h5py)\n"
                f"   {split_dir}/labels.npy (or labels.h5py)\n\n"
                "3. CHECK IF FILES EXIST:\n"
                f"   - List directory: ls -la {split_dir}\n"
                f"   - Verify files are not empty: ls -lh {split_dir}/*.npy\n\n"
                "4. MANUAL DOWNLOAD:\n"
                "   - Download from: https://github.com/basveeling/pcam\n"
                "   - Place files in the structure shown above\n\n"
                "5. CHECK SPLIT NAME:\n"
                f"   - Current split: '{self.split}'\n"
                "   - Valid splits: 'train', 'val', 'test'\n"
                "   - Ensure split directory exists and contains data files\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            # Try loading from .npy files first
            if (split_dir / "images.npy").exists() and (split_dir / "labels.npy").exists():
                # Use memory-mapped mode to avoid loading entire dataset into RAM
                self.images = np.load(split_dir / "images.npy", mmap_mode="r")
                self.labels = np.load(split_dir / "labels.npy", mmap_mode="r")
                logger.info(
                    f"Loaded {len(self.images)} samples from .npy files for {self.split} split (memory-mapped)"
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
            error_type = type(e).__name__
            error_msg = (
                f"Failed to load dataset files: {error_type}: {str(e)}\n\n"
                "CORRUPTED DATASET FILES DETECTED\n\n"
                "TROUBLESHOOTING STEPS:\n"
                "1. VERIFY FILE INTEGRITY:\n"
                f"   - Check file sizes: ls -lh {split_dir}/*.npy\n"
                "   - Images file should be ~1-2GB\n"
                "   - Labels file should be ~100KB-1MB\n\n"
                "2. TEST FILE LOADING:\n"
                "   - Try loading manually:\n"
                "     python -c \"import numpy as np; data = np.load('{split_dir}/images.npy'); print(data.shape)\"\n\n"
                "3. DELETE AND RE-DOWNLOAD:\n"
                f"   - Remove corrupted files: rm -rf {split_dir}\n"
                "   - Re-download: Set download=True in PCamDataset\n\n"
                "4. CHECK DISK SPACE:\n"
                "   - Ensure sufficient space during download\n"
                "   - Required: ~2GB per split\n\n"
                "5. CHECK MEMORY:\n"
                "   - Loading dataset requires ~2-3GB RAM per split\n"
                "   - Close other applications if memory is limited\n\n"
                "6. TRY ALTERNATIVE FORMAT:\n"
                "   - If .npy files fail, try .h5py format\n"
                "   - Install h5py: pip install h5py\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

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
            error_type = type(e).__name__
            logger.warning(
                f"CORRUPTED IMAGE DETECTED at index {idx} in {self.split} split\n"
                f"Error: {error_type}: {str(e)}\n"
                "Returning dummy sample to prevent dataloader crash.\n"
                "This sample will be skipped during training.\n\n"
                "TROUBLESHOOTING:\n"
                "- If many corrupted images are detected, consider re-downloading the dataset\n"
                "- Check dataset integrity with validate_dataset() function\n"
                "- Corrupted images may indicate incomplete download or disk errors"
            )
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
        error_msg = (
            "VALIDATION SET IS EMPTY\n\n"
            "TROUBLESHOOTING STEPS:\n"
            "1. CHECK SPLIT NAME:\n"
            "   - Verify you're using the correct split: 'train', 'val', or 'test'\n"
            "   - Check dataset initialization: PCamDataset(split='val')\n\n"
            "2. VERIFY DATASET FILES:\n"
            "   - Ensure dataset files exist and are not empty\n"
            "   - Check file sizes: ls -lh data/pcam/val/*.npy\n\n"
            "3. RE-DOWNLOAD DATASET:\n"
            "   - Delete existing files: rm -rf data/pcam/val\n"
            "   - Re-download: PCamDataset(download=True)\n\n"
            "4. CHECK DATASET LOADING:\n"
            "   - Verify images and labels were loaded correctly\n"
            "   - Check logs for loading errors during initialization\n"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

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
            error_msg = (
                f"INVALID IMAGE SHAPE DETECTED\n\n"
                f"Sample {i} has shape {sample['image'].shape}, expected (3, 96, 96)\n\n"
                "TROUBLESHOOTING STEPS:\n"
                "1. CHECK DATASET FORMAT:\n"
                "   - PatchCamelyon images should be 96x96 RGB patches\n"
                "   - Verify dataset was downloaded correctly\n\n"
                "2. CHECK TRANSFORMS:\n"
                "   - Ensure transforms don't change image dimensions\n"
                "   - Review transform pipeline in get_pcam_transforms()\n\n"
                "3. RE-DOWNLOAD DATASET:\n"
                "   - Dataset may be corrupted\n"
                "   - Delete and re-download with download=True\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate labels
        if sample["label"].item() not in [0, 1]:
            error_msg = (
                f"INVALID LABEL DETECTED\n\n"
                f"Sample {i} has label {sample['label'].item()}, expected 0 or 1\n\n"
                "TROUBLESHOOTING STEPS:\n"
                "1. CHECK LABEL FORMAT:\n"
                "   - PatchCamelyon uses binary labels: 0 (normal) or 1 (metastatic)\n"
                "   - Verify labels.npy contains only 0s and 1s\n\n"
                "2. CHECK DATASET VERSION:\n"
                "   - Ensure you're using the correct PatchCamelyon dataset\n"
                "   - Some versions may have different label encodings\n\n"
                "3. RE-DOWNLOAD DATASET:\n"
                "   - Labels file may be corrupted\n"
                "   - Delete and re-download with download=True\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    logger.info(f"Dataset validation passed: {len(dataset)} samples")
