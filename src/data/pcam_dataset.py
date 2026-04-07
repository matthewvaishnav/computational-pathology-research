"""
PatchCamelyon (PCam) Dataset Implementation.

This module provides a PyTorch Dataset class for the PatchCamelyon dataset,
a binary classification task for detecting metastatic tissue in histopathology
images.

Dataset originally from: https://github.com/basveeling/pcam
Paper: https://arxiv.org/abs/1803.04140

The dataset returns RAW IMAGES - feature extraction is handled separately
by ResNetFeatureExtractor in the training script to avoid per-sample overhead.
"""

import json
import logging
import os
import shutil
import tarfile
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    import tensorflow_datasets as tfds
    TFDS_AVAILABLE = True
except ImportError:
    TFDS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PCamDataset(Dataset):
    """
    PatchCamelyon (PCam) dataset for metastatic tissue detection.

    This dataset loads histopathology image patches (96x96 RGB) and returns
    them as PyTorch tensors ready for training. The dataset supports optional
    transforms for data augmentation.

    Args:
        root_dir: Root directory containing the dataset files. If the dataset
            is not found, it will be downloaded to this location.
        split: Which split to load - 'train', 'val', or 'test'.
        transform: Optional transform to apply to images. Should accept PIL
            images or tensors and return transformed tensors.
        download: If True, download the dataset if not found in root_dir.

    Attributes:
        images: H5py file handle for image data (loaded lazily).
        labels: H5py file handle for label data (loaded lazily).
        split: Current data split.
        transform: Transform pipeline being applied.

    Example:
        >>> dataset = PCamDataset(
        ...     root_dir='data/pcam',
        ...     split='train',
        ...     transform=transforms.Compose([
        ...         transforms.RandomHorizontalFlip(),
        ...         transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        ...     ])
        ... )
        >>> sample = dataset[0]
        >>> print(sample['image'].shape)  # torch.Size([3, 96, 96])
        >>> print(sample['label'])  # tensor(0) or tensor(1)
    """

    # URLs for direct download from PCam GitHub release
    PCAM_URLS = {
        'train': 'https://github.com/basveeling/pcam/raw/master/pcam_v1/train.tar.gz',
        'val': 'https://github.com/basveeling/pcam/raw/master/pcam_v1/valid.tar.gz',
        'test': 'https://github.com/basveeling/pcam/raw/master/pcam_v1/test.tar.gz',
    }

    SPLIT_SIZES = {
        'train': (262144, 8000),  # (images, samples_for_val)
        'val': (8000, 8000),
        'test': (32768, 32768),
    }

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        """Initialize the PCam dataset.

        Args:
            root_dir: Root directory for dataset storage.
            split: One of 'train', 'val', or 'test'.
            transform: Optional transform to apply to images.
            download: Whether to download if dataset is missing.

        Raises:
            ValueError: If split is not one of 'train', 'val', or 'test'.
            RuntimeError: If dataset is not found and download is False.
        """
        # Initialize cleanup-sensitive attributes early so __del__ is safe
        # even if construction fails before the dataset is fully configured.
        self._images_h5 = None
        self._labels_h5 = None
        self._num_images = 0
        self._num_labels = 0
        self.metadata = {}

        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.transform = transform
        self.download_flag = download

        if self.split not in ('train', 'val', 'test'):
            raise ValueError(
                f"Invalid split '{split}'. Must be one of 'train', 'val', or 'test'."
            )

        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        # Dataset files
        self.split_dir = self.root_dir / self.split
        self.images_file = self.split_dir / 'images.h5py'
        self.labels_file = self.split_dir / 'labels.h5py'
        self.metadata_file = self.root_dir / 'metadata.json'

        # Load or download dataset
        if self._check_exists():
            logger.info(f"Found PCam dataset at {self.root_dir}")
            self._load_metadata()
        elif download:
            logger.info("Dataset not found. Downloading...")
            self.download()
            self._load_metadata()
        else:
            raise RuntimeError(
                f"Dataset not found at {self.root_dir}. "
                f"Set download=True to download it."
            )

        # Open HDF5 files
        self._open_h5_files()

    def _check_exists(self) -> bool:
        """Check if the dataset files exist for the current split."""
        return self.images_file.exists() and self.labels_file.exists()

    def _open_h5_files(self) -> None:
        """Open HDF5 files for lazy access."""
        try:
            self._images_h5 = h5py.File(str(self.images_file), 'r')
            self._labels_h5 = h5py.File(str(self.labels_file), 'r')

            # Get dataset shapes
            self._num_images = self._images_h5['images'].shape[0]
            self._num_labels = self._labels_h5['labels'].shape[0]

            if self._num_images != self._num_labels:
                raise RuntimeError(
                    f"Number of images ({self._num_images}) does not match "
                    f"number of labels ({self._num_labels})"
                )

            logger.debug(
                f"Opened HDF5 files for split '{self.split}': "
                f"{self._num_images} samples"
            )

        except Exception as e:
            self._close_h5_files()
            raise RuntimeError(f"Failed to open HDF5 files: {e}")

    def _close_h5_files(self) -> None:
        """Close HDF5 file handles."""
        images_h5 = getattr(self, '_images_h5', None)
        if images_h5 is not None:
            try:
                images_h5.close()
            except Exception:
                pass
            self._images_h5 = None

        labels_h5 = getattr(self, '_labels_h5', None)
        if labels_h5 is not None:
            try:
                labels_h5.close()
            except Exception:
                pass
            self._labels_h5 = None

    def _load_metadata(self) -> None:
        """Load dataset metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save dataset metadata to JSON file."""
        self.metadata = metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def download(self) -> None:
        """
        Download the PCam dataset.

        First attempts to use TensorFlow Datasets. If TFDS is not available,
        falls back to direct download from the PCam GitHub release.

        The dataset is extracted and organized into HDF5 files for efficient
        random access during training.

        Raises:
            RuntimeError: If download fails.
        """
        self.root_dir.mkdir(parents=True, exist_ok=True)

        if TFDS_AVAILABLE:
            self._download_via_tfds()
        else:
            logger.warning(
                "TensorFlow Datasets not available. "
                "Falling back to direct download from GitHub."
            )
            self._download_direct()

        # Convert extracted data to HDF5 format
        self._process_downloaded_data()

        # Save metadata
        metadata = {
            'dataset': 'PCam',
            'version': '1.0',
            'splits': {
                split: {
                    'num_samples': self._get_split_size(split),
                }
                for split in ('train', 'val', 'test')
            },
            'image_shape': [96, 96, 3],
            'num_classes': 2,
            'class_names': ['normal', 'metastatic'],
        }
        self._save_metadata(metadata)

        logger.info("PCam dataset download complete!")

    def _download_via_tfds(self) -> None:
        """Download PCam using TensorFlow Datasets."""
        logger.info("Downloading PCam via TensorFlow Datasets...")

        try:
            # Download and process each split
            for split_name, tfds_split in [
                ('train', 'train'),
                ('val', 'validation'),
                ('test', 'test')
            ]:
                logger.info(f"Processing split: {split_name}")

                # Get dataset from TFDS
                dataset = tfds.load(
                    'pcam',
                    split=tfds_split,
                    data_dir=str(self.root_dir),
                    download=True,
                )

                # Convert TFDS dataset to numpy arrays
                images_list = []
                labels_list = []

                for example in tfds.as_numpy(dataset):
                    images_list.append(example['image'])
                    labels_list.append(example['label'])

                images = np.array(images_list, dtype=np.uint8)
                labels = np.array(labels_list, dtype=np.int32)

                # Save as HDF5
                split_dir = self.root_dir / split_name
                split_dir.mkdir(parents=True, exist_ok=True)

                with h5py.File(split_dir / 'images.h5py', 'w') as f:
                    f.create_dataset('images', data=images, compression='gzip')

                with h5py.File(split_dir / 'labels.h5py', 'w') as f:
                    f.create_dataset('labels', data=labels, compression='gzip')

                logger.info(
                    f"Saved {split_name} split: {len(images)} samples"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to download via TFDS: {e}")

    def _download_direct(self) -> None:
        """Download PCam directly from GitHub release."""
        import urllib.request

        for split_name, url in self.PCAM_URLS.items():
            logger.info(f"Downloading {split_name} split from {url}")

            try:
                # Create temp directory for extraction
                temp_dir = self.root_dir / 'temp'
                temp_dir.mkdir(parents=True, exist_ok=True)

                # Download tarball
                tarball_path = temp_dir / f'{split_name}.tar.gz'
                urllib.request.urlretrieve(url, str(tarball_path))

                # Extract tarball
                logger.info(f"Extracting {split_name} split...")
                with tarfile.open(str(tarball_path), 'r:gz') as tar:
                    tar.extractall(str(temp_dir))

                # Clean up tarball
                tarball_path.unlink()

                # Move extracted files to final location
                extracted_dir = temp_dir / split_name
                if extracted_dir.exists():
                    dest_dir = self.root_dir / split_name
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.move(str(extracted_dir), str(dest_dir))

                # Clean up temp
                shutil.rmtree(temp_dir)

                logger.info(f"Downloaded and extracted {split_name} split")

            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {split_name} split: {e}"
                )

    def _process_downloaded_data(self) -> None:
        """
        Process downloaded/extracted data into HDF5 format.

        Converts the downloaded data (in various formats) into standardized
        HDF5 files for efficient random access.
        """
        # Process each split directory
        for split_name in ('train', 'val', 'test'):
            split_dir = self.root_dir / split_name

            # Check if already processed
            if (split_dir / 'images.h5py').exists():
                continue

            # Find image files
            image_files = self._find_image_files(split_dir)

            if not image_files:
                logger.warning(f"No image files found in {split_dir}")
                continue

            # Load images and labels
            images = []
            labels = []

            for img_path in image_files:
                try:
                    # Load image
                    img = Image.open(img_path)
                    img_array = np.array(img)

                    # Extract label from filename
                    # PCam naming convention: ID_label.tif (label is 0 or 1)
                    # or center_XXX.tif for test set
                    filename = img_path.stem

                    if '_' in filename:
                        # Training/validation: filename is like "kidney_1010_0_128_128_48"
                        # or "tumor_1010_0_128_128_48"
                        label_str = filename.split('_')[-1]
                        # Check if last element is a valid label
                        if label_str in ('0', '1'):
                            label = int(label_str)
                        else:
                            # Check for 'tumor' prefix
                            label = 1 if 'tumor' in filename else 0
                    else:
                        # Fallback: assume test set has no labels
                        label = 0

                    images.append(img_array)
                    labels.append(label)

                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")
                    continue

            if images:
                # Convert to numpy arrays
                images = np.array(images, dtype=np.uint8)
                labels = np.array(labels, dtype=np.int32)

                # Save as HDF5
                with h5py.File(split_dir / 'images.h5py', 'w') as f:
                    f.create_dataset('images', data=images, compression='gzip')

                with h5py.File(split_dir / 'labels.h5py', 'w') as f:
                    f.create_dataset('labels', data=labels, compression='gzip')

                logger.info(
                    f"Processed {split_name}: {len(images)} images saved"
                )

    def _find_image_files(self, directory: Path) -> List[Path]:
        """Find all image files in a directory."""
        image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        image_files = []

        for ext in image_extensions:
            image_files.extend(directory.glob(f'**/*{ext}'))

        return sorted(image_files)

    def _get_split_size(self, split: str) -> int:
        """Get the expected size of a split."""
        return self.SPLIT_SIZES.get(split, (0, 0))[0]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._num_images

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
                - 'image': PyTorch tensor of shape [3, 96, 96]
                - 'label': PyTorch tensor (scalar, 0 or 1)
                - 'image_id': String identifier for the image

        Raises:
            IndexError: If idx is out of bounds.
            RuntimeError: If there is an error loading the image.
        """
        if idx < 0 or idx >= self._num_images:
            raise IndexError(
                f"Index {idx} out of bounds for dataset with {self._num_images} samples"
            )

        try:
            # Load image and label from HDF5
            image = self._images_h5['images'][idx]
            label = self._labels_h5['labels'][idx]

            # Convert numpy arrays to proper types
            if isinstance(image, np.ndarray):
                # Ensure correct dtype and shape
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)

                # Handle grayscale images (expand to 3 channels)
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.shape[-1] != 3:
                    raise ValueError(
                        f"Expected 3-channel image, got shape {image.shape}"
                    )

                # Convert HWC to CHW format for PyTorch
                image = np.transpose(image, (2, 0, 1))

                # Convert to PIL Image for transform compatibility
                # Clamp to valid range first
                image = np.clip(image, 0, 255).astype(np.uint8)
                image_pil = Image.fromarray(
                    np.transpose(image, (1, 2, 0)),  # CHW -> HWC for PIL
                    mode='RGB'
                )

                # Apply transforms
                if self.transform is not None:
                    image = self.transform(image_pil)
                else:
                    # Default: convert to tensor and normalize to [0, 1]
                    image = torch.from_numpy(image).float() / 255.0

            # Ensure label is proper type
            label = int(label) if isinstance(label, (np.integer, np.int32, np.int64)) else label
            label_tensor = torch.tensor(label, dtype=torch.long)

            # Create image_id
            image_id = f"{self.split}_{idx:08d}"

            return {
                'image': image,
                'label': label_tensor,
                'image_id': image_id,
            }

        except IndexError as e:
            raise IndexError(
                f"Error accessing index {idx}: {e}"
            )
        except Exception as e:
            # Log warning and return a placeholder for corrupted images
            warnings.warn(
                f"Error loading sample {idx} from {self.split} split: {e}. "
                f"Returning zeros as placeholder.",
                RuntimeWarning
            )

            # Return a placeholder sample
            return {
                'image': torch.zeros(3, 96, 96, dtype=torch.float32),
                'label': torch.tensor(0, dtype=torch.long),
                'image_id': f"{self.split}_{idx:08d}_error",
            }

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return (
            f"PCamDataset(\n"
            f"    root_dir='{self.root_dir}',\n"
            f"    split='{self.split}',\n"
            f"    num_samples={self._num_images},\n"
            f"    transform={self.transform},\n"
            f")"
        )

    def __del__(self) -> None:
        """Clean up HDF5 file handles when dataset is destroyed."""
        try:
            self._close_h5_files()
        except Exception:
            pass


class PCamDatasetWithFeatures(Dataset):
    """
    Extended PCam dataset that also provides pre-extracted features.

    This class wraps PCamDataset and adds pre-extracted ResNet features
    for scenarios where feature extraction should be done once upfront
    rather than during training.

    Note: Most use cases should use PCamDataset directly and let the
    training script handle feature extraction to avoid duplication.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        features_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        """
        Initialize the dataset with optional pre-extracted features.

        Args:
            root_dir: Root directory containing the dataset files.
            split: Which split to load.
            features_path: Optional path to pre-extracted features file.
            transform: Optional transform to apply to images.
            download: Whether to download if dataset is missing.
        """
        self.base_dataset = PCamDataset(
            root_dir=root_dir,
            split=split,
            transform=transform,
            download=download,
        )

        self.features_path = features_path
        self._features_h5 = None

        if features_path is not None:
            self._load_features()

    def _load_features(self) -> None:
        """Load pre-extracted features from HDF5 file."""
        if self.features_path and Path(self.features_path).exists():
            self._features_h5 = h5py.File(self.features_path, 'r')
            logger.info(f"Loaded features from {self.features_path}")
        else:
            logger.warning(
                f"Features file not found at {self.features_path}. "
                f"Set will return raw images only."
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample with both image and features (if available).

        Returns:
            Dictionary containing:
                - 'image': PyTorch tensor [3, 96, 96] (raw image)
                - 'wsi_features': PyTorch tensor (pre-extracted features)
                - 'label': PyTorch tensor
                - 'image_id': str
        """
        sample = self.base_dataset[idx]

        if self._features_h5 is not None:
            try:
                features = self._features_h5['features'][idx]
                sample['wsi_features'] = torch.from_numpy(features).float()
            except Exception as e:
                logger.warning(
                    f"Failed to load features for index {idx}: {e}"
                )
                sample['wsi_features'] = None

        return sample

    def __del__(self) -> None:
        """Clean up resources."""
        features_h5 = getattr(self, '_features_h5', None)
        if features_h5 is not None:
            try:
                features_h5.close()
            except Exception:
                pass


def get_pcam_transforms(
    split: str = 'train',
    augmentation: bool = True,
) -> Optional[Callable]:
    """
    Get standard transforms for PCam dataset.

    Args:
        split: 'train', 'val', or 'test'.
        augmentation: If True, apply data augmentation for training.

    Returns:
        Composed transform pipeline.
    """
    if split == 'train' and augmentation:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        # Validation/test: minimal preprocessing
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


if __name__ == '__main__':
    # Simple test
    logging.basicConfig(level=logging.INFO)

    # Try to load dataset (will download if not present)
    dataset = PCamDataset(
        root_dir='data/pcam',
        split='train',
        download=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset repr:\n{dataset}")

    # Get a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label: {sample['label']}")
    print(f"Image ID: {sample['image_id']}")
