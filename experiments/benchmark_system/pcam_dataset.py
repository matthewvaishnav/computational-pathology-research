"""
PCam Dataset Loader for Benchmark System.

Loads real PatchCamelyon (PCam) data from H5 files for benchmarking.
Supports subset loading for medium/quick benchmarks.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PCamDataset(Dataset):
    """
    PCam dataset loader from H5 files.
    
    Loads images and labels from PatchCamelyon H5 files.
    Supports subset loading for faster benchmarks.
    
    Args:
        h5_path_x: Path to H5 file with images (e.g., camelyonpatch_level_2_split_train_x.h5)
        h5_path_y: Path to H5 file with labels (e.g., camelyonpatch_level_2_split_train_y.h5)
        max_samples: Maximum samples to load (None = all). For medium benchmark: 10k-50k
        transform: Optional transform to apply to images
    """
    
    def __init__(
        self,
        h5_path_x: Path,
        h5_path_y: Path,
        max_samples: Optional[int] = None,
        transform=None,
    ):
        self.h5_path_x = Path(h5_path_x)
        self.h5_path_y = Path(h5_path_y)
        self.max_samples = max_samples
        self.transform = transform
        
        # Load data into memory (faster than reading from H5 each time)
        logger.info(f"Loading PCam data from {self.h5_path_x.name}")
        
        with h5py.File(self.h5_path_x, 'r') as f:
            # PCam H5 structure: f['x'] contains images [N, 96, 96, 3]
            if max_samples is not None:
                self.images = f['x'][:max_samples]
            else:
                self.images = f['x'][:]
        
        with h5py.File(self.h5_path_y, 'r') as f:
            # PCam H5 structure: f['y'] contains labels [N, 1, 1, 1]
            if max_samples is not None:
                self.labels = f['y'][:max_samples]
            else:
                self.labels = f['y'][:]
        
        # Reshape labels from [N, 1, 1, 1] to [N]
        self.labels = self.labels.squeeze()
        
        logger.info(f"Loaded {len(self.images)} samples from PCam dataset")
        logger.info(f"Image shape: {self.images.shape}, Label shape: {self.labels.shape}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and label at index.
        
        Returns:
            Tuple of (image_tensor, label_tensor)
            - image_tensor: [3, 96, 96] float32 (normalized if transform applied)
            - label_tensor: scalar int64
        """
        # Get image [96, 96, 3] uint8
        image = self.images[idx]
        
        # Get label (scalar)
        label = self.labels[idx]
        
        # Apply transform if provided (expects uint8 numpy array)
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Convert image to float and normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            # Image: [H, W, C] -> [C, H, W]
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label


def create_pcam_loaders(
    data_root: Path,
    batch_size: int,
    max_samples_train: Optional[int] = None,
    max_samples_val: Optional[int] = None,
    max_samples_test: Optional[int] = None,
    num_workers: int = 0,
    train_transform=None,
    val_test_transform=None,
) -> Tuple:
    """
    Create PCam data loaders for train/val/test.
    
    Args:
        data_root: Root directory containing PCam H5 files (e.g., data/pcam_real/)
        batch_size: Batch size for data loaders
        max_samples_train: Max training samples (None = all 262144)
        max_samples_val: Max validation samples (None = all 32768)
        max_samples_test: Max test samples (None = all 32768)
        num_workers: Number of data loader workers
        train_transform: Transform for training data
        val_test_transform: Transform for val/test data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_root = Path(data_root)
    
    # Create datasets
    train_dataset = PCamDataset(
        h5_path_x=data_root / "camelyonpatch_level_2_split_train_x.h5",
        h5_path_y=data_root / "camelyonpatch_level_2_split_train_y.h5",
        max_samples=max_samples_train,
        transform=train_transform,
    )
    
    val_dataset = PCamDataset(
        h5_path_x=data_root / "camelyonpatch_level_2_split_valid_x.h5",
        h5_path_y=data_root / "camelyonpatch_level_2_split_valid_y.h5",
        max_samples=max_samples_val,
        transform=val_test_transform,
    )
    
    test_dataset = PCamDataset(
        h5_path_x=data_root / "camelyonpatch_level_2_split_test_x.h5",
        h5_path_y=data_root / "camelyonpatch_level_2_split_test_y.h5",
        max_samples=max_samples_test,
        transform=val_test_transform,
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    logger.info(
        f"Created PCam loaders - Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    
    return train_loader, val_loader, test_loader
