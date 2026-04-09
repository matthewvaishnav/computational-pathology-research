"""
Convert PCam Zenodo h5 files to the format expected by PCamDataset.

The Zenodo files are named:
- camelyonpatch_level_2_split_train_x.h5
- camelyonpatch_level_2_split_train_y.h5
- camelyonpatch_level_2_split_valid_x.h5
- camelyonpatch_level_2_split_valid_y.h5
- camelyonpatch_level_2_split_test_x.h5
- camelyonpatch_level_2_split_test_y.h5

The dataset expects:
- data/pcam_real/train/images.h5py
- data/pcam_real/train/labels.h5py
- data/pcam_real/val/images.h5py
- data/pcam_real/val/labels.h5py
- data/pcam_real/test/images.h5py
- data/pcam_real/test/labels.h5py
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_split(source_dir: Path, split_name: str, zenodo_split_name: str):
    """Convert one split from Zenodo format to dataset format."""
    logger.info(f"Converting {split_name} split...")
    
    # Source files
    images_source = source_dir / f"camelyonpatch_level_2_split_{zenodo_split_name}_x.h5"
    labels_source = source_dir / f"camelyonpatch_level_2_split_{zenodo_split_name}_y.h5"
    
    if not images_source.exists():
        raise FileNotFoundError(f"Source images file not found: {images_source}")
    if not labels_source.exists():
        raise FileNotFoundError(f"Source labels file not found: {labels_source}")
    
    # Destination directory
    dest_dir = source_dir / split_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Destination files
    images_dest = dest_dir / "images.h5py"
    labels_dest = dest_dir / "labels.h5py"
    
    # Load and convert images
    logger.info(f"  Loading images from {images_source.name}...")
    with h5py.File(images_source, "r") as f:
        # Zenodo files have data in 'x' or 'y' dataset
        if "x" in f:
            images = f["x"][:]
        else:
            # Try to find the dataset
            keys = list(f.keys())
            logger.info(f"  Available keys in images file: {keys}")
            images = f[keys[0]][:]
    
    logger.info(f"  Images shape: {images.shape}, dtype: {images.dtype}")
    
    # Load and convert labels
    logger.info(f"  Loading labels from {labels_source.name}...")
    with h5py.File(labels_source, "r") as f:
        if "y" in f:
            labels = f["y"][:]
        else:
            keys = list(f.keys())
            logger.info(f"  Available keys in labels file: {keys}")
            labels = f[keys[0]][:]
    
    # Flatten labels if needed (they might be shape (N, 1, 1, 1))
    labels = labels.flatten()
    
    logger.info(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}")
    logger.info(f"  Label distribution: {np.bincount(labels)}")
    
    # Save in expected format
    logger.info(f"  Saving to {images_dest}...")
    with h5py.File(images_dest, "w") as f:
        f.create_dataset("images", data=images, compression="gzip", compression_opts=4)
    
    logger.info(f"  Saving to {labels_dest}...")
    with h5py.File(labels_dest, "w") as f:
        f.create_dataset("labels", data=labels, compression="gzip", compression_opts=4)
    
    logger.info(f"  ✓ Converted {split_name} split: {len(images)} samples")
    
    return len(images)


def main():
    """Convert all splits."""
    source_dir = Path("data/pcam_real")
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    logger.info("=" * 60)
    logger.info("Converting PCam Zenodo files to dataset format")
    logger.info("=" * 60)
    
    # Convert each split
    splits = [
        ("train", "train"),
        ("val", "valid"),
        ("test", "test"),
    ]
    
    split_sizes = {}
    for split_name, zenodo_name in splits:
        num_samples = convert_split(source_dir, split_name, zenodo_name)
        split_sizes[split_name] = num_samples
    
    # Create metadata file
    metadata = {
        "dataset": "PCam",
        "version": "1.0",
        "source": "Zenodo",
        "splits": {
            split: {"num_samples": size}
            for split, size in split_sizes.items()
        },
        "image_shape": [96, 96, 3],
        "num_classes": 2,
        "class_names": ["normal", "metastatic"],
    }
    
    metadata_file = source_dir / "metadata.json"
    logger.info(f"\nSaving metadata to {metadata_file}...")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("Conversion complete!")
    logger.info("=" * 60)
    logger.info(f"Train samples: {split_sizes['train']}")
    logger.info(f"Val samples: {split_sizes['val']}")
    logger.info(f"Test samples: {split_sizes['test']}")


if __name__ == "__main__":
    main()
