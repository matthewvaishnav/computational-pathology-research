"""
Utility functions for testing.

This module provides helper functions for creating synthetic test data
and other testing utilities.
"""

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def create_synthetic_camelyon_data(
    data_dir: Path,
    num_slides: int = 10,
    num_patches: int = 50,
    feature_dim: int = 512,
    seed: int = 42,
) -> None:
    """Create synthetic CAMELYON data for testing attention models.

    This function generates:
    - slide_index.json with train/val/test splits (70/15/15)
    - HDF5 feature files for each slide with random features and coordinates
    - Alternating labels (0=normal, 1=tumor)

    Args:
        data_dir: Directory to create synthetic data in
        num_slides: Total number of slides to generate
        num_patches: Number of patches per slide
        feature_dim: Dimension of feature vectors
        seed: Random seed for reproducibility

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     data_dir = Path(tmp_dir) / "camelyon"
        ...     create_synthetic_camelyon_data(data_dir, num_slides=10)
        ...     assert (data_dir / "slide_index.json").exists()
        ...     assert len(list((data_dir / "features").glob("*.h5"))) == 10
    """
    data_dir = Path(data_dir)
    features_dir = data_dir / "features"

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    # Calculate split sizes (70/15/15)
    num_train = int(num_slides * 0.7)
    num_val = int(num_slides * 0.15)
    num_test = num_slides - num_train - num_val

    # Generate slide metadata
    slides = []
    slide_idx = 0

    for split, count in [("train", num_train), ("val", num_val), ("test", num_test)]:
        for i in range(count):
            # Alternate labels between normal (0) and tumor (1)
            label = slide_idx % 2

            slides.append(
                {
                    "slide_id": f"slide_{slide_idx:03d}",
                    "patient_id": f"patient_{slide_idx // 2:03d}",
                    "file_path": f"./slides/slide_{slide_idx:03d}.tif",
                    "label": label,
                    "split": split,
                    "annotation_path": None,
                    "width": 50000,
                    "height": 50000,
                    "magnification": 40.0,
                    "mpp": 0.25,
                }
            )
            slide_idx += 1

    # Save slide index
    index_data = {
        "dataset": "CAMELYON",
        "num_slides": len(slides),
        "slides": slides,
    }

    index_path = data_dir / "slide_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    # Generate features and coordinates for each slide
    rng = np.random.RandomState(seed)

    for slide in slides:
        slide_id = slide["slide_id"]
        label = slide["label"]

        # Generate features with slight class separation for realism
        if label == 0:
            # Normal slides: features centered around 0
            features = rng.randn(num_patches, feature_dim).astype(np.float32) * 0.5
        else:
            # Tumor slides: features shifted slightly positive
            features = (rng.randn(num_patches, feature_dim) * 0.5 + 0.3).astype(np.float32)

        # Generate random patch coordinates
        coordinates = rng.randint(0, 10000, size=(num_patches, 2)).astype(np.int32)

        # Save to HDF5
        h5_path = features_dir / f"{slide_id}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("features", data=features, compression="gzip")
            f.create_dataset("coordinates", data=coordinates, compression="gzip")
