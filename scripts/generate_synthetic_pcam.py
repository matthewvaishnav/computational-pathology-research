"""
Generate synthetic PatchCamelyon-like data for testing.

Creates small synthetic datasets that match PCam structure for quick testing.
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_pcam(
    root_dir: str,
    train_size: int = 1000,
    val_size: int = 200,
    test_size: int = 200,
    image_size: int = 96,
):
    """
    Generate synthetic PCam dataset.

    Creates random images with subtle patterns to simulate histopathology patches.
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic PCam dataset at {root}")

    splits = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }

    for split, size in splits.items():
        logger.info(f"Generating {split} split: {size} samples")

        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Generate synthetic images with some structure
        # Class 0: smoother, more uniform textures
        # Class 1: more variation, edge-like patterns
        images = []
        labels = []

        for i in range(size):
            label = np.random.randint(0, 2)

            if label == 0:
                # Normal tissue - smoother texture
                base = np.random.normal(128, 20, (image_size, image_size, 3))
            else:
                # Metastatic - more structured patterns
                base = np.random.normal(150, 40, (image_size, image_size, 3))
                # Add some edge-like structures
                x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
                pattern = np.sin(x / 8) * np.cos(y / 8) * 30
                base[:, :, 0] += pattern

            # Clip to valid range
            img = np.clip(base, 0, 255).astype(np.uint8)

            images.append(img)
            labels.append(label)

        images = np.array(images, dtype=np.uint8)
        labels = np.array(labels, dtype=np.int32)

        # Save as HDF5
        with h5py.File(split_dir / "images.h5py", "w") as f:
            f.create_dataset("images", data=images, compression="gzip")

        with h5py.File(split_dir / "labels.h5py", "w") as f:
            f.create_dataset("labels", data=labels, compression="gzip")

        # Class distribution
        class_0 = np.sum(labels == 0)
        class_1 = np.sum(labels == 1)
        logger.info(f"  Class distribution: {class_0} normal, {class_1} metastatic")

    # Save metadata
    metadata = {
        "dataset": "synthetic_pcam",
        "version": "test",
        "splits": {
            "train": {"num_samples": train_size},
            "val": {"num_samples": val_size},
            "test": {"num_samples": test_size},
        },
        "image_shape": [image_size, image_size, 3],
        "num_classes": 2,
        "class_names": ["normal", "metastatic"],
        "synthetic": True,
    }

    with open(root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Synthetic dataset created at {root}")
    logger.info(f"  Total samples: {train_size + val_size + test_size}")
    logger.info(f"  Image size: {image_size}x{image_size}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./data/pcam")
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=96)
    args = parser.parse_args()

    generate_synthetic_pcam(
        root_dir=args.root_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
