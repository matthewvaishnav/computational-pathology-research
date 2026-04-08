"""
Generate synthetic CAMELYON16 data for testing the training pipeline.

This script creates:
- Slide index JSON with train/val/test splits
- Pre-extracted HDF5 patch features for each slide
- Minimal synthetic data to verify the training path works

NOTE: This is synthetic data for testing only, not real CAMELYON16 data.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_slide_index(
    num_train: int = 20,
    num_val: int = 5,
    num_test: int = 5,
) -> List[Dict]:
    """Generate synthetic slide metadata.

    Args:
        num_train: Number of training slides
        num_val: Number of validation slides
        num_test: Number of test slides

    Returns:
        List of slide metadata dictionaries
    """
    slides = []
    slide_id = 0

    for split, num_slides in [("train", num_train), ("val", num_val), ("test", num_test)]:
        for i in range(num_slides):
            # Alternate between normal (0) and tumor (1)
            label = i % 2

            slides.append(
                {
                    "slide_id": f"slide_{slide_id:03d}",
                    "patient_id": f"patient_{slide_id // 2:03d}",
                    "file_path": f"./data/camelyon/slides/slide_{slide_id:03d}.tif",
                    "label": label,
                    "split": split,
                    "annotation_path": None,
                    "width": 100000,
                    "height": 100000,
                    "magnification": 40.0,
                    "mpp": 0.25,
                }
            )
            slide_id += 1

    return slides


def generate_patch_features(
    slide_id: str,
    label: int,
    num_patches: int = 100,
    feature_dim: int = 2048,
    seed: int = 42,
) -> tuple:
    """Generate synthetic patch features for one slide.

    Args:
        slide_id: Slide identifier
        label: Slide-level label (0=normal, 1=tumor)
        num_patches: Number of patches per slide
        feature_dim: Feature dimension
        seed: Random seed

    Returns:
        Tuple of (features, coordinates)
    """
    # Use slide_id hash for reproducibility
    rng = np.random.RandomState(seed + hash(slide_id) % 10000)

    # Generate features with slight class separation
    if label == 0:
        # Normal slides: features centered around 0
        features = rng.randn(num_patches, feature_dim) * 0.5
    else:
        # Tumor slides: features shifted slightly positive
        features = rng.randn(num_patches, feature_dim) * 0.5 + 0.3

    # Generate random coordinates
    coordinates = rng.randint(0, 10000, size=(num_patches, 2))

    return features.astype(np.float32), coordinates.astype(np.int32)


def save_slide_features(
    output_dir: Path,
    slide_id: str,
    features: np.ndarray,
    coordinates: np.ndarray,
) -> None:
    """Save patch features to HDF5 file.

    Args:
        output_dir: Output directory for HDF5 files
        slide_id: Slide identifier
        features: Patch features [num_patches, feature_dim]
        coordinates: Patch coordinates [num_patches, 2]
    """
    output_path = output_dir / f"{slide_id}.h5"

    with h5py.File(output_path, "w") as f:
        f.create_dataset("features", data=features, compression="gzip")
        f.create_dataset("coordinates", data=coordinates, compression="gzip")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CAMELYON16 data for testing")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/camelyon",
        help="Output directory for synthetic data",
    )
    parser.add_argument("--num-train", type=int, default=20, help="Number of training slides")
    parser.add_argument("--num-val", type=int, default=5, help="Number of validation slides")
    parser.add_argument("--num-test", type=int, default=5, help="Number of test slides")
    parser.add_argument("--num-patches", type=int, default=100, help="Number of patches per slide")
    parser.add_argument(
        "--feature-dim", type=int, default=2048, help="Feature dimension (ResNet-50 default)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    features_dir = output_dir / "features"

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Generating Synthetic CAMELYON16 Data")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Train slides: {args.num_train}")
    logger.info(f"Val slides: {args.num_val}")
    logger.info(f"Test slides: {args.num_test}")
    logger.info(f"Patches per slide: {args.num_patches}")
    logger.info(f"Feature dimension: {args.feature_dim}")
    logger.info("")

    # Generate slide index
    logger.info("Generating slide index...")
    slides = generate_slide_index(
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
    )

    # Save slide index
    index_path = output_dir / "slide_index.json"
    index_data = {
        "dataset": "CAMELYON",
        "num_slides": len(slides),
        "slides": slides,
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    logger.info(f"Saved slide index to {index_path}")
    logger.info(f"Total slides: {len(slides)}")
    logger.info("")

    # Generate patch features for each slide
    logger.info("Generating patch features...")
    for slide in slides:
        slide_id = slide["slide_id"]
        label = slide["label"]

        features, coordinates = generate_patch_features(
            slide_id=slide_id,
            label=label,
            num_patches=args.num_patches,
            feature_dim=args.feature_dim,
            seed=args.seed,
        )

        save_slide_features(
            output_dir=features_dir,
            slide_id=slide_id,
            features=features,
            coordinates=coordinates,
        )

        if (slides.index(slide) + 1) % 10 == 0:
            logger.info(f"  Generated {slides.index(slide) + 1}/{len(slides)} slides")

    logger.info(f"Generated {len(slides)} slides")
    logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("Generation Complete")
    logger.info("=" * 80)
    logger.info(f"Slide index: {index_path}")
    logger.info(f"Features directory: {features_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Train model:")
    logger.info(
        "     python experiments/train_camelyon.py --config experiments/configs/camelyon.yaml"
    )
    logger.info("")
    logger.info("⚠️  NOTE: This is synthetic data for testing only!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
