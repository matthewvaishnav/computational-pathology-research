"""
Script to prepare CAMELYON dataset index and metadata.

This script creates a slide-level index from CAMELYON16/17 WSI files,
optionally incorporating annotation data and official train/test splits.

Example usage:
    # Create index from slide directory with automatic splitting
    python scripts/data/prepare_camelyon_index.py \
        --slide_dir data/camelyon16/slides \
        --output data/camelyon16/slide_index.json

    # Create index with annotations and official splits
    python scripts/data/prepare_camelyon_index.py \
        --slide_dir data/camelyon16/slides \
        --annotation_dir data/camelyon16/annotations \
        --split_file data/camelyon16/splits.csv \
        --output data/camelyon16/slide_index.json
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.camelyon_dataset import CAMELYONSlideIndex, SlideMetadata

logger = logging.getLogger(__name__)


def load_official_splits(split_file: str) -> Dict[str, str]:
    """Load official train/test splits from CSV.

    Expected CSV format:
        slide_id,split
        patient_001_node_0,train
        patient_001_node_1,train
        patient_002_node_0,test
        ...

    Args:
        split_file: Path to CSV with slide_id and split columns

    Returns:
        Dictionary mapping slide_id to split name
    """
    splits = {}
    with open(split_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            splits[row["slide_id"]] = row["split"]
    return splits


def load_ground_truth_labels(label_file: str) -> Dict[str, int]:
    """Load slide-level labels from ground truth file.

    CAMELYON16 provides CSV files with metastasis labels.

    Args:
        label_file: Path to CSV with slide labels

    Returns:
        Dictionary mapping slide_id to label (0=normal, 1=metastasis)
    """
    labels = {}
    with open(label_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try common column names
            slide_id = row.get("slide_id") or row.get("image_name") or row.get("filename")
            label = row.get("label") or row.get("tumor") or row.get("metastasis")

            if slide_id and label is not None:
                # Convert to int if string
                if isinstance(label, str):
                    label = 1 if label.lower() in ["1", "true", "yes", "tumor", "metastasis"] else 0
                labels[slide_id] = int(label)

    return labels


def create_camelyon_index(
    slide_dir: str,
    output_path: str,
    annotation_dir: Optional[str] = None,
    split_file: Optional[str] = None,
    label_file: Optional[str] = None,
    slide_pattern: str = "*.tif",
    seed: int = 42,
) -> CAMELYONSlideIndex:
    """Create CAMELYON slide index with optional metadata.

    Args:
        slide_dir: Directory containing WSI files
        output_path: Where to save the index JSON
        annotation_dir: Optional directory with annotation XML files
        split_file: Optional CSV with official train/test splits
        label_file: Optional CSV with ground truth labels
        slide_pattern: Glob pattern for slide files
        seed: Random seed for splitting (if no split_file)

    Returns:
        CAMELYONSlideIndex with all metadata
    """
    slide_dir = Path(slide_dir)
    annotation_dir = Path(annotation_dir) if annotation_dir else None

    # Load external metadata if provided
    splits = load_official_splits(split_file) if split_file else None
    labels = load_ground_truth_labels(label_file) if label_file else None

    # Discover slides
    slide_files = sorted(slide_dir.glob(slide_pattern))
    logger.info(f"Found {len(slide_files)} slides in {slide_dir}")

    if len(slide_files) == 0:
        logger.error(f"No slides found matching pattern '{slide_pattern}' in {slide_dir}")
        raise ValueError(f"No slides found in {slide_dir}")

    # Create metadata for each slide
    slides = []

    for slide_path in slide_files:
        slide_id = slide_path.stem

        # Determine split
        if splits and slide_id in splits:
            split = splits[slide_id]
        elif splits:
            # If split file provided but slide not in it, skip
            logger.warning(f"Slide {slide_id} not in split file, skipping")
            continue
        else:
            # Random split
            np.random.seed(seed)
            split = np.random.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15])

        # Determine label
        if labels and slide_id in labels:
            label = labels[slide_id]
        else:
            label = -1  # Unknown

        # Look for annotation file
        annotation_path = None
        if annotation_dir:
            # Try common annotation formats
            for ext in [".xml", ".json", ".npy", ".png", ".tif"]:
                ann_file = annotation_dir / f"{slide_id}{ext}"
                if ann_file.exists():
                    annotation_path = str(ann_file)
                    break

        # Extract patient ID from filename
        # CAMELYON format: "patient_XXX_node_Y.tif"
        if "_node_" in slide_id:
            patient_id = slide_id.split("_node_")[0]
        else:
            patient_id = slide_id

        slides.append(
            SlideMetadata(
                slide_id=slide_id,
                patient_id=patient_id,
                file_path=str(slide_path),
                label=label,
                split=split,
                annotation_path=annotation_path,
            )
        )

    # Create and save index
    index = CAMELYONSlideIndex(slides)
    index.save(output_path)

    # Print summary
    train_count = len(index.get_slides_by_split("train"))
    val_count = len(index.get_slides_by_split("val"))
    test_count = len(index.get_slides_by_split("test"))
    annotated_count = len(index.get_annotated_slides())
    labeled_count = len([s for s in slides if s.label >= 0])

    logger.info(f"Created index with {len(slides)} slides:")
    logger.info(f"  Train: {train_count}")
    logger.info(f"  Val: {val_count}")
    logger.info(f"  Test: {test_count}")
    logger.info(f"  With annotations: {annotated_count}")
    logger.info(f"  With labels: {labeled_count}")

    return index


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CAMELYON slide index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic index from slide directory
  python scripts/data/prepare_camelyon_index.py \\
      --slide_dir data/camelyon16/slides \\
      --output data/camelyon16/slide_index.json

  # With annotations and official splits
  python scripts/data/prepare_camelyon_index.py \\
      --slide_dir data/camelyon16/slides \\
      --annotation_dir data/camelyon16/annotations \\
      --split_file data/camelyon16/splits.csv \\
      --label_file data/camelyon16/labels.csv \\
      --output data/camelyon16/slide_index.json
        """,
    )

    parser.add_argument(
        "--slide_dir", type=str, required=True, help="Directory containing WSI slide files"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for slide index JSON"
    )
    parser.add_argument(
        "--annotation_dir", type=str, default=None, help="Directory containing annotation files"
    )
    parser.add_argument(
        "--split_file", type=str, default=None, help="CSV file with official train/test splits"
    )
    parser.add_argument(
        "--label_file", type=str, default=None, help="CSV file with ground truth labels"
    )
    parser.add_argument(
        "--slide_pattern",
        type=str,
        default="*.tif",
        help="Glob pattern for slide files (default: *.tif)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting (default: 42)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        index = create_camelyon_index(
            slide_dir=args.slide_dir,
            output_path=args.output,
            annotation_dir=args.annotation_dir,
            split_file=args.split_file,
            label_file=args.label_file,
            slide_pattern=args.slide_pattern,
            seed=args.seed,
        )
        print(f"\n✓ Successfully created index at {args.output}")
        print(f"  Total slides: {len(index)}")

    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
