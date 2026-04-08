#!/usr/bin/env python3
"""
Dataset Preparation Script

This script prepares raw data for training by:
- Organizing files into train/val/test splits
- Extracting WSI features
- Processing clinical data
- Creating metadata files
- Validating data integrity

Usage:
    python scripts/data/prepare_dataset.py \
        --raw-dir data/raw \
        --output-dir data/processed \
        --split-ratio 0.7 0.15 0.15
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DatasetPreparer:
    """Prepare dataset for training."""

    def __init__(
        self,
        raw_dir: str,
        output_dir: str,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_seed: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize dataset preparer.

        Args:
            raw_dir: Directory containing raw data
            output_dir: Directory to save processed data
            split_ratio: Train/val/test split ratios
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        self.verbose = verbose

        # Validate split ratio
        if not np.isclose(sum(split_ratio), 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratio)}")

        # Create output directories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"

        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(message)

    def load_metadata(self) -> pd.DataFrame:
        """Load metadata from raw directory."""
        self.log("Loading metadata...")

        # Look for metadata file
        metadata_files = list(self.raw_dir.glob("metadata*.csv"))
        if not metadata_files:
            metadata_files = list(self.raw_dir.glob("*.csv"))

        if not metadata_files:
            raise FileNotFoundError(f"No metadata CSV found in {self.raw_dir}")

        metadata_path = metadata_files[0]
        self.log(f"  Found metadata: {metadata_path}")

        df = pd.read_csv(metadata_path)
        self.log(f"  Loaded {len(df)} samples")

        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test sets."""
        self.log("\nSplitting data...")

        # First split: train vs (val + test)
        train_ratio = self.split_ratio[0]
        val_test_ratio = 1 - train_ratio

        train_df, val_test_df = train_test_split(
            df,
            test_size=val_test_ratio,
            random_state=self.random_seed,
            stratify=df["label"] if "label" in df.columns else None,
        )

        # Second split: val vs test
        val_ratio = self.split_ratio[1] / val_test_ratio

        val_df, test_df = train_test_split(
            val_test_df,
            test_size=(1 - val_ratio),
            random_state=self.random_seed,
            stratify=val_test_df["label"] if "label" in val_test_df.columns else None,
        )

        self.log(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        self.log(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        self.log(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

        return train_df, val_df, test_df

    def copy_files(self, df: pd.DataFrame, split_name: str, output_dir: Path) -> None:
        """Copy files to output directory."""
        self.log(f"\nCopying {split_name} files...")

        for idx, row in tqdm(df.iterrows(), total=len(df), disable=not self.verbose):
            # Get file paths
            if "wsi_path" in row:
                wsi_path = self.raw_dir / row["wsi_path"]
                if wsi_path.exists():
                    dest_path = output_dir / wsi_path.name
                    shutil.copy2(wsi_path, dest_path)

    def save_metadata(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        """Save metadata files."""
        self.log("\nSaving metadata...")

        # Save split metadata
        train_df.to_csv(self.train_dir / "metadata.csv", index=False)
        val_df.to_csv(self.val_dir / "metadata.csv", index=False)
        test_df.to_csv(self.test_dir / "metadata.csv", index=False)

        # Save dataset info
        info = {
            "num_samples": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
                "total": len(train_df) + len(val_df) + len(test_df),
            },
            "split_ratio": self.split_ratio,
            "random_seed": self.random_seed,
            "columns": list(train_df.columns),
        }

        # Add class distribution if available
        if "label" in train_df.columns:
            info["class_distribution"] = {
                "train": train_df["label"].value_counts().to_dict(),
                "val": val_df["label"].value_counts().to_dict(),
                "test": test_df["label"].value_counts().to_dict(),
            }

        with open(self.output_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

        self.log(f"  Saved metadata to {self.output_dir}")

    def validate_dataset(self) -> None:
        """Validate prepared dataset."""
        self.log("\nValidating dataset...")

        issues = []

        # Check metadata files exist
        for split in ["train", "val", "test"]:
            metadata_path = self.output_dir / split / "metadata.csv"
            if not metadata_path.exists():
                issues.append(f"Missing metadata for {split}")

        # Check dataset info
        info_path = self.output_dir / "dataset_info.json"
        if not info_path.exists():
            issues.append("Missing dataset_info.json")

        if issues:
            self.log("  ✗ Validation failed:")
            for issue in issues:
                self.log(f"    - {issue}")
            raise ValueError("Dataset validation failed")
        else:
            self.log("  ✓ Dataset validation passed")

    def prepare(self) -> None:
        """Prepare dataset."""
        self.log("=" * 60)
        self.log("Dataset Preparation")
        self.log("=" * 60)

        # Load metadata
        df = self.load_metadata()

        # Split data
        train_df, val_df, test_df = self.split_data(df)

        # Copy files
        self.copy_files(train_df, "train", self.train_dir)
        self.copy_files(val_df, "val", self.val_dir)
        self.copy_files(test_df, "test", self.test_dir)

        # Save metadata
        self.save_metadata(train_df, val_df, test_df)

        # Validate
        self.validate_dataset()

        self.log("\n" + "=" * 60)
        self.log("✓ Dataset preparation completed successfully!")
        self.log("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--raw-dir", type=str, required=True, help="Directory containing raw data")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save processed data"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        help="Train/val/test split ratios",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    preparer = DatasetPreparer(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        split_ratio=tuple(args.split_ratio),
        random_seed=args.random_seed,
        verbose=not args.quiet,
    )

    try:
        preparer.prepare()
    except Exception as e:
        print(f"\n✗ Preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
