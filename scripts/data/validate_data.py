#!/usr/bin/env python3
"""
Data Validation Script

This script validates dataset integrity by checking:
- File existence and accessibility
- Data format and structure
- Missing values
- Label distribution
- Feature statistics
- Data quality issues

Usage:
    python scripts/data/validate_data.py --data-dir data/processed
    python scripts/data/validate_data.py --data-dir data/processed --split train
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DataValidator:
    """Validate dataset integrity and quality."""

    def __init__(self, data_dir: str, split: Optional[str] = None, verbose: bool = True):
        """
        Initialize data validator.

        Args:
            data_dir: Directory containing processed data
            split: Specific split to validate (train/val/test), or None for all
            verbose: Whether to print detailed information
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.verbose = verbose

        self.issues = []
        self.warnings = []

    def log(self, message: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(message)

    def add_issue(self, issue: str) -> None:
        """Add critical issue."""
        self.issues.append(issue)
        if self.verbose:
            print(f"  ✗ ISSUE: {issue}")

    def add_warning(self, warning: str) -> None:
        """Add warning."""
        self.warnings.append(warning)
        if self.verbose:
            print(f"  ⚠ WARNING: {warning}")

    def validate_directory_structure(self) -> None:
        """Validate directory structure."""
        self.log("\n" + "=" * 60)
        self.log("Validating Directory Structure")
        self.log("=" * 60)

        # Check main directory
        if not self.data_dir.exists():
            self.add_issue(f"Data directory not found: {self.data_dir}")
            return

        # Check splits
        splits = [self.split] if self.split else ["train", "val", "test"]

        for split in splits:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                self.add_issue(f"Split directory not found: {split_dir}")
            else:
                self.log(f"✓ Found {split} directory")

                # Check metadata
                metadata_path = split_dir / "metadata.csv"
                if not metadata_path.exists():
                    self.add_issue(f"Metadata not found: {metadata_path}")
                else:
                    self.log(f"  ✓ Found metadata.csv")

    def validate_metadata(self, split: str) -> Optional[pd.DataFrame]:
        """Validate metadata file."""
        self.log(f"\nValidating {split} metadata...")

        metadata_path = self.data_dir / split / "metadata.csv"
        if not metadata_path.exists():
            return None

        try:
            df = pd.read_csv(metadata_path)
            self.log(f"  ✓ Loaded {len(df)} samples")

            # Check required columns
            required_cols = ["sample_id"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.add_warning(f"Missing columns: {missing_cols}")

            # Check for duplicates
            if df["sample_id"].duplicated().any():
                num_dupes = df["sample_id"].duplicated().sum()
                self.add_issue(f"Found {num_dupes} duplicate sample IDs")

            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.any():
                self.log("  Missing values:")
                for col, count in missing_counts[missing_counts > 0].items():
                    pct = count / len(df) * 100
                    self.log(f"    {col}: {count} ({pct:.1f}%)")
                    if pct > 10:
                        self.add_warning(f"{col} has {pct:.1f}% missing values")

            return df

        except Exception as e:
            self.add_issue(f"Failed to load metadata: {e}")
            return None

    def validate_labels(self, df: pd.DataFrame, split: str) -> None:
        """Validate label distribution."""
        self.log(f"\nValidating {split} labels...")

        if "label" not in df.columns:
            self.add_warning("No 'label' column found")
            return

        # Check label distribution
        label_counts = df["label"].value_counts()
        self.log(f"  Label distribution:")
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            self.log(f"    Class {label}: {count} ({pct:.1f}%)")

        # Check for class imbalance
        min_count = label_counts.min()
        max_count = label_counts.max()
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        if imbalance_ratio > 10:
            self.add_warning(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        elif imbalance_ratio > 5:
            self.add_warning(f"Moderate class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")

    def validate_features(self, df: pd.DataFrame, split: str) -> None:
        """Validate feature statistics."""
        self.log(f"\nValidating {split} features...")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            self.add_warning("No numeric features found")
            return

        self.log(f"  Found {len(numeric_cols)} numeric features")

        # Check for constant features
        for col in numeric_cols:
            if df[col].nunique() == 1:
                self.add_warning(f"Constant feature: {col}")

        # Check for outliers (simple IQR method)
        for col in numeric_cols:
            if col == "label" or col == "sample_id":
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            outliers = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()
            if outliers > 0:
                pct = outliers / len(df) * 100
                if pct > 5:
                    self.add_warning(f"{col} has {outliers} outliers ({pct:.1f}%)")

    def validate_files(self, df: pd.DataFrame, split: str) -> None:
        """Validate file existence."""
        self.log(f"\nValidating {split} files...")

        if "wsi_path" not in df.columns:
            self.add_warning("No 'wsi_path' column found, skipping file validation")
            return

        split_dir = self.data_dir / split
        missing_files = []

        for idx, row in tqdm(df.iterrows(), total=len(df), disable=not self.verbose):
            wsi_path = split_dir / row["wsi_path"]
            if not wsi_path.exists():
                missing_files.append(row["wsi_path"])

        if missing_files:
            self.add_issue(f"Found {len(missing_files)} missing files")
            if self.verbose and len(missing_files) <= 10:
                for file in missing_files:
                    self.log(f"    Missing: {file}")
        else:
            self.log(f"  ✓ All files exist")

    def validate_split(self, split: str) -> None:
        """Validate a single split."""
        self.log(f"\n{'=' * 60}")
        self.log(f"Validating {split.upper()} Split")
        self.log(f"{'=' * 60}")

        # Load metadata
        df = self.validate_metadata(split)
        if df is None:
            return

        # Validate labels
        self.validate_labels(df, split)

        # Validate features
        self.validate_features(df, split)

        # Validate files
        self.validate_files(df, split)

    def validate_dataset_info(self) -> None:
        """Validate dataset info file."""
        self.log(f"\n{'=' * 60}")
        self.log("Validating Dataset Info")
        self.log(f"{'=' * 60}")

        info_path = self.data_dir / "dataset_info.json"
        if not info_path.exists():
            self.add_warning("dataset_info.json not found")
            return

        try:
            with open(info_path) as f:
                info = json.load(f)

            self.log("  ✓ Dataset info loaded")
            self.log(f"    Total samples: {info.get('num_samples', {}).get('total', 'N/A')}")
            self.log(f"    Train: {info.get('num_samples', {}).get('train', 'N/A')}")
            self.log(f"    Val: {info.get('num_samples', {}).get('val', 'N/A')}")
            self.log(f"    Test: {info.get('num_samples', {}).get('test', 'N/A')}")

        except Exception as e:
            self.add_issue(f"Failed to load dataset info: {e}")

    def print_summary(self) -> None:
        """Print validation summary."""
        self.log(f"\n{'=' * 60}")
        self.log("Validation Summary")
        self.log(f"{'=' * 60}")

        if not self.issues and not self.warnings:
            self.log("✓ All validation checks passed!")
        else:
            if self.issues:
                self.log(f"\n✗ Found {len(self.issues)} critical issues:")
                for issue in self.issues:
                    self.log(f"  - {issue}")

            if self.warnings:
                self.log(f"\n⚠ Found {len(self.warnings)} warnings:")
                for warning in self.warnings:
                    self.log(f"  - {warning}")

    def validate(self) -> bool:
        """Run all validation checks."""
        # Validate directory structure
        self.validate_directory_structure()

        # Validate dataset info
        if not self.split:
            self.validate_dataset_info()

        # Validate splits
        splits = [self.split] if self.split else ["train", "val", "test"]
        for split in splits:
            self.validate_split(split)

        # Print summary
        self.print_summary()

        # Return True if no critical issues
        return len(self.issues) == 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate dataset integrity and quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir", type=str, required=True, help="Directory containing processed data"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        help="Specific split to validate (default: all)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    validator = DataValidator(data_dir=args.data_dir, split=args.split, verbose=not args.quiet)

    try:
        success = validator.validate()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
