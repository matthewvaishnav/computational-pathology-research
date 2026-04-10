#!/usr/bin/env python3
"""
Verify the downloaded PatchCamelyon dataset.

Usage:
    python scripts/verify_pcam.py
    python scripts/verify_pcam.py --data-dir data/pcam_real
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def verify_pcam_dataset(data_dir: Path):
    """Verify PCam dataset integrity."""
    print("=" * 80)
    print("PatchCamelyon Dataset Verification")
    print("=" * 80)
    print(f"Data directory: {data_dir.absolute()}")
    print()

    expected_files = {
        "train": {
            "x": "camelyonpatch_level_2_split_train_x.h5",
            "y": "camelyonpatch_level_2_split_train_y.h5",
            "expected_samples": 262144,
        },
        "valid": {
            "x": "camelyonpatch_level_2_split_valid_x.h5",
            "y": "camelyonpatch_level_2_split_valid_y.h5",
            "expected_samples": 32768,
        },
        "test": {
            "x": "camelyonpatch_level_2_split_test_x.h5",
            "y": "camelyonpatch_level_2_split_test_y.h5",
            "expected_samples": 32768,
        },
    }

    all_valid = True

    for split_name, split_info in expected_files.items():
        print(f"Checking {split_name} split...")

        x_path = data_dir / split_info["x"]
        y_path = data_dir / split_info["y"]

        # Check files exist
        if not x_path.exists():
            print(f"  ✗ Missing: {split_info['x']}")
            all_valid = False
            continue
        if not y_path.exists():
            print(f"  ✗ Missing: {split_info['y']}")
            all_valid = False
            continue

        # Check file contents
        try:
            with h5py.File(x_path, "r") as f:
                x_data = f["x"]
                x_shape = x_data.shape
                x_dtype = x_data.dtype

            with h5py.File(y_path, "r") as f:
                y_data = f["y"]
                y_shape = y_data.shape
                y_dtype = y_data.dtype

            # Verify shapes
            expected_samples = split_info["expected_samples"]
            if x_shape[0] != expected_samples:
                print(f"  ✗ Wrong number of samples: {x_shape[0]} (expected {expected_samples})")
                all_valid = False
            elif x_shape != (expected_samples, 96, 96, 3):
                print(f"  ✗ Wrong x shape: {x_shape} (expected ({expected_samples}, 96, 96, 3))")
                all_valid = False
            elif y_shape != (expected_samples, 1, 1, 1):
                print(f"  ✗ Wrong y shape: {y_shape} (expected ({expected_samples}, 1, 1, 1))")
                all_valid = False
            else:
                print(f"  ✓ {split_name} split valid")
                print(f"    - Samples: {x_shape[0]:,}")
                print(f"    - Image shape: {x_shape[1:]}")
                print(f"    - Image dtype: {x_dtype}")
                print(f"    - Label shape: {y_shape}")
                print(f"    - Label dtype: {y_dtype}")

                # Check label distribution
                with h5py.File(y_path, "r") as f:
                    labels = f["y"][:].flatten()
                    n_positive = np.sum(labels == 1)
                    n_negative = np.sum(labels == 0)
                    pos_ratio = n_positive / len(labels) * 100

                print(f"    - Positive samples: {n_positive:,} ({pos_ratio:.1f}%)")
                print(f"    - Negative samples: {n_negative:,} ({100-pos_ratio:.1f}%)")

        except Exception as e:
            print(f"  ✗ Error reading files: {e}")
            all_valid = False

        print()

    print("=" * 80)
    if all_valid:
        print("✓ Dataset verification PASSED")
        print()
        print("Total samples: 327,680")
        print("  - Train: 262,144")
        print("  - Valid: 32,768")
        print("  - Test: 32,768")
        print()
        print("Ready to train!")
    else:
        print("✗ Dataset verification FAILED")
        print()
        print("Please re-download the dataset:")
        print("  python scripts/download_pcam.py")
    print("=" * 80)

    return all_valid


def main():
    parser = argparse.ArgumentParser(description="Verify PatchCamelyon dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/pcam_real",
        help="Directory containing PCam dataset",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        print()
        print("Please download the dataset first:")
        print("  python scripts/download_pcam.py")
        return 1

    success = verify_pcam_dataset(data_dir)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
