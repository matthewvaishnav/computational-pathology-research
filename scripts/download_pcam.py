#!/usr/bin/env python3
"""
Download the real PatchCamelyon dataset from Zenodo.

Dataset: 327,680 color images (96x96px) extracted from histopathologic scans
Source: https://zenodo.org/record/2546921
Total size: ~7GB compressed, ~7GB extracted

Usage:
    python scripts/download_pcam.py
    python scripts/download_pcam.py --output-dir data/pcam_real
"""

import argparse
import gzip
import hashlib
import shutil
from pathlib import Path
from urllib.request import urlretrieve

# Zenodo URLs for PCam dataset
PCAM_FILES = {
    "train_x": {
        "url": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz",
        "size_gb": 2.0,
    },
    "train_y": {
        "url": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz",
        "size_gb": 0.01,
    },
    "valid_x": {
        "url": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz",
        "size_gb": 0.25,
    },
    "valid_y": {
        "url": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz",
        "size_gb": 0.001,
    },
    "test_x": {
        "url": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz",
        "size_gb": 0.25,
    },
    "test_y": {
        "url": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz",
        "size_gb": 0.001,
    },
}


def download_with_progress(url: str, output_path: Path):
    """Download file with progress bar."""
    print(f"Downloading {url}")
    print(f"  -> {output_path}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(
                f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                end="",
            )

    urlretrieve(url, output_path, reporthook=progress_hook)
    print()  # New line after progress


def extract_gz(gz_path: Path, output_path: Path):
    """Extract .gz file."""
    print(f"Extracting {gz_path.name}")
    print(f"  -> {output_path}")

    with gzip.open(gz_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"  Extracted successfully")


def main():
    parser = argparse.ArgumentParser(description="Download PatchCamelyon dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pcam_real",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--keep-compressed",
        action="store_true",
        help="Keep .gz files after extraction",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PatchCamelyon Dataset Download")
    print("=" * 80)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Total files: {len(PCAM_FILES)}")
    print(
        f"Total size: ~{sum(f['size_gb'] for f in PCAM_FILES.values()):.2f} GB compressed"
    )
    print("=" * 80)
    print()

    for name, info in PCAM_FILES.items():
        url = info["url"]
        filename = url.split("/")[-1]
        gz_path = output_dir / filename
        h5_path = output_dir / filename.replace(".gz", "")

        # Check if already extracted
        if args.skip_existing and h5_path.exists():
            print(f"✓ {name}: Already exists, skipping")
            print()
            continue

        # Download
        if not gz_path.exists():
            download_with_progress(url, gz_path)
        else:
            print(f"✓ {name}: Compressed file already downloaded")

        # Extract
        if not h5_path.exists():
            extract_gz(gz_path, h5_path)
        else:
            print(f"✓ {name}: Already extracted")

        # Clean up compressed file
        if not args.keep_compressed and gz_path.exists():
            print(f"  Removing {gz_path.name}")
            gz_path.unlink()

        print()

    print("=" * 80)
    print("Download Complete!")
    print("=" * 80)
    print(f"Dataset location: {output_dir.absolute()}")
    print()
    print("Files:")
    for h5_file in sorted(output_dir.glob("*.h5")):
        size_mb = h5_file.stat().st_size / (1024 * 1024)
        print(f"  {h5_file.name}: {size_mb:.1f} MB")
    print()
    print("Next steps:")
    print("  1. Verify dataset integrity")
    print("  2. Run training: python experiments/train_pcam.py")
    print()


if __name__ == "__main__":
    main()
