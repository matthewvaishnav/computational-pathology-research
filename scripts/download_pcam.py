#!/usr/bin/env python3
"""Download the real PatchCamelyon dataset from Zenodo."""

import argparse
import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://zenodo.org/record/2546921/files"
PCAM_FILES = {
    "train_x": {
        "filename": "camelyonpatch_level_2_split_train_x.h5.gz",
        "size_gb": 2.0,
    },
    "train_y": {
        "filename": "camelyonpatch_level_2_split_train_y.h5.gz",
        "size_gb": 0.01,
    },
    "valid_x": {
        "filename": "camelyonpatch_level_2_split_valid_x.h5.gz",
        "size_gb": 0.25,
    },
    "valid_y": {
        "filename": "camelyonpatch_level_2_split_valid_y.h5.gz",
        "size_gb": 0.001,
    },
    "test_x": {
        "filename": "camelyonpatch_level_2_split_test_x.h5.gz",
        "size_gb": 0.25,
    },
    "test_y": {
        "filename": "camelyonpatch_level_2_split_test_y.h5.gz",
        "size_gb": 0.001,
    },
}


def build_download_url(filename: str) -> str:
    """Build a Zenodo download URL."""
    return f"{BASE_URL}/{filename}"


def download_with_progress(url: str, output_path: Path) -> None:
    """Download file with progress output."""
    print(f"Downloading {url}")
    print(f"  -> {output_path}")

    def progress_hook(
        block_num: int,
        block_size: int,
        total_size: int,
    ) -> None:
        downloaded = block_num * block_size
        if total_size <= 0:
            return
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        progress = f"\r  Progress: {percent:.1f}% " f"({mb_downloaded:.1f}/{mb_total:.1f} MB)"
        print(progress, end="")

    urlretrieve(url, output_path, reporthook=progress_hook)
    print()


def extract_gz(gz_path: Path, output_path: Path) -> None:
    """Extract a .gz archive."""
    print(f"Extracting {gz_path.name}")
    print(f"  -> {output_path}")

    with gzip.open(gz_path, "rb") as compressed:
        with open(output_path, "wb") as output:
            shutil.copyfileobj(compressed, output)

    print("  Extracted successfully")


def main() -> None:
    """Download and extract the dataset."""
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

    total_size = sum(file_info["size_gb"] for file_info in PCAM_FILES.values())

    print("=" * 80)
    print("PatchCamelyon Dataset Download")
    print("=" * 80)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Total files: {len(PCAM_FILES)}")
    print(f"Total size: ~{total_size:.2f} GB compressed")
    print("=" * 80)
    print()

    for name, info in PCAM_FILES.items():
        filename = info["filename"]
        url = build_download_url(filename)
        gz_path = output_dir / filename
        h5_path = output_dir / filename.replace(".gz", "")

        if args.skip_existing and h5_path.exists():
            print(f"[OK] {name}: Already exists, skipping")
            print()
            continue

        if not gz_path.exists():
            download_with_progress(url, gz_path)
        else:
            print(f"[OK] {name}: Compressed file already downloaded")

        if not h5_path.exists():
            extract_gz(gz_path, h5_path)
        else:
            print(f"[OK] {name}: Already extracted")

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
