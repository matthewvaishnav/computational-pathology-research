#!/usr/bin/env python3
"""
Download publicly available medical imaging datasets.

This script automates the download of free, publicly available datasets
for multi-disease training.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import tarfile


def download_file(url: str, output_path: Path, description: str = ""):
    """Download a file with progress bar."""
    print(f"Downloading {description or url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
    
    print(f"✓ Downloaded to {output_path}")


def extract_archive(archive_path: Path, extract_to: Path):
    """Extract zip or tar archive."""
    print(f"Extracting {archive_path.name}...")
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        print(f"⚠ Unknown archive format: {archive_path.suffix}")
        return
    
    print(f"✓ Extracted to {extract_to}")


def download_lc25000(data_root: Path):
    """Download LC25000 lung and colon cancer dataset."""
    print("\n=== Downloading LC25000 (Lung & Colon Cancer) ===")
    
    lung_dir = data_root / "multi_disease" / "lung" / "lc25000"
    
    # Clone from GitHub
    if lung_dir.exists():
        print(f"⚠ {lung_dir} already exists, skipping...")
        return
    
    print("Cloning LC25000 repository...")
    subprocess.run([
        "git", "clone",
        "https://github.com/tampapath/lung_colon_image_set.git",
        str(lung_dir)
    ], check=True)
    
    print("✓ LC25000 downloaded successfully")
    print(f"  Location: {lung_dir}")
    print(f"  Contains: 25,000 lung histopathology images")


def download_nct_crc(data_root: Path):
    """Download NCT-CRC-HE-100K colorectal cancer dataset."""
    print("\n=== Downloading NCT-CRC-HE-100K (Colorectal Cancer) ===")
    
    colon_dir = data_root / "multi_disease" / "colon" / "nct_crc"
    archive_path = colon_dir.parent / "NCT-CRC-HE-100K.zip"
    
    if colon_dir.exists():
        print(f"⚠ {colon_dir} already exists, skipping...")
        return
    
    # Download from Zenodo
    url = "https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip"
    download_file(url, archive_path, "NCT-CRC-HE-100K")
    
    # Extract
    extract_archive(archive_path, colon_dir)
    
    # Clean up archive
    archive_path.unlink()
    
    print("✓ NCT-CRC-HE-100K downloaded successfully")
    print(f"  Location: {colon_dir}")
    print(f"  Contains: 100,000 colorectal cancer patches")


def download_ham10000(data_root: Path):
    """Download HAM10000 melanoma dataset."""
    print("\n=== Downloading HAM10000 (Melanoma) ===")
    
    melanoma_dir = data_root / "multi_disease" / "melanoma" / "ham10000"
    
    if melanoma_dir.exists():
        print(f"⚠ {melanoma_dir} already exists, skipping...")
        return
    
    print("HAM10000 requires manual download from Harvard Dataverse:")
    print("  1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
    print("  2. Download all files")
    print(f"  3. Extract to: {melanoma_dir}")
    print("\nAlternatively, use the Kaggle dataset:")
    print("  kaggle datasets download -d kmader/skin-cancer-mnist-ham10000")


def download_crc_val(data_root: Path):
    """Download CRC-VAL-HE-7K colorectal cancer validation set."""
    print("\n=== Downloading CRC-VAL-HE-7K (Colorectal Validation) ===")
    
    colon_dir = data_root / "multi_disease" / "colon" / "crc_val"
    archive_path = colon_dir.parent / "CRC-VAL-HE-7K.zip"
    
    if colon_dir.exists():
        print(f"⚠ {colon_dir} already exists, skipping...")
        return
    
    # Download from Zenodo
    url = "https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip"
    download_file(url, archive_path, "CRC-VAL-HE-7K")
    
    # Extract
    extract_archive(archive_path, colon_dir)
    
    # Clean up archive
    archive_path.unlink()
    
    print("✓ CRC-VAL-HE-7K downloaded successfully")
    print(f"  Location: {colon_dir}")
    print(f"  Contains: 7,180 colorectal cancer images")


def download_sicap(data_root: Path):
    """Download SICAPv2 prostate cancer dataset."""
    print("\n=== Downloading SICAPv2 (Prostate Cancer) ===")
    
    prostate_dir = data_root / "multi_disease" / "prostate" / "sicap"
    
    if prostate_dir.exists():
        print(f"⚠ {prostate_dir} already exists, skipping...")
        return
    
    print("SICAPv2 requires manual download from Mendeley Data:")
    print("  1. Visit: https://data.mendeley.com/datasets/9xxm58dvs3/1")
    print("  2. Download the dataset")
    print(f"  3. Extract to: {prostate_dir}")


def check_kaggle_setup():
    """Check if Kaggle CLI is set up."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_panda(data_root: Path):
    """Download PANDA prostate cancer dataset from Kaggle."""
    print("\n=== Downloading PANDA (Prostate Cancer) ===")
    
    prostate_dir = data_root / "multi_disease" / "prostate" / "panda"
    
    if prostate_dir.exists():
        print(f"⚠ {prostate_dir} already exists, skipping...")
        return
    
    if not check_kaggle_setup():
        print("⚠ Kaggle CLI not found. Please install:")
        print("  pip install kaggle")
        print("  Then set up API credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        return
    
    print("Downloading PANDA dataset from Kaggle...")
    prostate_dir.mkdir(parents=True, exist_ok=True)
    
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "prostate-cancer-grade-assessment",
        "-p", str(prostate_dir)
    ], check=True)
    
    print("✓ PANDA downloaded successfully")
    print(f"  Location: {prostate_dir}")
    print(f"  Contains: 10,616 WSI with Gleason grading")


def create_dataset_registry(data_root: Path):
    """Create a registry of downloaded datasets."""
    import json
    
    registry = {
        "multi_disease": {
            "lung": {
                "lc25000": {
                    "path": "data/multi_disease/lung/lc25000",
                    "num_images": 25000,
                    "classes": ["normal", "adenocarcinoma", "squamous_cell_carcinoma"],
                    "license": "CC0 1.0",
                    "source": "https://github.com/tampapath/lung_colon_image_set"
                }
            },
            "colon": {
                "nct_crc": {
                    "path": "data/multi_disease/colon/nct_crc",
                    "num_images": 100000,
                    "classes": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
                    "license": "CC BY 4.0",
                    "source": "https://zenodo.org/record/1214456"
                },
                "crc_val": {
                    "path": "data/multi_disease/colon/crc_val",
                    "num_images": 7180,
                    "classes": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
                    "license": "CC BY 4.0",
                    "source": "https://zenodo.org/record/1214456"
                }
            },
            "melanoma": {
                "ham10000": {
                    "path": "data/multi_disease/melanoma/ham10000",
                    "num_images": 10015,
                    "classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                    "license": "CC BY-NC 4.0",
                    "source": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T"
                }
            },
            "prostate": {
                "panda": {
                    "path": "data/multi_disease/prostate/panda",
                    "num_slides": 10616,
                    "classes": ["gleason_0", "gleason_6", "gleason_7", "gleason_8", "gleason_9", "gleason_10"],
                    "license": "Competition license",
                    "source": "https://www.kaggle.com/c/prostate-cancer-grade-assessment"
                },
                "sicap": {
                    "path": "data/multi_disease/prostate/sicap",
                    "num_patches": 18783,
                    "classes": ["NC", "G3", "G4", "G5"],
                    "license": "CC BY 4.0",
                    "source": "https://data.mendeley.com/datasets/9xxm58dvs3/1"
                }
            },
            "breast": {
                "pcam": {
                    "path": "data/pcam_real",
                    "num_images": 327680,
                    "classes": ["normal", "tumor"],
                    "license": "CC0 1.0",
                    "source": "https://github.com/basveeling/pcam"
                }
            }
        }
    }
    
    registry_path = data_root / "metadata" / "dataset_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n✓ Dataset registry created: {registry_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download publicly available medical imaging datasets"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for datasets (default: data/)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["lc25000", "nct_crc", "crc_val", "ham10000", "panda", "sicap", "all"],
        default=["all"],
        help="Datasets to download (default: all)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Medical Imaging Dataset Downloader")
    print("=" * 60)
    
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["lc25000", "nct_crc", "crc_val", "ham10000", "panda", "sicap"]
    
    # Download selected datasets
    if "lc25000" in datasets:
        try:
            download_lc25000(args.data_root)
        except Exception as e:
            print(f"✗ Failed to download LC25000: {e}")
    
    if "nct_crc" in datasets:
        try:
            download_nct_crc(args.data_root)
        except Exception as e:
            print(f"✗ Failed to download NCT-CRC: {e}")
    
    if "crc_val" in datasets:
        try:
            download_crc_val(args.data_root)
        except Exception as e:
            print(f"✗ Failed to download CRC-VAL: {e}")
    
    if "ham10000" in datasets:
        download_ham10000(args.data_root)
    
    if "panda" in datasets:
        try:
            download_panda(args.data_root)
        except Exception as e:
            print(f"✗ Failed to download PANDA: {e}")
    
    if "sicap" in datasets:
        download_sicap(args.data_root)
    
    # Create dataset registry
    create_dataset_registry(args.data_root)
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print("\nAutomatically downloaded:")
    print("  ✓ LC25000 (Lung cancer)")
    print("  ✓ NCT-CRC-HE-100K (Colon cancer)")
    print("  ✓ CRC-VAL-HE-7K (Colon validation)")
    print("\nRequire manual download:")
    print("  ⚠ HAM10000 (Melanoma) - Harvard Dataverse")
    print("  ⚠ PANDA (Prostate) - Kaggle (requires API setup)")
    print("  ⚠ SICAPv2 (Prostate) - Mendeley Data")
    print("\nNext steps:")
    print("  1. Complete manual downloads")
    print("  2. Run: python scripts/verify_datasets.py")
    print("  3. Start training: python experiments/train_multi_disease.py")


if __name__ == "__main__":
    main()
