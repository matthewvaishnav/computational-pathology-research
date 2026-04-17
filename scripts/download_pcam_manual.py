"""
Manual PCam dataset download from direct URLs.

Downloads PCam dataset files directly from Zenodo/GitHub sources.
"""

import argparse
import gzip
import logging
import shutil
from pathlib import Path
from urllib.request import urlretrieve

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Direct download URLs from PCam GitHub
URLS = {
    "train_x": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz",
    "train_y": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz",
    "val_x": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz",
    "val_y": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz",
    "test_x": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz",
    "test_y": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz",
}


def download_and_extract(url: str, output_path: Path):
    """Download and extract gzipped file."""
    gz_path = output_path.with_suffix(output_path.suffix + ".gz")
    
    logger.info(f"Downloading {url}...")
    urlretrieve(url, gz_path)
    
    logger.info(f"Extracting {gz_path}...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove gz file
    gz_path.unlink()
    logger.info(f"Saved to {output_path}")


def convert_to_split_format(root_dir: Path):
    """Convert downloaded H5 files to split format."""
    logger.info("Converting to split format...")
    
    splits = {
        "train": ("train_x", "train_y"),
        "val": ("val_x", "val_y"),
        "test": ("test_x", "test_y"),
    }
    
    for split_name, (x_key, y_key) in splits.items():
        logger.info(f"Processing {split_name} split...")
        
        split_dir = root_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Load H5 files
        x_path = root_dir / "temp" / f"{x_key}.h5"
        y_path = root_dir / "temp" / f"{y_key}.h5"
        
        with h5py.File(x_path, "r") as f:
            images = f["x"][:]
        
        with h5py.File(y_path, "r") as f:
            labels = f["y"][:].squeeze()
        
        logger.info(f"  Images shape: {images.shape}")
        logger.info(f"  Labels shape: {labels.shape}")
        
        # Save as H5
        with h5py.File(split_dir / "images.h5py", "w") as f:
            f.create_dataset("images", data=images, compression="gzip")
        
        with h5py.File(split_dir / "labels.h5py", "w") as f:
            f.create_dataset("labels", data=labels, compression="gzip")
        
        logger.info(f"  Saved {len(images)} samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./data/pcam")
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    temp_dir = root_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Download all files
    for key, url in URLS.items():
        output_path = temp_dir / f"{key}.h5"
        if not output_path.exists():
            download_and_extract(url, output_path)
        else:
            logger.info(f"Skipping {key} (already downloaded)")
    
    # Convert to split format
    convert_to_split_format(root_dir)
    
    # Clean up temp directory
    logger.info("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    logger.info("Download complete!")


if __name__ == "__main__":
    main()
