"""
Download PANDA dataset from Kaggle.

Requires Kaggle API credentials (~/.kaggle/kaggle.json)

Usage:
    python scripts/download_panda.py --output_dir data/panda
"""

import argparse
import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_kaggle_api():
    """Setup and authenticate Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Kaggle API not installed. Install with: pip install kaggle\n"
            "Then setup credentials: https://github.com/Kaggle/kaggle-api#api-credentials"
        )
    
    api = KaggleApi()
    api.authenticate()
    return api


def download_panda(output_dir: Path, download_images: bool = True, download_masks: bool = False):
    """Download PANDA dataset from Kaggle.
    
    Args:
        output_dir: Directory to save dataset
        download_images: Whether to download train images
        download_masks: Whether to download tissue masks
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading PANDA dataset to {output_dir}")
    
    api = setup_kaggle_api()
    competition = "prostate-cancer-grade-assessment"
    
    # Download files
    files_to_download = ["train.csv", "test.csv"]
    
    if download_images:
        files_to_download.append("train_images.zip")
    
    if download_masks:
        files_to_download.append("train_label_masks.zip")
    
    for file_name in files_to_download:
        logger.info(f"Downloading {file_name}...")
        api.competition_download_file(
            competition=competition,
            file_name=file_name,
            path=str(output_dir),
        )
        
        # Extract if zip
        if file_name.endswith(".zip"):
            zip_path = output_dir / file_name
            logger.info(f"Extracting {zip_path}...")
            
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Remove zip file
            zip_path.unlink()
            logger.info(f"Extracted and removed {file_name}")
    
    logger.info("PANDA dataset download complete!")
    logger.info(f"Dataset location: {output_dir}")
    logger.info(f"Train CSV: {output_dir / 'train.csv'}")
    logger.info(f"Train images: {output_dir / 'train_images'}")


def main():
    parser = argparse.ArgumentParser(description="Download PANDA dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/panda",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="Skip downloading train images (only download CSVs)",
    )
    parser.add_argument(
        "--download_masks",
        action="store_true",
        help="Download tissue masks (large file ~50GB)",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    try:
        download_panda(
            output_dir=Path(args.output_dir),
            download_images=not args.no_images,
            download_masks=args.download_masks,
        )
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.error(
            "\nTROUBLESHOOTING:\n"
            "1. Install Kaggle API: pip install kaggle\n"
            "2. Setup credentials:\n"
            "   - Go to https://www.kaggle.com/settings\n"
            "   - Click 'Create New API Token'\n"
            "   - Save kaggle.json to ~/.kaggle/kaggle.json\n"
            "   - On Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json\n"
            "3. Accept competition rules:\n"
            "   - Visit https://www.kaggle.com/c/prostate-cancer-grade-assessment\n"
            "   - Click 'Join Competition' and accept rules\n"
        )
        raise


if __name__ == "__main__":
    main()
