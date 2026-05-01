#!/usr/bin/env python3
"""
Foundation Model Weight Downloader

Downloads and verifies foundation model weights for UNI, Phikon, CONCH, CTransPath.
Handles authentication, progress tracking, and checksum verification.
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.pretrained import PRETRAINED_MODELS

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Downloads and manages foundation model weights."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize downloader with cache directory."""
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/medical_ai_models"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # HuggingFace token for private models
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """
        Download a specific foundation model.
        
        Args:
            model_name: Name of model to download
            force: Force re-download even if cached
            
        Returns:
            True if successful
        """
        if model_name not in PRETRAINED_MODELS:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        config = PRETRAINED_MODELS[model_name]
        
        # Skip torchvision models (downloaded automatically)
        if config.get('source') == 'torchvision':
            logger.info(f"Skipping {model_name} (torchvision model)")
            return True
        
        # Handle HuggingFace models
        if config.get('source', '').startswith('hf_hub:'):
            return self._download_huggingface_model(model_name, config, force)
        
        # Handle direct download models
        if config.get('download_url'):
            return self._download_direct_model(model_name, config, force)
        
        logger.warning(f"No download method for {model_name}")
        return False
    
    def _download_huggingface_model(self, model_name: str, config: Dict, force: bool) -> bool:
        """Download model from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download, login
            
            # Login if token available
            if self.hf_token:
                login(token=self.hf_token)
            
            repo_id = config['source'].replace('hf_hub:', '')
            
            # Common model files to download
            files_to_download = [
                'pytorch_model.bin',
                'model.safetensors', 
                'config.json',
                'preprocessor_config.json'
            ]
            
            logger.info(f"Downloading {model_name} from HuggingFace: {repo_id}")
            
            downloaded_files = []
            for filename in files_to_download:
                try:
                    file_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=str(self.cache_dir),
                        force_download=force
                    )
                    downloaded_files.append(file_path)
                    logger.info(f"Downloaded: {filename}")
                except Exception as e:
                    logger.debug(f"Could not download {filename}: {e}")
            
            if downloaded_files:
                logger.info(f"Successfully downloaded {model_name} ({len(downloaded_files)} files)")
                return True
            else:
                logger.error(f"No files downloaded for {model_name}")
                return False
                
        except ImportError:
            logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False
    
    def _download_direct_model(self, model_name: str, config: Dict, force: bool) -> bool:
        """Download model from direct URL."""
        url = config['download_url']
        filename = f"{model_name}_weights.pth"
        filepath = self.cache_dir / filename
        
        # Check if already exists
        if filepath.exists() and not force:
            if self._verify_file(filepath, config.get('checksum')):
                logger.info(f"Using cached {model_name}: {filepath}")
                return True
            else:
                logger.warning(f"Cached file corrupted, re-downloading: {filepath}")
                filepath.unlink()
        
        logger.info(f"Downloading {model_name} from {url}")
        
        try:
            # Handle Google Drive URLs
            if 'drive.google.com' in url:
                return self._download_google_drive(url, filepath)
            
            # Standard HTTP download
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify download
            if self._verify_file(filepath, config.get('checksum')):
                logger.info(f"Successfully downloaded {model_name}: {filepath}")
                return True
            else:
                filepath.unlink()
                logger.error(f"Download verification failed for {model_name}")
                return False
                
        except Exception as e:
            if filepath.exists():
                filepath.unlink()
            logger.error(f"Failed to download {model_name}: {e}")
            return False
    
    def _download_google_drive(self, url: str, filepath: Path) -> bool:
        """Download from Google Drive with session handling."""
        try:
            import re
            
            # Extract file ID from Google Drive URL
            file_id_match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
            if not file_id_match:
                logger.error("Could not extract Google Drive file ID")
                return False
            
            file_id = file_id_match.group(1)
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            session = requests.Session()
            response = session.get(download_url, stream=True)
            
            # Handle Google Drive virus scan warning
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    params = {'id': file_id, 'confirm': value}
                    response = session.get(download_url, params=params, stream=True)
                    break
            
            response.raise_for_status()
            
            # Download with progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded from Google Drive: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Google Drive download failed: {e}")
            return False
    
    def _verify_file(self, filepath: Path, expected_checksum: Optional[str]) -> bool:
        """Verify downloaded file integrity."""
        if not filepath.exists():
            return False
        
        # Basic size check (models should be > 10MB)
        file_size = filepath.stat().st_size
        if file_size < 10 * 1024 * 1024:
            logger.warning(f"File suspiciously small: {file_size} bytes")
            return False
        
        # Checksum verification (when checksums are provided in model config)
        # Note: Most foundation models don't publish official checksums
        # This implementation will verify if checksum is provided and not placeholder
        if expected_checksum and expected_checksum != "placeholder":
            # Calculate SHA256
            sha256_hash = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            calculated = sha256_hash.hexdigest()
            if calculated != expected_checksum:
                logger.error(f"Checksum mismatch: expected {expected_checksum}, got {calculated}")
                return False
            logger.info(f"Checksum verification passed: {calculated}")
        else:
            logger.debug("No checksum provided for verification")
        
        logger.info(f"File verification passed: {filepath}")
        return True
    
    def download_all(self, force: bool = False) -> Dict[str, bool]:
        """Download all available foundation models."""
        results = {}
        
        for model_name in PRETRAINED_MODELS:
            logger.info(f"Processing {model_name}...")
            results[model_name] = self.download_model(model_name, force)
        
        return results
    
    def list_cached_models(self) -> Dict[str, Dict]:
        """List all cached models with metadata."""
        cached = {}
        
        for model_name, config in PRETRAINED_MODELS.items():
            model_files = []
            
            # Check for various possible filenames
            possible_files = [
                f"{model_name}_weights.pth",
                f"{model_name}.pth",
                "pytorch_model.bin",
                "model.safetensors"
            ]
            
            for filename in possible_files:
                filepath = self.cache_dir / filename
                if filepath.exists():
                    model_files.append({
                        'filename': filename,
                        'size_mb': filepath.stat().st_size / (1024 * 1024),
                        'modified': filepath.stat().st_mtime
                    })
            
            if model_files:
                cached[model_name] = {
                    'config': config,
                    'files': model_files,
                    'total_size_mb': sum(f['size_mb'] for f in model_files)
                }
        
        return cached


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Download foundation model weights")
    parser.add_argument(
        '--model', 
        choices=list(PRETRAINED_MODELS.keys()) + ['all'],
        default='all',
        help='Model to download (default: all)'
    )
    parser.add_argument(
        '--cache-dir',
        help='Cache directory for models'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if cached'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List cached models'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize downloader
    downloader = ModelDownloader(args.cache_dir)
    
    # List cached models
    if args.list:
        cached = downloader.list_cached_models()
        if cached:
            print("\nCached Models:")
            for model_name, info in cached.items():
                print(f"  {model_name}: {info['total_size_mb']:.1f} MB ({len(info['files'])} files)")
        else:
            print("No cached models found")
        return
    
    # Download models
    if args.model == 'all':
        print("Downloading all foundation models...")
        results = downloader.download_all(args.force)
    else:
        print(f"Downloading {args.model}...")
        results = {args.model: downloader.download_model(args.model, args.force)}
    
    # Print results
    print("\nDownload Results:")
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {model_name}: {status}")
    
    # Summary
    successful = sum(results.values())
    total = len(results)
    print(f"\nCompleted: {successful}/{total} models downloaded successfully")
    
    if successful > 0:
        print(f"\nModels cached in: {downloader.cache_dir}")
        print("You can now use these models with PretrainedFeatureExtractor!")


if __name__ == "__main__":
    main()