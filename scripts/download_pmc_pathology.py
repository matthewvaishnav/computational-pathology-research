#!/usr/bin/env python3
"""
Download pathology images from PubMed Central Open Access Subset.

This script searches PMC for pathology-related articles and downloads
images with their captions for vision-language training.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import time
from tqdm import tqdm

try:
    from Bio import Entrez
    import requests
except ImportError:
    print("Error: Required packages not installed")
    print("Install with: pip install biopython requests")
    sys.exit(1)


# Set your email for NCBI Entrez (required)
Entrez.email = "your_email@example.com"


def search_pmc_articles(
    keywords: List[str],
    max_results: int = 1000,
    retstart: int = 0
) -> List[str]:
    """Search PMC for articles matching keywords."""
    
    # Build search query
    query = " OR ".join(keywords)
    query += " AND open access[filter]"
    
    print(f"Searching PMC for: {query}")
    
    try:
        handle = Entrez.esearch(
            db="pmc",
            term=query,
            retmax=max_results,
            retstart=retstart,
            sort="relevance"
        )
        
        record = Entrez.read(handle)
        handle.close()
        
        pmc_ids = record["IdList"]
        total_count = int(record["Count"])
        
        print(f"Found {total_count} articles, retrieving {len(pmc_ids)}")
        
        return pmc_ids
    
    except Exception as e:
        print(f"Error searching PMC: {e}")
        return []


def fetch_article_metadata(pmc_id: str) -> Optional[Dict]:
    """Fetch metadata for a PMC article."""
    
    try:
        handle = Entrez.efetch(
            db="pmc",
            id=pmc_id,
            rettype="xml"
        )
        
        # Parse XML (simplified - full parsing would be more complex)
        xml_data = handle.read()
        handle.close()
        
        # Extract basic info (this is simplified)
        return {
            "pmc_id": pmc_id,
            "xml": xml_data
        }
    
    except Exception as e:
        print(f"Error fetching article {pmc_id}: {e}")
        return None


def extract_images_from_article(pmc_id: str) -> List[Dict]:
    """Extract image URLs and captions from a PMC article."""
    
    # PMC images are available via FTP
    # Format: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{id}/bin/{filename}
    
    base_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
    
    # This is a simplified version - full implementation would parse the XML
    # to extract actual image filenames and captions
    
    images = []
    
    # Try common image patterns
    for i in range(1, 20):  # Try up to 20 figures
        for ext in ['jpg', 'png', 'tif']:
            image_url = f"{base_url}bin/figure{i}.{ext}"
            
            # Check if image exists
            try:
                response = requests.head(image_url, timeout=5)
                if response.status_code == 200:
                    images.append({
                        "url": image_url,
                        "filename": f"PMC{pmc_id}_figure{i}.{ext}",
                        "caption": f"Figure {i} from PMC{pmc_id}"
                    })
            except:
                continue
    
    return images


def download_image(url: str, output_path: Path) -> bool:
    """Download an image from URL."""
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return True
    
    except Exception as e:
        return False


def filter_pathology_images(image_path: Path, caption: str) -> bool:
    """
    Filter images to keep only pathology-related ones.
    
    This is a simple keyword-based filter. A more sophisticated
    approach would use image classification.
    """
    
    pathology_keywords = [
        'histopathology', 'histology', 'microscopy', 'tissue',
        'staining', 'h&e', 'hematoxylin', 'eosin',
        'immunohistochemistry', 'ihc', 'pathology',
        'biopsy', 'tumor', 'cancer', 'carcinoma',
        'adenocarcinoma', 'squamous', 'metastasis'
    ]
    
    caption_lower = caption.lower()
    
    return any(keyword in caption_lower for keyword in pathology_keywords)


def download_pmc_pathology_dataset(
    keywords: List[str],
    output_dir: Path,
    max_articles: int = 1000,
    min_images: int = 10000
):
    """Download pathology images from PMC."""
    
    print("=" * 60)
    print("PMC Pathology Image Downloader")
    print("=" * 60)
    
    # Create output directories
    images_dir = output_dir / "images"
    captions_dir = output_dir / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)
    
    # Search for articles
    pmc_ids = search_pmc_articles(keywords, max_results=max_articles)
    
    if not pmc_ids:
        print("No articles found")
        return
    
    # Download images from articles
    total_images = 0
    downloaded_images = 0
    metadata = []
    
    for pmc_id in tqdm(pmc_ids, desc="Processing articles"):
        # Extract images from article
        images = extract_images_from_article(pmc_id)
        
        for image_info in images:
            total_images += 1
            
            # Download image
            image_path = images_dir / image_info["filename"]
            
            if download_image(image_info["url"], image_path):
                # Check if it's a pathology image
                if filter_pathology_images(image_path, image_info["caption"]):
                    # Save caption
                    caption_path = captions_dir / f"{image_path.stem}.txt"
                    with open(caption_path, 'w') as f:
                        f.write(image_info["caption"])
                    
                    metadata.append({
                        "pmc_id": pmc_id,
                        "image": str(image_path.relative_to(output_dir)),
                        "caption": image_info["caption"],
                        "url": image_info["url"]
                    })
                    
                    downloaded_images += 1
                else:
                    # Remove non-pathology image
                    image_path.unlink()
            
            # Check if we've reached target
            if downloaded_images >= min_images:
                break
        
        if downloaded_images >= min_images:
            break
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save metadata
    metadata_path = output_dir / "pmc_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Download Complete")
    print(f"{'='*60}")
    print(f"Total images found: {total_images}")
    print(f"Pathology images downloaded: {downloaded_images}")
    print(f"Images directory: {images_dir}")
    print(f"Captions directory: {captions_dir}")
    print(f"Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download pathology images from PubMed Central"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for images and captions"
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="histopathology,pathology,microscopy",
        help="Comma-separated keywords for search"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=1000,
        help="Maximum number of articles to process"
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=10000,
        help="Minimum number of images to download"
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Your email for NCBI Entrez (required)"
    )
    
    args = parser.parse_args()
    
    # Set email for Entrez
    if args.email:
        Entrez.email = args.email
    elif Entrez.email == "your_email@example.com":
        print("Error: Please provide your email via --email")
        print("This is required by NCBI for API access")
        sys.exit(1)
    
    # Parse keywords
    keywords = [k.strip() for k in args.keywords.split(',')]
    
    print(f"Keywords: {keywords}")
    print(f"Max articles: {args.max_articles}")
    print(f"Min images: {args.min_images}")
    print(f"Output: {args.output_dir}")
    
    # Download dataset
    download_pmc_pathology_dataset(
        keywords=keywords,
        output_dir=args.output_dir,
        max_articles=args.max_articles,
        min_images=args.min_images
    )


if __name__ == "__main__":
    main()
