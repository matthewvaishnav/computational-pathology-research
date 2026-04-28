#!/usr/bin/env python3
"""
Multi-Disease Dataset Collection Framework

Automates collection, validation, and preparation of datasets for lung, prostate, 
colon, and melanoma cancer detection. Handles data partnerships, quality control,
and training pipeline integration.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class MultiDiseaseDatasetCollector:
    """Collects and manages multi-disease pathology datasets."""
    
    def __init__(self, data_root: Optional[str] = None):
        """Initialize dataset collector."""
        self.data_root = Path(data_root or "data/multi_disease")
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = {
            'lung': {
                'name': 'LC25000 Lung Cancer Dataset',
                'url': 'https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images',
                'classes': ['lung_aca', 'lung_scc', 'lung_n'],
                'target_samples': 15000,
                'image_size': (768, 768),
                'description': 'Lung adenocarcinoma, squamous cell carcinoma, and normal tissue'
            },
            'prostate': {
                'name': 'PANDA Prostate Cancer Dataset',
                'url': 'https://www.kaggle.com/competitions/prostate-cancer-grade-assessment',
                'classes': ['grade_0', 'grade_1', 'grade_2', 'grade_3', 'grade_4', 'grade_5'],
                'target_samples': 10000,
                'image_size': (512, 512),
                'description': 'Prostate cancer grading (Gleason scores)'
            },
            'colon': {
                'name': 'NCT-CRC-HE-100K Colorectal Cancer Dataset',
                'url': 'https://zenodo.org/record/1214456',
                'classes': ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'],
                'target_samples': 100000,
                'image_size': (224, 224),
                'description': 'Colorectal cancer histology classification'
            },
            'melanoma': {
                'name': 'HAM10000 Melanoma Dataset',
                'url': 'https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000',
                'classes': ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc'],
                'target_samples': 10000,
                'image_size': (450, 600),
                'description': 'Melanoma and skin lesion classification'
            }
        }
        
        # Data quality thresholds
        self.quality_thresholds = {
            'min_image_size': (224, 224),
            'max_image_size': (2048, 2048),
            'min_file_size': 1024,  # 1KB
            'max_file_size': 50 * 1024 * 1024,  # 50MB
            'supported_formats': ['.jpg', '.jpeg', '.png', '.tiff', '.tif'],
            'min_samples_per_class': 100
        }
        
        logger.info(f"Dataset root directory: {self.data_root}")
    
    def download_dataset(self, disease: str, force: bool = False) -> bool:
        """Download dataset for specific disease."""
        if disease not in self.dataset_configs:
            logger.error(f"Unknown disease: {disease}")
            return False
        
        config = self.dataset_configs[disease]
        dataset_dir = self.data_root / disease
        
        # Check if already exists
        if dataset_dir.exists() and not force:
            logger.info(f"{disease} dataset already exists: {dataset_dir}")
            return True
        
        logger.info(f"Downloading {config['name']}...")
        
        try:
            # Create dataset directory
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Download based on source
            if 'kaggle.com' in config['url']:
                return self._download_kaggle_dataset(disease, config, dataset_dir)
            elif 'zenodo.org' in config['url']:
                return self._download_zenodo_dataset(disease, config, dataset_dir)
            else:
                return self._download_direct_dataset(disease, config, dataset_dir)
                
        except Exception as e:
            logger.error(f"Failed to download {disease} dataset: {e}")
            return False
    
    def _download_kaggle_dataset(self, disease: str, config: Dict, dataset_dir: Path) -> bool:
        """Download dataset from Kaggle."""
        try:
            # Check for Kaggle API
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
            except ImportError:
                logger.error("Kaggle API not installed. Run: pip install kaggle")
                return False
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Extract dataset identifier from URL
            if '/datasets/' in config['url']:
                dataset_id = config['url'].split('/datasets/')[1]
            elif '/competitions/' in config['url']:
                dataset_id = config['url'].split('/competitions/')[1]
            else:
                logger.error(f"Cannot parse Kaggle URL: {config['url']}")
                return False
            
            logger.info(f"Downloading Kaggle dataset: {dataset_id}")
            
            # Download dataset
            if '/competitions/' in config['url']:
                api.competition_download_files(dataset_id, path=str(dataset_dir))
            else:
                api.dataset_download_files(dataset_id, path=str(dataset_dir), unzip=True)
            
            logger.info(f"Successfully downloaded {disease} dataset from Kaggle")
            return True
            
        except Exception as e:
            logger.error(f"Kaggle download failed for {disease}: {e}")
            return False
    
    def _download_zenodo_dataset(self, disease: str, config: Dict, dataset_dir: Path) -> bool:
        """Download dataset from Zenodo."""
        try:
            # Extract record ID from URL
            record_id = config['url'].split('/')[-1]
            api_url = f"https://zenodo.org/api/records/{record_id}"
            
            # Get record metadata
            response = requests.get(api_url)
            response.raise_for_status()
            record = response.json()
            
            # Download files
            for file_info in record['files']:
                file_url = file_info['links']['self']
                filename = file_info['key']
                filepath = dataset_dir / filename
                
                logger.info(f"Downloading {filename}...")
                
                response = requests.get(file_url, stream=True)
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
                
                # Extract if zip file
                if filename.endswith('.zip'):
                    import zipfile
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(dataset_dir)
                    filepath.unlink()  # Remove zip file
            
            logger.info(f"Successfully downloaded {disease} dataset from Zenodo")
            return True
            
        except Exception as e:
            logger.error(f"Zenodo download failed for {disease}: {e}")
            return False
    
    def _download_direct_dataset(self, disease: str, config: Dict, dataset_dir: Path) -> bool:
        """Download dataset from direct URL."""
        try:
            response = requests.get(config['url'], stream=True)
            response.raise_for_status()
            
            filename = config['url'].split('/')[-1]
            if not filename or '.' not in filename:
                filename = f"{disease}_dataset.zip"
            
            filepath = dataset_dir / filename
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
            
            # Extract if archive
            if filename.endswith(('.zip', '.tar.gz', '.tar')):
                if filename.endswith('.zip'):
                    import zipfile
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(dataset_dir)
                else:
                    import tarfile
                    with tarfile.open(filepath, 'r') as tar_ref:
                        tar_ref.extractall(dataset_dir)
                
                filepath.unlink()  # Remove archive
            
            logger.info(f"Successfully downloaded {disease} dataset")
            return True
            
        except Exception as e:
            logger.error(f"Direct download failed for {disease}: {e}")
            return False
    
    def validate_dataset(self, disease: str) -> Dict:
        """Validate dataset quality and structure."""
        if disease not in self.dataset_configs:
            logger.error(f"Unknown disease: {disease}")
            return {}
        
        config = self.dataset_configs[disease]
        dataset_dir = self.data_root / disease
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return {}
        
        logger.info(f"Validating {disease} dataset...")
        
        validation_results = {
            'disease': disease,
            'dataset_path': str(dataset_dir),
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'classes_found': {},
            'quality_issues': [],
            'recommendations': []
        }
        
        # Find all image files
        image_files = []
        for ext in self.quality_thresholds['supported_formats']:
            image_files.extend(dataset_dir.rglob(f"*{ext}"))
            image_files.extend(dataset_dir.rglob(f"*{ext.upper()}"))
        
        validation_results['total_images'] = len(image_files)
        
        # Validate each image
        for image_path in tqdm(image_files, desc=f"Validating {disease} images"):
            try:
                # Check file size
                file_size = image_path.stat().st_size
                if file_size < self.quality_thresholds['min_file_size']:
                    validation_results['invalid_images'] += 1
                    validation_results['quality_issues'].append(f"File too small: {image_path}")
                    continue
                
                if file_size > self.quality_thresholds['max_file_size']:
                    validation_results['invalid_images'] += 1
                    validation_results['quality_issues'].append(f"File too large: {image_path}")
                    continue
                
                # Check image properties
                with Image.open(image_path) as img:
                    width, height = img.size
                    
                    # Check image size
                    min_w, min_h = self.quality_thresholds['min_image_size']
                    max_w, max_h = self.quality_thresholds['max_image_size']
                    
                    if width < min_w or height < min_h:
                        validation_results['invalid_images'] += 1
                        validation_results['quality_issues'].append(f"Image too small: {image_path} ({width}x{height})")
                        continue
                    
                    if width > max_w or height > max_h:
                        validation_results['invalid_images'] += 1
                        validation_results['quality_issues'].append(f"Image too large: {image_path} ({width}x{height})")
                        continue
                    
                    # Check image mode
                    if img.mode not in ['RGB', 'L']:
                        validation_results['invalid_images'] += 1
                        validation_results['quality_issues'].append(f"Unsupported image mode: {image_path} ({img.mode})")
                        continue
                
                # Extract class from path
                class_name = self._extract_class_from_path(image_path, config['classes'])
                if class_name:
                    if class_name not in validation_results['classes_found']:
                        validation_results['classes_found'][class_name] = 0
                    validation_results['classes_found'][class_name] += 1
                
                validation_results['valid_images'] += 1
                
            except Exception as e:
                validation_results['invalid_images'] += 1
                validation_results['quality_issues'].append(f"Error processing {image_path}: {e}")
        
        # Generate recommendations
        self._generate_validation_recommendations(validation_results, config)
        
        # Save validation report
        report_file = dataset_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation complete: {validation_results['valid_images']}/{validation_results['total_images']} valid images")
        return validation_results
    
    def _extract_class_from_path(self, image_path: Path, expected_classes: List[str]) -> Optional[str]:
        """Extract class label from image path."""
        path_parts = image_path.parts
        
        # Check each part of the path for class names
        for part in path_parts:
            part_lower = part.lower()
            for class_name in expected_classes:
                if class_name.lower() in part_lower:
                    return class_name
        
        return None
    
    def _generate_validation_recommendations(self, results: Dict, config: Dict):
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check class balance
        classes_found = results['classes_found']
        if len(classes_found) < len(config['classes']):
            missing_classes = set(config['classes']) - set(classes_found.keys())
            recommendations.append(f"Missing classes: {missing_classes}")
        
        # Check sample counts
        min_samples = self.quality_thresholds['min_samples_per_class']
        for class_name, count in classes_found.items():
            if count < min_samples:
                recommendations.append(f"Class '{class_name}' has only {count} samples (minimum: {min_samples})")
        
        # Check overall quality
        valid_ratio = results['valid_images'] / results['total_images'] if results['total_images'] > 0 else 0
        if valid_ratio < 0.9:
            recommendations.append(f"Low quality ratio: {valid_ratio:.2%} valid images")
        
        # Check target sample count
        if results['valid_images'] < config['target_samples']:
            recommendations.append(f"Below target sample count: {results['valid_images']}/{config['target_samples']}")
        
        results['recommendations'] = recommendations
    
    def prepare_training_data(self, disease: str, train_split: float = 0.8, val_split: float = 0.1) -> bool:
        """Prepare dataset for training with train/val/test splits."""
        if disease not in self.dataset_configs:
            logger.error(f"Unknown disease: {disease}")
            return False
        
        config = self.dataset_configs[disease]
        dataset_dir = self.data_root / disease
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return False
        
        logger.info(f"Preparing {disease} training data...")
        
        # Create output directories
        output_dir = dataset_dir / "prepared"
        for split in ['train', 'val', 'test']:
            for class_name in config['classes']:
                (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Collect all valid images by class
        class_images = {class_name: [] for class_name in config['classes']}
        
        for ext in self.quality_thresholds['supported_formats']:
            for image_path in dataset_dir.rglob(f"*{ext}"):
                if 'prepared' in str(image_path):  # Skip already prepared data
                    continue
                
                class_name = self._extract_class_from_path(image_path, config['classes'])
                if class_name and self._is_valid_image(image_path):
                    class_images[class_name].append(image_path)
        
        # Split data for each class
        np.random.seed(42)  # For reproducible splits
        
        split_info = {'train': {}, 'val': {}, 'test': {}}
        
        for class_name, images in class_images.items():
            if not images:
                logger.warning(f"No images found for class: {class_name}")
                continue
            
            # Shuffle images
            np.random.shuffle(images)
            
            # Calculate split indices
            n_images = len(images)
            n_train = int(n_images * train_split)
            n_val = int(n_images * val_split)
            
            # Split images
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Copy images to appropriate directories
            for split, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
                split_info[split][class_name] = len(split_images)
                
                for i, src_path in enumerate(tqdm(split_images, desc=f"Copying {class_name} {split}")):
                    dst_path = output_dir / split / class_name / f"{class_name}_{split}_{i:06d}{src_path.suffix}"
                    shutil.copy2(src_path, dst_path)
        
        # Save split information
        split_file = output_dir / "split_info.json"
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        # Create dataset metadata
        metadata = {
            'disease': disease,
            'dataset_name': config['name'],
            'classes': config['classes'],
            'splits': split_info,
            'total_samples': sum(sum(class_counts.values()) for class_counts in split_info.values()),
            'image_size': config['image_size'],
            'created': datetime.now().isoformat()
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training data prepared for {disease}: {output_dir}")
        return True
    
    def _is_valid_image(self, image_path: Path) -> bool:
        """Check if image meets quality thresholds."""
        try:
            # Check file size
            file_size = image_path.stat().st_size
            if file_size < self.quality_thresholds['min_file_size'] or file_size > self.quality_thresholds['max_file_size']:
                return False
            
            # Check image properties
            with Image.open(image_path) as img:
                width, height = img.size
                min_w, min_h = self.quality_thresholds['min_image_size']
                max_w, max_h = self.quality_thresholds['max_image_size']
                
                if width < min_w or height < min_h or width > max_w or height > max_h:
                    return False
                
                if img.mode not in ['RGB', 'L']:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def generate_collection_report(self) -> str:
        """Generate comprehensive dataset collection report."""
        report = f"""
# Multi-Disease Dataset Collection Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Status Overview
"""
        
        total_diseases = len(self.dataset_configs)
        collected_diseases = 0
        
        for disease, config in self.dataset_configs.items():
            dataset_dir = self.data_root / disease
            status = "✅ Collected" if dataset_dir.exists() else "⏳ Pending"
            if dataset_dir.exists():
                collected_diseases += 1
            
            report += f"""
### {disease.title()} Cancer
- **Dataset**: {config['name']}
- **Status**: {status}
- **Classes**: {len(config['classes'])} ({', '.join(config['classes'])})
- **Target Samples**: {config['target_samples']:,}
- **Image Size**: {config['image_size']}
- **Source**: {config['url']}
"""
        
        report += f"""

## Collection Progress

- **Total Diseases**: {total_diseases}
- **Collected**: {collected_diseases}
- **Remaining**: {total_diseases - collected_diseases}
- **Progress**: {collected_diseases/total_diseases:.1%}

## Quality Thresholds

- **Image Size Range**: {self.quality_thresholds['min_image_size']} - {self.quality_thresholds['max_image_size']}
- **File Size Range**: {self.quality_thresholds['min_file_size']:,} - {self.quality_thresholds['max_file_size']:,} bytes
- **Supported Formats**: {', '.join(self.quality_thresholds['supported_formats'])}
- **Min Samples per Class**: {self.quality_thresholds['min_samples_per_class']}

## Next Steps

1. **Complete Data Collection**
   - Download remaining datasets
   - Validate data quality
   - Prepare training splits

2. **Data Partnerships**
   - Contact pathology labs for additional data
   - Negotiate data sharing agreements
   - Ensure HIPAA compliance

3. **Training Pipeline Integration**
   - Update foundation model for new diseases
   - Implement multi-disease training
   - Validate cross-disease transfer learning

## Commands

```bash
# Download all datasets
python scripts/multi_disease_dataset_collector.py --action download --disease all

# Validate specific dataset
python scripts/multi_disease_dataset_collector.py --action validate --disease lung

# Prepare training data
python scripts/multi_disease_dataset_collector.py --action prepare --disease lung
```
"""
        
        return report.strip()


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Multi-Disease Dataset Collector")
    parser.add_argument(
        '--action',
        choices=['download', 'validate', 'prepare', 'report'],
        required=True,
        help='Action to perform'
    )
    parser.add_argument(
        '--disease',
        choices=list(MultiDiseaseDatasetCollector({}).dataset_configs.keys()) + ['all'],
        help='Disease dataset to process'
    )
    parser.add_argument(
        '--data-root',
        help='Root directory for datasets'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if exists'
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
    
    # Initialize collector
    collector = MultiDiseaseDatasetCollector(args.data_root)
    
    if args.action == 'download':
        if not args.disease:
            print("Error: --disease required for download action")
            return
        
        if args.disease == 'all':
            diseases = list(collector.dataset_configs.keys())
        else:
            diseases = [args.disease]
        
        for disease in diseases:
            print(f"Downloading {disease} dataset...")
            success = collector.download_dataset(disease, args.force)
            if success:
                print(f"✅ {disease} dataset downloaded successfully")
            else:
                print(f"❌ Failed to download {disease} dataset")
    
    elif args.action == 'validate':
        if not args.disease:
            print("Error: --disease required for validate action")
            return
        
        if args.disease == 'all':
            diseases = list(collector.dataset_configs.keys())
        else:
            diseases = [args.disease]
        
        for disease in diseases:
            print(f"Validating {disease} dataset...")
            results = collector.validate_dataset(disease)
            if results:
                print(f"✅ {disease}: {results['valid_images']}/{results['total_images']} valid images")
                if results['quality_issues']:
                    print(f"⚠️  {len(results['quality_issues'])} quality issues found")
            else:
                print(f"❌ Failed to validate {disease} dataset")
    
    elif args.action == 'prepare':
        if not args.disease:
            print("Error: --disease required for prepare action")
            return
        
        if args.disease == 'all':
            diseases = list(collector.dataset_configs.keys())
        else:
            diseases = [args.disease]
        
        for disease in diseases:
            print(f"Preparing {disease} training data...")
            success = collector.prepare_training_data(disease)
            if success:
                print(f"✅ {disease} training data prepared")
            else:
                print(f"❌ Failed to prepare {disease} training data")
    
    elif args.action == 'report':
        print("Generating collection report...")
        report = collector.generate_collection_report()
        
        # Save report
        report_file = collector.data_root / f"collection_report_{datetime.now().strftime('%Y%m%d')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_file}")
        print("\n" + report)


if __name__ == "__main__":
    main()