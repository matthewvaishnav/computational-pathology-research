"""
Dataset Manager for Research Platform

Organize, catalog, and manage medical imaging datasets.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import logging


@dataclass
class DatasetMetadata:
    """Dataset metadata"""
    id: str
    name: str
    description: str
    dataset_type: str  # 'wsi', 'patch', 'annotation'
    disease_types: List[str]
    num_samples: int
    size_gb: float
    created_at: str
    updated_at: str
    version: str
    tags: List[str]
    source: str
    license: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DatasetMetadata':
        return cls(**data)


@dataclass
class DatasetStats:
    """Dataset statistics"""
    total_samples: int
    disease_distribution: Dict[str, int]
    quality_scores: Dict[str, float]
    size_distribution: Dict[str, int]
    annotation_coverage: float


class DatasetOrganizer:
    """
    Dataset organization interface
    
    Features:
    - Hierarchical organization
    - Metadata management
    - Search and filtering
    - Version tracking
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / 'datasets'
        self.metadata_dir = self.base_dir / 'metadata'
        self.index_file = self.metadata_dir / 'index.json'
        
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Load index
        self.index = self._load_index()
    
    def create_dataset(self, name: str, description: str,
                      dataset_type: str, disease_types: List[str],
                      source: str = '', license: str = 'Research Only') -> str:
        """
        Create new dataset
        
        Args:
            name: Dataset name
            description: Description
            dataset_type: Type (wsi, patch, annotation)
            disease_types: List of diseases
            source: Data source
            license: License
            
        Returns:
            Dataset ID
        """
        
        # Generate ID
        dataset_id = self._generate_id(name)
        
        # Create metadata
        metadata = DatasetMetadata(
            id=dataset_id,
            name=name,
            description=description,
            dataset_type=dataset_type,
            disease_types=disease_types,
            num_samples=0,
            size_gb=0.0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            version='1.0.0',
            tags=[],
            source=source,
            license=license
        )
        
        # Create dataset directory
        dataset_dir = self.datasets_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        self._save_metadata(dataset_id, metadata)
        
        # Update index
        self.index[dataset_id] = metadata.to_dict()
        self._save_index()
        
        self.logger.info(f"Created dataset: {name} ({dataset_id})")
        
        return dataset_id
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata"""
        
        if dataset_id not in self.index:
            return None
        
        return DatasetMetadata.from_dict(self.index[dataset_id])
    
    def list_datasets(self, dataset_type: Optional[str] = None,
                     disease_type: Optional[str] = None,
                     tags: Optional[List[str]] = None) -> List[DatasetMetadata]:
        """
        List datasets with filters
        
        Args:
            dataset_type: Filter by type
            disease_type: Filter by disease
            tags: Filter by tags
            
        Returns:
            List of datasets
        """
        
        datasets = []
        
        for dataset_id, data in self.index.items():
            metadata = DatasetMetadata.from_dict(data)
            
            # Apply filters
            if dataset_type and metadata.dataset_type != dataset_type:
                continue
            
            if disease_type and disease_type not in metadata.disease_types:
                continue
            
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            
            datasets.append(metadata)
        
        return datasets
    
    def update_dataset(self, dataset_id: str, **kwargs) -> bool:
        """Update dataset metadata"""
        
        if dataset_id not in self.index:
            return False
        
        metadata = self.get_dataset(dataset_id)
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        metadata.updated_at = datetime.now().isoformat()
        
        # Save
        self._save_metadata(dataset_id, metadata)
        self.index[dataset_id] = metadata.to_dict()
        self._save_index()
        
        return True
    
    def delete_dataset(self, dataset_id: str, remove_files: bool = False) -> bool:
        """Delete dataset"""
        
        if dataset_id not in self.index:
            return False
        
        # Remove files
        if remove_files:
            dataset_dir = self.datasets_dir / dataset_id
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
        
        # Remove metadata
        metadata_file = self.metadata_dir / f'{dataset_id}.json'
        if metadata_file.exists():
            metadata_file.unlink()
        
        # Update index
        del self.index[dataset_id]
        self._save_index()
        
        self.logger.info(f"Deleted dataset: {dataset_id}")
        
        return True
    
    def add_samples(self, dataset_id: str, sample_paths: List[Path]) -> int:
        """Add samples to dataset"""
        
        if dataset_id not in self.index:
            return 0
        
        dataset_dir = self.datasets_dir / dataset_id
        samples_dir = dataset_dir / 'samples'
        samples_dir.mkdir(exist_ok=True)
        
        added = 0
        for sample_path in sample_paths:
            if not sample_path.exists():
                continue
            
            # Copy to dataset
            dest = samples_dir / sample_path.name
            shutil.copy2(sample_path, dest)
            added += 1
        
        # Update metadata
        metadata = self.get_dataset(dataset_id)
        metadata.num_samples += added
        metadata.size_gb = self._calculate_size(dataset_dir)
        
        self.update_dataset(dataset_id, 
                          num_samples=metadata.num_samples,
                          size_gb=metadata.size_gb)
        
        return added
    
    def get_dataset_path(self, dataset_id: str) -> Optional[Path]:
        """Get dataset directory path"""
        
        if dataset_id not in self.index:
            return None
        
        return self.datasets_dir / dataset_id
    
    def get_stats(self, dataset_id: str) -> Optional[DatasetStats]:
        """Get dataset statistics"""
        
        if dataset_id not in self.index:
            return None
        
        metadata = self.get_dataset(dataset_id)
        
        # Calculate stats
        stats = DatasetStats(
            total_samples=metadata.num_samples,
            disease_distribution={},
            quality_scores={},
            size_distribution={},
            annotation_coverage=0.0
        )
        
        # Calculate detailed statistics from samples
        if samples:
            stats = self._calculate_detailed_stats(samples, dataset_id)
        
        return stats
    
    def _calculate_detailed_stats(self, samples: List[Dict], dataset_id: str) -> DatasetStatistics:
        """Calculate comprehensive dataset statistics."""
        import numpy as np
        from collections import Counter, defaultdict
        
        # Initialize counters
        disease_counter = Counter()
        quality_scores = []
        size_distribution = defaultdict(int)
        annotation_count = 0
        total_samples = len(samples)
        
        # Image statistics
        image_sizes = []
        file_sizes = []
        
        # Quality metrics
        blur_scores = []
        contrast_scores = []
        brightness_scores = []
        
        for sample in samples:
            # Disease distribution
            disease = sample.get('disease', 'unknown')
            disease_counter[disease] += 1
            
            # Quality scores
            quality = sample.get('quality_score', 0.0)
            quality_scores.append(quality)
            
            # Size distribution (in MB)
            file_size = sample.get('file_size_bytes', 0) / (1024 * 1024)
            if file_size < 10:
                size_distribution['small'] += 1
            elif file_size < 100:
                size_distribution['medium'] += 1
            else:
                size_distribution['large'] += 1
            
            file_sizes.append(file_size)
            
            # Image dimensions
            width = sample.get('width', 0)
            height = sample.get('height', 0)
            if width > 0 and height > 0:
                image_sizes.append((width, height))
            
            # Annotation coverage
            if sample.get('has_annotations', False):
                annotation_count += 1
            
            # Image quality metrics
            blur_scores.append(sample.get('blur_score', 0.0))
            contrast_scores.append(sample.get('contrast_score', 0.0))
            brightness_scores.append(sample.get('brightness_score', 0.0))
        
        # Calculate statistics
        disease_distribution = {
            disease: count / total_samples 
            for disease, count in disease_counter.items()
        }
        
        quality_stats = {
            'mean': float(np.mean(quality_scores)) if quality_scores else 0.0,
            'std': float(np.std(quality_scores)) if quality_scores else 0.0,
            'min': float(np.min(quality_scores)) if quality_scores else 0.0,
            'max': float(np.max(quality_scores)) if quality_scores else 0.0,
            'median': float(np.median(quality_scores)) if quality_scores else 0.0,
            'q25': float(np.percentile(quality_scores, 25)) if quality_scores else 0.0,
            'q75': float(np.percentile(quality_scores, 75)) if quality_scores else 0.0
        }
        
        size_stats = {
            'small_percent': size_distribution['small'] / total_samples * 100,
            'medium_percent': size_distribution['medium'] / total_samples * 100,
            'large_percent': size_distribution['large'] / total_samples * 100,
            'mean_size_mb': float(np.mean(file_sizes)) if file_sizes else 0.0,
            'total_size_gb': sum(file_sizes) / 1024 if file_sizes else 0.0
        }
        
        # Image dimension statistics
        if image_sizes:
            widths, heights = zip(*image_sizes)
            dimension_stats = {
                'mean_width': float(np.mean(widths)),
                'mean_height': float(np.mean(heights)),
                'min_width': int(np.min(widths)),
                'max_width': int(np.max(widths)),
                'min_height': int(np.min(heights)),
                'max_height': int(np.max(heights)),
                'aspect_ratio_mean': float(np.mean([w/h for w, h in image_sizes if h > 0]))
            }
        else:
            dimension_stats = {}
        
        # Quality distribution
        quality_distribution = {
            'excellent': len([q for q in quality_scores if q >= 0.9]) / total_samples * 100,
            'good': len([q for q in quality_scores if 0.7 <= q < 0.9]) / total_samples * 100,
            'fair': len([q for q in quality_scores if 0.5 <= q < 0.7]) / total_samples * 100,
            'poor': len([q for q in quality_scores if q < 0.5]) / total_samples * 100
        }
        
        # Advanced quality metrics
        advanced_quality = {
            'blur_stats': {
                'mean': float(np.mean(blur_scores)) if blur_scores else 0.0,
                'std': float(np.std(blur_scores)) if blur_scores else 0.0
            },
            'contrast_stats': {
                'mean': float(np.mean(contrast_scores)) if contrast_scores else 0.0,
                'std': float(np.std(contrast_scores)) if contrast_scores else 0.0
            },
            'brightness_stats': {
                'mean': float(np.mean(brightness_scores)) if brightness_scores else 0.0,
                'std': float(np.std(brightness_scores)) if brightness_scores else 0.0
            }
        }
        
        return DatasetStatistics(
            total_samples=total_samples,
            disease_distribution=disease_distribution,
            quality_scores=quality_stats,
            size_distribution=size_stats,
            annotation_coverage=annotation_count / total_samples * 100,
            dimension_stats=dimension_stats,
            quality_distribution=quality_distribution,
            advanced_quality=advanced_quality
        )
    
    def search_datasets(self, query: str) -> List[DatasetMetadata]:
        """Search datasets by name/description"""
        
        results = []
        query_lower = query.lower()
        
        for dataset_id, data in self.index.items():
            metadata = DatasetMetadata.from_dict(data)
            
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.description.lower()):
                results.append(metadata)
        
        return results
    
    def export_metadata(self, dataset_id: str, output_path: Path) -> bool:
        """Export dataset metadata"""
        
        metadata = self.get_dataset(dataset_id)
        if not metadata:
            return False
        
        with open(output_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        return True
    
    def import_metadata(self, metadata_path: Path) -> Optional[str]:
        """Import dataset metadata"""
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        metadata = DatasetMetadata.from_dict(data)
        
        # Add to index
        self.index[metadata.id] = metadata.to_dict()
        self._save_index()
        
        # Save metadata
        self._save_metadata(metadata.id, metadata)
        
        return metadata.id
    
    # Private methods
    
    def _generate_id(self, name: str) -> str:
        """Generate dataset ID"""
        
        timestamp = datetime.now().isoformat()
        hash_input = f"{name}_{timestamp}"
        hash_obj = hashlib.md5(hash_input.encode())
        
        return hash_obj.hexdigest()[:12]
    
    def _load_index(self) -> Dict:
        """Load dataset index"""
        
        if not self.index_file.exists():
            return {}
        
        with open(self.index_file, 'r') as f:
            return json.load(f)
    
    def _save_index(self):
        """Save dataset index"""
        
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _save_metadata(self, dataset_id: str, metadata: DatasetMetadata):
        """Save dataset metadata"""
        
        metadata_file = self.metadata_dir / f'{dataset_id}.json'
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def _calculate_size(self, directory: Path) -> float:
        """Calculate directory size in GB"""
        
        total_size = 0
        for file in directory.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
        
        return total_size / (1024 ** 3)


# Convenience functions

def create_dataset_organizer(base_dir: str = 'data/research') -> DatasetOrganizer:
    """Create dataset organizer"""
    return DatasetOrganizer(Path(base_dir))


def organize_dataset(organizer: DatasetOrganizer,
                    name: str,
                    sample_paths: List[Path],
                    **kwargs) -> str:
    """Organize dataset with samples"""
    
    dataset_id = organizer.create_dataset(name, **kwargs)
    organizer.add_samples(dataset_id, sample_paths)
    
    return dataset_id
