"""
Calibration Dataset Management for Quantization

Implements calibration dataset creation and management for post-training
quantization (PTQ) to ensure representative data distribution.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import random
from collections import defaultdict

from ..utils.model_utils import get_model_size, count_parameters


@dataclass
class CalibrationConfig:
    """Configuration for calibration dataset"""
    num_samples: int = 1000                # Number of calibration samples
    sampling_strategy: str = 'random'       # 'random', 'stratified', 'diverse', 'representative'
    batch_size: int = 32                   # Calibration batch size
    num_batches: int = None                # Number of batches (overrides num_samples if set)
    shuffle: bool = True                   # Shuffle calibration data
    seed: int = 42                         # Random seed for reproducibility
    balance_classes: bool = True           # Balance classes in calibration set
    diversity_metric: str = 'feature'      # 'feature', 'prediction', 'uncertainty'
    min_samples_per_class: int = 10        # Minimum samples per class
    
    def __post_init__(self):
        if self.num_batches is not None:
            self.num_samples = self.num_batches * self.batch_size


@dataclass
class CalibrationStats:
    """Statistics about calibration dataset"""
    total_samples: int
    num_classes: int
    class_distribution: Dict[int, int]
    mean_activation: float
    std_activation: float
    min_activation: float
    max_activation: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_samples': self.total_samples,
            'num_classes': self.num_classes,
            'class_distribution': self.class_distribution,
            'mean_activation': self.mean_activation,
            'std_activation': self.std_activation,
            'min_activation': self.min_activation,
            'max_activation': self.max_activation
        }


class CalibrationDatasetBuilder:
    """
    Calibration dataset builder for quantization
    
    Creates representative calibration datasets using various strategies:
    - Random sampling: Simple random selection
    - Stratified sampling: Balanced class distribution
    - Diverse sampling: Maximize feature diversity
    - Representative sampling: Match original distribution
    
    Key considerations:
    - Representative of full dataset distribution
    - Sufficient samples for accurate statistics
    - Balanced across classes for fairness
    - Diverse to cover edge cases
    """
    
    def __init__(self, config: CalibrationConfig):
        """Initialize calibration dataset builder"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set random seed for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Calibration state
        self.calibration_indices = []
        self.class_distribution = {}
        
    def build_calibration_dataset(self, dataset: Dataset,
                                 model: nn.Module = None,
                                 device: torch.device = None) -> Tuple[Dataset, CalibrationStats]:
        """
        Build calibration dataset from full dataset
        
        Args:
            dataset: Full dataset
            model: Model for feature extraction (optional, for diverse sampling)
            device: Device for computation (optional)
            
        Returns:
            Tuple of (calibration_dataset, statistics)
        """
        try:
            if self.config.sampling_strategy == 'random':
                indices = self._random_sampling(dataset)
            elif self.config.sampling_strategy == 'stratified':
                indices = self._stratified_sampling(dataset)
            elif self.config.sampling_strategy == 'diverse':
                indices = self._diverse_sampling(dataset, model, device)
            elif self.config.sampling_strategy == 'representative':
                indices = self._representative_sampling(dataset)
            else:
                raise ValueError(f"Unknown sampling strategy: {self.config.sampling_strategy}")
            
            self.calibration_indices = indices
            
            # Create calibration subset
            calibration_dataset = Subset(dataset, indices)
            
            # Compute statistics
            stats = self._compute_statistics(calibration_dataset, dataset)
            
            self.logger.info(f"Built calibration dataset - "
                           f"Strategy: {self.config.sampling_strategy}, "
                           f"Samples: {len(indices)}, "
                           f"Classes: {len(stats.class_distribution)}")
            
            return calibration_dataset, stats
            
        except Exception as e:
            self.logger.error(f"Error building calibration dataset: {e}")
            raise
    
    def _random_sampling(self, dataset: Dataset) -> List[int]:
        """Random sampling strategy"""
        try:
            dataset_size = len(dataset)
            num_samples = min(self.config.num_samples, dataset_size)
            
            indices = random.sample(range(dataset_size), num_samples)
            
            self.logger.info(f"Random sampling: {num_samples} samples")
            return indices
            
        except Exception as e:
            self.logger.error(f"Error in random sampling: {e}")
            raise
    
    def _stratified_sampling(self, dataset: Dataset) -> List[int]:
        """Stratified sampling to balance classes"""
        try:
            # Group indices by class
            class_indices = defaultdict(list)
            
            for idx in range(len(dataset)):
                try:
                    _, label = dataset[idx]
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    class_indices[label].append(idx)
                except Exception as e:
                    self.logger.warning(f"Error accessing sample {idx}: {e}")
                    continue
            
            num_classes = len(class_indices)
            
            if num_classes == 0:
                self.logger.warning("No classes found, falling back to random sampling")
                return self._random_sampling(dataset)
            
            # Calculate samples per class
            if self.config.balance_classes:
                samples_per_class = self.config.num_samples // num_classes
                samples_per_class = max(samples_per_class, self.config.min_samples_per_class)
            else:
                # Proportional to class distribution
                total_samples = sum(len(indices) for indices in class_indices.values())
                samples_per_class = {}
                for label, indices in class_indices.items():
                    proportion = len(indices) / total_samples
                    samples_per_class[label] = max(
                        int(self.config.num_samples * proportion),
                        self.config.min_samples_per_class
                    )
            
            # Sample from each class
            selected_indices = []
            for label, indices in class_indices.items():
                if self.config.balance_classes:
                    n_samples = min(samples_per_class, len(indices))
                else:
                    n_samples = min(samples_per_class[label], len(indices))
                
                selected = random.sample(indices, n_samples)
                selected_indices.extend(selected)
            
            # Shuffle combined indices
            random.shuffle(selected_indices)
            
            # Trim to exact number if needed
            if len(selected_indices) > self.config.num_samples:
                selected_indices = selected_indices[:self.config.num_samples]
            
            self.logger.info(f"Stratified sampling: {len(selected_indices)} samples, "
                           f"{num_classes} classes")
            return selected_indices
            
        except Exception as e:
            self.logger.error(f"Error in stratified sampling: {e}")
            raise
    
    def _diverse_sampling(self, dataset: Dataset,
                         model: nn.Module,
                         device: torch.device) -> List[int]:
        """Diverse sampling to maximize feature diversity"""
        try:
            if model is None or device is None:
                self.logger.warning("Model or device not provided, falling back to stratified sampling")
                return self._stratified_sampling(dataset)
            
            # Extract features for all samples
            self.logger.info("Extracting features for diversity sampling...")
            features = self._extract_features(dataset, model, device)
            
            # Select diverse samples using k-means++ initialization
            selected_indices = self._kmeans_plus_plus_sampling(features, self.config.num_samples)
            
            self.logger.info(f"Diverse sampling: {len(selected_indices)} samples")
            return selected_indices
            
        except Exception as e:
            self.logger.error(f"Error in diverse sampling: {e}")
            # Fallback to stratified sampling
            return self._stratified_sampling(dataset)
    
    def _representative_sampling(self, dataset: Dataset) -> List[int]:
        """Representative sampling to match original distribution"""
        try:
            # Group by class
            class_indices = defaultdict(list)
            
            for idx in range(len(dataset)):
                try:
                    _, label = dataset[idx]
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    class_indices[label].append(idx)
                except Exception:
                    continue
            
            # Calculate proportional samples per class
            total_samples = sum(len(indices) for indices in class_indices.values())
            selected_indices = []
            
            for label, indices in class_indices.items():
                proportion = len(indices) / total_samples
                n_samples = max(
                    int(self.config.num_samples * proportion),
                    self.config.min_samples_per_class
                )
                n_samples = min(n_samples, len(indices))
                
                selected = random.sample(indices, n_samples)
                selected_indices.extend(selected)
            
            # Shuffle and trim
            random.shuffle(selected_indices)
            if len(selected_indices) > self.config.num_samples:
                selected_indices = selected_indices[:self.config.num_samples]
            
            self.logger.info(f"Representative sampling: {len(selected_indices)} samples")
            return selected_indices
            
        except Exception as e:
            self.logger.error(f"Error in representative sampling: {e}")
            raise
    
    def _extract_features(self, dataset: Dataset,
                         model: nn.Module,
                         device: torch.device,
                         max_samples: int = 5000) -> np.ndarray:
        """Extract features from dataset using model"""
        try:
            model.eval()
            model = model.to(device)
            
            # Create temporary dataloader
            temp_loader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                num_workers=0
            )
            
            features_list = []
            sample_count = 0
            
            with torch.no_grad():
                for data, _ in temp_loader:
                    if sample_count >= max_samples:
                        break
                    
                    data = data.to(device)
                    
                    # Extract features from penultimate layer
                    # Assuming model has a feature extraction method or we use output
                    output = model(data)
                    
                    # Flatten features
                    features = output.view(output.size(0), -1)
                    features_list.append(features.cpu().numpy())
                    
                    sample_count += data.size(0)
            
            features = np.vstack(features_list)
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            raise
    
    def _kmeans_plus_plus_sampling(self, features: np.ndarray,
                                  num_samples: int) -> List[int]:
        """K-means++ initialization for diverse sampling"""
        try:
            n_features = features.shape[0]
            selected_indices = []
            
            # Select first sample randomly
            first_idx = random.randint(0, n_features - 1)
            selected_indices.append(first_idx)
            
            # Select remaining samples
            for _ in range(num_samples - 1):
                # Compute distances to nearest selected sample
                distances = np.full(n_features, np.inf)
                
                for idx in selected_indices:
                    dist = np.linalg.norm(features - features[idx], axis=1)
                    distances = np.minimum(distances, dist)
                
                # Select sample with maximum distance
                next_idx = np.argmax(distances)
                selected_indices.append(next_idx)
            
            return selected_indices
            
        except Exception as e:
            self.logger.error(f"Error in k-means++ sampling: {e}")
            raise
    
    def _compute_statistics(self, calibration_dataset: Dataset,
                          full_dataset: Dataset) -> CalibrationStats:
        """Compute statistics about calibration dataset"""
        try:
            # Count class distribution
            class_counts = defaultdict(int)
            activations = []
            
            for idx in range(len(calibration_dataset)):
                try:
                    data, label = calibration_dataset[idx]
                    
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    class_counts[label] += 1
                    
                    # Collect activation statistics
                    if isinstance(data, torch.Tensor):
                        activations.append(data.numpy().flatten())
                except Exception:
                    continue
            
            # Compute activation statistics
            if activations:
                all_activations = np.concatenate(activations)
                mean_act = float(np.mean(all_activations))
                std_act = float(np.std(all_activations))
                min_act = float(np.min(all_activations))
                max_act = float(np.max(all_activations))
            else:
                mean_act = std_act = min_act = max_act = 0.0
            
            stats = CalibrationStats(
                total_samples=len(calibration_dataset),
                num_classes=len(class_counts),
                class_distribution=dict(class_counts),
                mean_activation=mean_act,
                std_activation=std_act,
                min_activation=min_act,
                max_activation=max_act
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error computing statistics: {e}")
            raise
    
    def create_calibration_dataloader(self, calibration_dataset: Dataset) -> DataLoader:
        """Create dataloader for calibration"""
        try:
            dataloader = DataLoader(
                calibration_dataset,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle,
                num_workers=0,  # Single worker for calibration
                pin_memory=False
            )
            
            return dataloader
            
        except Exception as e:
            self.logger.error(f"Error creating calibration dataloader: {e}")
            raise
    
    def save_calibration_indices(self, save_path: Path):
        """Save calibration indices for reproducibility"""
        try:
            data = {
                'config': self.config.__dict__,
                'indices': self.calibration_indices,
                'class_distribution': self.class_distribution
            }
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Saved calibration indices to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving calibration indices: {e}")
            raise
    
    def load_calibration_indices(self, load_path: Path) -> List[int]:
        """Load calibration indices"""
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            self.calibration_indices = data['indices']
            self.class_distribution = data.get('class_distribution', {})
            
            self.logger.info(f"Loaded calibration indices from {load_path}")
            return self.calibration_indices
            
        except Exception as e:
            self.logger.error(f"Error loading calibration indices: {e}")
            raise


def create_calibration_config(num_samples: int = 1000,
                             strategy: str = 'stratified',
                             balance_classes: bool = True) -> CalibrationConfig:
    """Create calibration configuration"""
    return CalibrationConfig(
        num_samples=num_samples,
        sampling_strategy=strategy,
        batch_size=32,
        shuffle=True,
        balance_classes=balance_classes,
        min_samples_per_class=10
    )


def build_calibration_dataset(dataset: Dataset,
                             num_samples: int = 1000,
                             strategy: str = 'stratified') -> Tuple[Dataset, DataLoader, CalibrationStats]:
    """
    Build calibration dataset and dataloader
    
    Args:
        dataset: Full dataset
        num_samples: Number of calibration samples
        strategy: Sampling strategy
        
    Returns:
        Tuple of (calibration_dataset, calibration_loader, statistics)
    """
    config = CalibrationConfig(
        num_samples=num_samples,
        sampling_strategy=strategy,
        batch_size=32,
        shuffle=True,
        balance_classes=True
    )
    
    builder = CalibrationDatasetBuilder(config)
    calibration_dataset, stats = builder.build_calibration_dataset(dataset)
    calibration_loader = builder.create_calibration_dataloader(calibration_dataset)
    
    return calibration_dataset, calibration_loader, stats


# Example usage for medical AI
def create_medical_calibration_dataset(dataset: Dataset,
                                      num_samples: int = 2000) -> Tuple[Dataset, DataLoader, CalibrationStats]:
    """Create calibration dataset optimized for medical AI"""
    config = CalibrationConfig(
        num_samples=num_samples,
        sampling_strategy='stratified',  # Balanced for fairness
        batch_size=16,                   # Smaller batches for medical data
        shuffle=True,
        balance_classes=True,            # Critical for medical fairness
        min_samples_per_class=20,        # Higher minimum for rare diseases
        seed=42
    )
    
    builder = CalibrationDatasetBuilder(config)
    calibration_dataset, stats = builder.build_calibration_dataset(dataset)
    calibration_loader = builder.create_calibration_dataloader(calibration_dataset)
    
    return calibration_dataset, calibration_loader, stats
