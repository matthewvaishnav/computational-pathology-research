"""
Configuration management for nnMIL Multiple Instance Learning.

This module implements rule-based configuration with dataset fingerprinting
to automatically derive optimal hyperparameters based on dataset characteristics.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml


@dataclass
class nnMILConfig:
    """
    Configuration class for nnMIL with automatic dataset fingerprinting.
    
    Automatically derives optimal hyperparameters based on dataset characteristics
    including median patches per slide, class distribution, and task type.
    
    Attributes:
        # Model architecture
        feature_dim: Input feature dimension (auto-detected)
        hidden_dim: Hidden dimension for attention (default: 256)
        num_classes: Number of output classes (auto-detected)
        dropout: Dropout rate (default: 0.25)
        multi_scale: Enable multi-scale processing (default: False)
        fusion_type: Multi-scale fusion type ('early' or 'late')
        
        # Training configuration
        batch_size: Training batch size (default: 32)
        learning_rate: Learning rate (task-dependent defaults)
        weight_decay: L2 regularization (default: 1e-4)
        num_epochs: Maximum training epochs (default: 100)
        patience: Early stopping patience (default: 10)
        
        # Data configuration
        bag_length: Fixed bag length (auto-derived from dataset)
        sampler_type: Batch sampler type ('balanced', 'regression', 'survival')
        
        # Inference configuration
        window_size: Sliding window size (default: bag_length)
        stride: Sliding window stride (default: window_size // 4)
        enable_uncertainty: Enable uncertainty quantification (default: True)
        num_mc_samples: Monte Carlo samples for uncertainty (default: 10)
        
        # Task configuration
        task_type: Task type ('classification', 'regression', 'survival')
        class_weights: Class weights for imbalanced datasets
        
        # Logging and checkpointing
        log_interval: Logging interval in batches (default: 10)
        checkpoint_interval: Checkpoint interval in epochs (default: 5)
        save_best_only: Save only best model (default: True)
        
        # Dataset fingerprint (auto-computed)
        dataset_fingerprint: Dataset characteristics
    
    Example:
        >>> # Automatic configuration from dataset
        >>> config = nnMILConfig.from_dataset(
        ...     dataset_path="data/pcam",
        ...     task_type="classification"
        ... )
        >>> 
        >>> # Manual configuration
        >>> config = nnMILConfig(
        ...     feature_dim=1024,
        ...     num_classes=2,
        ...     bag_length=512,
        ...     task_type="classification"
        ... )
    """
    
    # Model architecture
    feature_dim: int = 1024
    hidden_dim: int = 256
    num_classes: int = 2
    dropout: float = 0.25
    multi_scale: bool = False
    fusion_type: str = 'early'
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    patience: int = 10
    use_amp: bool = True  # Mixed precision training
    
    # Data configuration
    bag_length: int = 512
    sampler_type: str = 'balanced'
    
    # Inference configuration
    window_size: Optional[int] = None
    stride: Optional[int] = None
    enable_uncertainty: bool = True
    num_mc_samples: int = 10
    
    # Task configuration
    task_type: str = 'classification'
    class_weights: Optional[List[float]] = None
    
    # Logging and checkpointing
    log_interval: int = 10
    checkpoint_interval: int = 5
    save_best_only: bool = True
    
    # Dataset fingerprint
    dataset_fingerprint: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and derive configuration after initialization."""
        self._validate_config()
        self._derive_dependent_params()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Model validation
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")
        
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        
        if self.fusion_type not in ['early', 'late']:
            raise ValueError(f"fusion_type must be 'early' or 'late', got {self.fusion_type}")
        
        # Training validation
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")
        
        # Data validation
        if self.bag_length <= 0:
            raise ValueError(f"bag_length must be positive, got {self.bag_length}")
        
        if self.sampler_type not in ['balanced', 'regression', 'survival']:
            raise ValueError(
                f"sampler_type must be 'balanced', 'regression', or 'survival', "
                f"got {self.sampler_type}"
            )
        
        # Inference validation
        if self.num_mc_samples <= 0:
            raise ValueError(f"num_mc_samples must be positive, got {self.num_mc_samples}")
        
        # Task validation
        if self.task_type not in ['classification', 'regression', 'survival']:
            raise ValueError(
                f"task_type must be 'classification', 'regression', or 'survival', "
                f"got {self.task_type}"
            )
        
        # Logging validation
        if self.log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {self.log_interval}")
        
        if self.checkpoint_interval <= 0:
            raise ValueError(f"checkpoint_interval must be positive, got {self.checkpoint_interval}")
    
    def _derive_dependent_params(self):
        """Derive dependent parameters from base configuration."""
        # Set window_size to bag_length if not specified
        if self.window_size is None:
            self.window_size = self.bag_length
        
        # Set stride to window_size // 4 if not specified (75% overlap)
        if self.stride is None:
            self.stride = max(1, self.window_size // 4)
        
        # Set task-specific defaults
        if self.task_type == 'classification':
            if self.sampler_type == 'regression':
                self.sampler_type = 'balanced'
        elif self.task_type == 'regression':
            if self.sampler_type == 'balanced':
                self.sampler_type = 'regression'
            # Lower learning rate for regression
            if self.learning_rate == 3e-4:  # Default classification LR
                self.learning_rate = 1e-4
        elif self.task_type == 'survival':
            if self.sampler_type in ['balanced', 'regression']:
                self.sampler_type = 'survival'
            # Lower learning rate for survival
            if self.learning_rate == 3e-4:  # Default classification LR
                self.learning_rate = 1e-4
    
    @classmethod
    def from_dataset(
        cls,
        dataset_path: Union[str, Path],
        task_type: str,
        feature_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        **kwargs
    ) -> 'nnMILConfig':
        """
        Create configuration from dataset fingerprinting.
        
        Analyzes dataset characteristics to automatically derive optimal
        hyperparameters including bag_length, learning_rate, and class_weights.
        
        Args:
            dataset_path: Path to dataset directory
            task_type: Task type ('classification', 'regression', 'survival')
            feature_dim: Feature dimension (auto-detected if None)
            num_classes: Number of classes (auto-detected if None)
            **kwargs: Additional configuration overrides
        
        Returns:
            nnMILConfig with dataset-derived parameters
        
        Example:
            >>> config = nnMILConfig.from_dataset(
            ...     dataset_path="data/pcam",
            ...     task_type="classification"
            ... )
            >>> print(f"Derived bag_length: {config.bag_length}")
        """
        dataset_path = Path(dataset_path)
        
        # Extract dataset fingerprint
        fingerprint = cls._extract_dataset_fingerprint(dataset_path, task_type)
        
        # Derive configuration from fingerprint
        config_params = cls._derive_config_from_fingerprint(
            fingerprint, task_type, feature_dim, num_classes
        )
        
        # Apply user overrides
        config_params.update(kwargs)
        config_params['dataset_fingerprint'] = fingerprint
        
        return cls(**config_params)
    
    @staticmethod
    def _extract_dataset_fingerprint(
        dataset_path: Path,
        task_type: str
    ) -> Dict[str, Any]:
        """
        Extract dataset characteristics for configuration derivation.
        
        Args:
            dataset_path: Path to dataset directory
            task_type: Task type for analysis
        
        Returns:
            Dictionary with dataset fingerprint:
            - median_patches: Median patches per slide
            - iqr_patches: IQR of patches per slide
            - num_slides: Total number of slides
            - class_distribution: Class prevalence (classification)
            - target_range: Target value range (regression)
            - event_rate: Event rate (survival)
        """
        fingerprint = {
            'dataset_path': str(dataset_path),
            'task_type': task_type,
            'median_patches': 512,  # Default fallback
            'iqr_patches': (256, 1024),
            'num_slides': 1000,
            'class_distribution': None,
            'target_range': None,
            'event_rate': None
        }
        
        try:
            # Try to analyze actual dataset
            if dataset_path.exists():
                fingerprint.update(
                    nnMILConfig._analyze_dataset_structure(dataset_path, task_type)
                )
            else:
                logging.warning(
                    f"Dataset path {dataset_path} does not exist. "
                    f"Using default fingerprint."
                )
        
        except Exception as e:
            logging.warning(
                f"Failed to analyze dataset {dataset_path}: {e}. "
                f"Using default fingerprint."
            )
        
        return fingerprint
    
    @staticmethod
    def _analyze_dataset_structure(
        dataset_path: Path,
        task_type: str
    ) -> Dict[str, Any]:
        """
        Analyze dataset structure to extract characteristics.
        
        This is a simplified implementation that would need to be
        customized based on actual dataset format.
        """
        analysis = {}
        
        # Count slides (simplified - assumes one file per slide)
        slide_files = list(dataset_path.glob("**/*.pt")) + list(dataset_path.glob("**/*.h5"))
        analysis['num_slides'] = len(slide_files)
        
        # Estimate patches per slide (would need actual analysis)
        # For now, use reasonable defaults based on common datasets
        if 'pcam' in str(dataset_path).lower():
            analysis['median_patches'] = 100  # PatchCamelyon has ~100 patches
            analysis['iqr_patches'] = (80, 120)
        elif 'camelyon' in str(dataset_path).lower():
            analysis['median_patches'] = 1000  # Camelyon16 has ~1000 patches
            analysis['iqr_patches'] = (500, 2000)
        else:
            analysis['median_patches'] = 512  # Default
            analysis['iqr_patches'] = (256, 1024)
        
        # Task-specific analysis
        if task_type == 'classification':
            # Estimate class distribution (simplified)
            analysis['class_distribution'] = [0.5, 0.5]  # Balanced default
        elif task_type == 'regression':
            analysis['target_range'] = (0.0, 1.0)  # Default range
        elif task_type == 'survival':
            analysis['event_rate'] = 0.3  # Default event rate
        
        return analysis
    
    @staticmethod
    def _derive_config_from_fingerprint(
        fingerprint: Dict[str, Any],
        task_type: str,
        feature_dim: Optional[int],
        num_classes: Optional[int]
    ) -> Dict[str, Any]:
        """
        Derive configuration parameters from dataset fingerprint.
        
        Args:
            fingerprint: Dataset characteristics
            task_type: Task type
            feature_dim: Feature dimension override
            num_classes: Number of classes override
        
        Returns:
            Dictionary with derived configuration parameters
        """
        config = {}
        
        # Derive bag_length from median patches
        median_patches = fingerprint.get('median_patches', 512)
        config['bag_length'] = max(100, min(10000, median_patches // 2))
        
        # Set task type
        config['task_type'] = task_type
        
        # Set feature dimension
        if feature_dim is not None:
            config['feature_dim'] = feature_dim
        else:
            # Common feature dimensions from foundation models
            config['feature_dim'] = 1024  # UNI default
        
        # Set number of classes
        if task_type == 'classification':
            if num_classes is not None:
                config['num_classes'] = num_classes
            else:
                class_dist = fingerprint.get('class_distribution', [0.5, 0.5])
                config['num_classes'] = len(class_dist)
            
            # Set class weights for imbalanced datasets
            class_dist = fingerprint.get('class_distribution')
            if class_dist and len(class_dist) > 1:
                # Inverse frequency weighting
                total = sum(class_dist)
                weights = [total / (len(class_dist) * freq) for freq in class_dist]
                config['class_weights'] = weights
        
        elif task_type in ['regression', 'survival']:
            config['num_classes'] = 1
        
        # Set learning rate based on task type
        if task_type == 'classification':
            config['learning_rate'] = 3e-4
        else:  # regression or survival
            config['learning_rate'] = 1e-4
        
        # Set sampler type based on task
        if task_type == 'classification':
            config['sampler_type'] = 'balanced'
        elif task_type == 'regression':
            config['sampler_type'] = 'regression'
        elif task_type == 'survival':
            config['sampler_type'] = 'survival'
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'nnMILConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
        
        Returns:
            nnMILConfig loaded from YAML
        
        Example:
            >>> config = nnMILConfig.from_yaml("configs/disease_subtyping.yaml")
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle inheritance
        if 'inherit_from' in config_dict:
            base_config = cls._load_base_config(config_dict['inherit_from'], yaml_path.parent)
            base_config.update(config_dict)
            config_dict = base_config
            del config_dict['inherit_from']
        
        return cls(**config_dict)
    
    @staticmethod
    def _load_base_config(inherit_path: str, config_dir: Path) -> Dict[str, Any]:
        """Load base configuration for inheritance."""
        base_path = config_dir / inherit_path
        
        if not base_path.exists():
            raise FileNotFoundError(f"Base configuration not found: {base_path}")
        
        with open(base_path, 'r') as f:
            return yaml.safe_load(f)
    
    def to_yaml(self, yaml_path: Union[str, Path]):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        
        Example:
            >>> config.to_yaml("configs/my_config.yaml")
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary, excluding non-serializable fields
        config_dict = {}
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                config_dict[key] = value
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    def log_config(self, logger: Optional[logging.Logger] = None):
        """
        Log configuration parameters.
        
        Args:
            logger: Logger instance (creates new if None)
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        
        logger.info("nnMIL Configuration:")
        logger.info("=" * 50)
        
        # Group parameters by category
        categories = {
            'Model': ['feature_dim', 'hidden_dim', 'num_classes', 'dropout', 'multi_scale', 'fusion_type'],
            'Training': ['batch_size', 'learning_rate', 'weight_decay', 'num_epochs', 'patience'],
            'Data': ['bag_length', 'sampler_type'],
            'Inference': ['window_size', 'stride', 'enable_uncertainty', 'num_mc_samples'],
            'Task': ['task_type', 'class_weights'],
            'Logging': ['log_interval', 'checkpoint_interval', 'save_best_only']
        }
        
        for category, params in categories.items():
            logger.info(f"{category}:")
            for param in params:
                if hasattr(self, param):
                    value = getattr(self, param)
                    logger.info(f"  {param}: {value}")
            logger.info("")
        
        # Log dataset fingerprint if available
        if self.dataset_fingerprint:
            logger.info("Dataset Fingerprint:")
            for key, value in self.dataset_fingerprint.items():
                logger.info(f"  {key}: {value}")
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get parameters for model initialization."""
        return {
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'multi_scale': self.multi_scale,
            'fusion_type': self.fusion_type
        }
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get parameters for training setup."""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'sampler_type': self.sampler_type,
            'class_weights': self.class_weights
        }
    
    def get_inference_params(self) -> Dict[str, Any]:
        """Get parameters for inference setup."""
        return {
            'window_size': self.window_size,
            'stride': self.stride,
            'enable_uncertainty': self.enable_uncertainty,
            'num_mc_samples': self.num_mc_samples
        }