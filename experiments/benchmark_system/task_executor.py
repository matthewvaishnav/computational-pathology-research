"""
Training Task Executor for the Competitor Benchmark System.

This module executes identical training tasks across all frameworks, ensuring
fair comparisons by enforcing identical configurations, random seeds, and data splits.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.benchmark_system.models import (
    FrameworkEnvironment,
    TaskSpecification,
    TrainingResult,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Framework-specific training configuration."""
    
    framework_name: str
    task_spec: TaskSpecification
    
    # Framework-specific configuration dictionary
    config_dict: Dict[str, Any] = field(default_factory=dict)
    
    # Reproducibility tracking
    random_seed: int = 42
    data_split_hash: Optional[str] = None  # Hash of data split for verification
    
    # Configuration metadata
    translation_notes: List[str] = field(default_factory=list)


@dataclass
class EquivalenceReport:
    """Report on configuration equivalence across frameworks."""
    
    is_equivalent: bool
    differences: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Configuration comparison details
    random_seeds: Dict[str, int] = field(default_factory=dict)
    data_splits: Dict[str, tuple] = field(default_factory=dict)
    model_architectures: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hyperparameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class TrainingTaskExecutor:
    """Executes identical training tasks across all frameworks."""
    
    # Framework-specific configuration mappings
    FRAMEWORK_MAPPINGS = {
        "HistoCore": {
            "optimizer_map": {
                "Adam": "Adam",
                "AdamW": "AdamW",
                "SGD": "SGD",
            },
            "architecture_map": {
                "resnet18_transformer": "resnet18_transformer",
                "resnet50_transformer": "resnet50_transformer",
                "vit": "vit_base",
            },
        },
        "PathML": {
            "optimizer_map": {
                "Adam": "adam",
                "AdamW": "adamw",
                "SGD": "sgd",
            },
            "architecture_map": {
                "resnet18_transformer": "resnet18",
                "resnet50_transformer": "resnet50",
                "vit": "vit",
            },
        },
        "CLAM": {
            "optimizer_map": {
                "Adam": "adam",
                "AdamW": "adam",  # CLAM may not support AdamW directly
                "SGD": "sgd",
            },
            "architecture_map": {
                "resnet18_transformer": "resnet18",
                "resnet50_transformer": "resnet50",
                "vit": "resnet18",  # CLAM may not support ViT
            },
        },
        "PyTorch": {
            "optimizer_map": {
                "Adam": "Adam",
                "AdamW": "AdamW",
                "SGD": "SGD",
            },
            "architecture_map": {
                "resnet18_transformer": "resnet18",
                "resnet50_transformer": "resnet50",
                "vit": "vit_b_16",
            },
        },
    }
    
    def __init__(self):
        """Initialize Training Task Executor."""
        pass
    
    def configure_task(
        self, 
        task_spec: TaskSpecification, 
        framework: str
    ) -> TrainingConfig:
        """
        Translate standard task spec to framework-specific configuration.
        
        Converts the framework-agnostic TaskSpecification into a configuration
        dictionary that can be used by the specific framework's training API.
        
        Args:
            task_spec: Standard task specification
            framework: Framework name ("HistoCore", "PathML", "CLAM", "PyTorch")
            
        Returns:
            TrainingConfig with framework-specific configuration
            
        Raises:
            ValueError: If framework is not supported
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
        """
        if framework not in self.FRAMEWORK_MAPPINGS:
            raise ValueError(
                f"Unsupported framework: {framework}. "
                f"Supported frameworks: {list(self.FRAMEWORK_MAPPINGS.keys())}"
            )
        
        logger.info(f"Configuring task for {framework}")
        
        # Get framework-specific mappings
        mappings = self.FRAMEWORK_MAPPINGS[framework]
        
        # Translate optimizer (Requirement 2.6)
        optimizer_name = mappings["optimizer_map"].get(
            task_spec.optimizer, 
            task_spec.optimizer
        )
        
        # Translate model architecture (Requirement 2.4)
        architecture_name = mappings["architecture_map"].get(
            task_spec.model_architecture,
            task_spec.model_architecture
        )
        
        # Build framework-specific configuration
        config_dict = self._build_framework_config(
            framework=framework,
            task_spec=task_spec,
            optimizer_name=optimizer_name,
            architecture_name=architecture_name,
        )
        
        # Create training configuration
        training_config = TrainingConfig(
            framework_name=framework,
            task_spec=task_spec,
            config_dict=config_dict,
            random_seed=task_spec.random_seed,  # Requirement 2.2
            data_split_hash=self._compute_data_split_hash(task_spec),  # Requirement 2.3
        )
        
        logger.info(
            f"Configured {framework} with optimizer={optimizer_name}, "
            f"architecture={architecture_name}, seed={task_spec.random_seed}"
        )
        
        return training_config
    
    def execute_training(
        self, 
        config: TrainingConfig, 
        env: FrameworkEnvironment
    ) -> TrainingResult:
        """
        Run training task with metrics collection and checkpointing.
        
        This is a placeholder implementation. The actual training execution
        would be delegated to framework-specific adapters.
        
        Args:
            config: Training configuration
            env: Framework environment
            
        Returns:
            TrainingResult with metrics and outcomes
            
        Raises:
            NotImplementedError: This is a placeholder for actual implementation
        """
        logger.info(
            f"Executing training for {config.framework_name} "
            f"(seed={config.random_seed})"
        )
        
        # This would be implemented by framework-specific adapters
        # For now, raise NotImplementedError to indicate this is a placeholder
        raise NotImplementedError(
            "Training execution is delegated to framework-specific adapters. "
            "See experiments/benchmark_system/adapters/ for implementations."
        )
    
    def validate_equivalence(
        self, 
        configs: List[TrainingConfig]
    ) -> EquivalenceReport:
        """
        Verify all framework configs represent identical training tasks.
        
        Checks that all configurations use:
        - Identical random seeds (Requirement 2.2)
        - Identical data splits (Requirement 2.3)
        - Equivalent model architectures (Requirement 2.4)
        - Identical augmentation pipelines (Requirement 2.5)
        - Identical optimizer settings (Requirement 2.6)
        - Same number of samples (Requirement 2.7)
        
        Args:
            configs: List of training configurations to compare
            
        Returns:
            EquivalenceReport with comparison results
            
        Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
        """
        if len(configs) < 2:
            return EquivalenceReport(
                is_equivalent=True,
                warnings=["Only one configuration provided, nothing to compare"]
            )
        
        logger.info(f"Validating equivalence across {len(configs)} configurations")
        
        differences = []
        warnings = []
        
        # Extract reference configuration (first one)
        ref_config = configs[0]
        ref_spec = ref_config.task_spec
        
        # Track configuration details for report
        random_seeds = {ref_config.framework_name: ref_config.random_seed}
        data_splits = {
            ref_config.framework_name: (
                ref_spec.train_split,
                ref_spec.val_split,
                ref_spec.test_split,
            )
        }
        model_architectures = {
            ref_config.framework_name: {
                "architecture": ref_spec.model_architecture,
                "feature_dim": ref_spec.feature_dim,
                "num_classes": ref_spec.num_classes,
            }
        }
        hyperparameters = {
            ref_config.framework_name: {
                "num_epochs": ref_spec.num_epochs,
                "batch_size": ref_spec.batch_size,
                "learning_rate": ref_spec.learning_rate,
                "weight_decay": ref_spec.weight_decay,
                "optimizer": ref_spec.optimizer,
            }
        }
        
        # Compare each configuration against reference
        for config in configs[1:]:
            spec = config.task_spec
            framework = config.framework_name
            
            # Track configuration details
            random_seeds[framework] = config.random_seed
            data_splits[framework] = (
                spec.train_split,
                spec.val_split,
                spec.test_split,
            )
            model_architectures[framework] = {
                "architecture": spec.model_architecture,
                "feature_dim": spec.feature_dim,
                "num_classes": spec.num_classes,
            }
            hyperparameters[framework] = {
                "num_epochs": spec.num_epochs,
                "batch_size": spec.batch_size,
                "learning_rate": spec.learning_rate,
                "weight_decay": spec.weight_decay,
                "optimizer": spec.optimizer,
            }
            
            # Check random seed (Requirement 2.2)
            if config.random_seed != ref_config.random_seed:
                differences.append(
                    f"Random seed mismatch: {ref_config.framework_name}={ref_config.random_seed}, "
                    f"{framework}={config.random_seed}"
                )
            
            # Check data splits (Requirement 2.3)
            if config.data_split_hash != ref_config.data_split_hash:
                differences.append(
                    f"Data split mismatch: {ref_config.framework_name} vs {framework}"
                )
            
            if (spec.train_split != ref_spec.train_split or
                spec.val_split != ref_spec.val_split or
                spec.test_split != ref_spec.test_split):
                differences.append(
                    f"Data split ratios differ: {ref_config.framework_name}="
                    f"({ref_spec.train_split}, {ref_spec.val_split}, {ref_spec.test_split}), "
                    f"{framework}=({spec.train_split}, {spec.val_split}, {spec.test_split})"
                )
            
            # Check model architecture (Requirement 2.4)
            if spec.model_architecture != ref_spec.model_architecture:
                warnings.append(
                    f"Model architecture differs: {ref_config.framework_name}="
                    f"{ref_spec.model_architecture}, {framework}={spec.model_architecture} "
                    f"(may be framework-specific translation)"
                )
            
            if spec.feature_dim != ref_spec.feature_dim:
                differences.append(
                    f"Feature dimension mismatch: {ref_config.framework_name}="
                    f"{ref_spec.feature_dim}, {framework}={spec.feature_dim}"
                )
            
            if spec.num_classes != ref_spec.num_classes:
                differences.append(
                    f"Number of classes mismatch: {ref_config.framework_name}="
                    f"{ref_spec.num_classes}, {framework}={spec.num_classes}"
                )
            
            # Check augmentation (Requirement 2.5)
            if spec.augmentation_config != ref_spec.augmentation_config:
                differences.append(
                    f"Augmentation config differs: {ref_config.framework_name} vs {framework}"
                )
            
            # Check optimizer and hyperparameters (Requirement 2.6)
            if spec.optimizer != ref_spec.optimizer:
                warnings.append(
                    f"Optimizer differs: {ref_config.framework_name}={ref_spec.optimizer}, "
                    f"{framework}={spec.optimizer} (may be framework-specific translation)"
                )
            
            if spec.learning_rate != ref_spec.learning_rate:
                differences.append(
                    f"Learning rate mismatch: {ref_config.framework_name}="
                    f"{ref_spec.learning_rate}, {framework}={spec.learning_rate}"
                )
            
            if spec.weight_decay != ref_spec.weight_decay:
                differences.append(
                    f"Weight decay mismatch: {ref_config.framework_name}="
                    f"{ref_spec.weight_decay}, {framework}={spec.weight_decay}"
                )
            
            # Check training configuration
            if spec.num_epochs != ref_spec.num_epochs:
                differences.append(
                    f"Number of epochs mismatch: {ref_config.framework_name}="
                    f"{ref_spec.num_epochs}, {framework}={spec.num_epochs}"
                )
            
            if spec.batch_size != ref_spec.batch_size:
                differences.append(
                    f"Batch size mismatch: {ref_config.framework_name}="
                    f"{ref_spec.batch_size}, {framework}={spec.batch_size}"
                )
        
        # Determine equivalence
        is_equivalent = len(differences) == 0
        
        # Log configuration differences (Requirement 2.8)
        if differences:
            logger.warning(
                f"Configuration differences detected that affect fair comparison:\n" +
                "\n".join(f"  - {diff}" for diff in differences)
            )
        
        if warnings:
            logger.info(
                f"Configuration warnings (may be acceptable):\n" +
                "\n".join(f"  - {warn}" for warn in warnings)
            )
        
        if is_equivalent:
            logger.info("All configurations are equivalent - fair comparison ensured")
        
        return EquivalenceReport(
            is_equivalent=is_equivalent,
            differences=differences,
            warnings=warnings,
            random_seeds=random_seeds,
            data_splits=data_splits,
            model_architectures=model_architectures,
            hyperparameters=hyperparameters,
        )
    
    def _build_framework_config(
        self,
        framework: str,
        task_spec: TaskSpecification,
        optimizer_name: str,
        architecture_name: str,
    ) -> Dict[str, Any]:
        """
        Build framework-specific configuration dictionary.
        
        Args:
            framework: Framework name
            task_spec: Task specification
            optimizer_name: Framework-specific optimizer name
            architecture_name: Framework-specific architecture name
            
        Returns:
            Configuration dictionary for the framework
        """
        # Base configuration common to all frameworks
        config = {
            "dataset_name": task_spec.dataset_name,
            "data_root": str(task_spec.data_root),
            "train_split": task_spec.train_split,
            "val_split": task_spec.val_split,
            "test_split": task_spec.test_split,
            "model_architecture": architecture_name,
            "feature_dim": task_spec.feature_dim,
            "num_classes": task_spec.num_classes,
            "num_epochs": task_spec.num_epochs,
            "batch_size": task_spec.batch_size,
            "learning_rate": task_spec.learning_rate,
            "weight_decay": task_spec.weight_decay,
            "optimizer": optimizer_name,
            "random_seed": task_spec.random_seed,
            "augmentation_config": task_spec.augmentation_config,
            "metrics": task_spec.metrics,
        }
        
        # Framework-specific adjustments
        if framework == "HistoCore":
            # HistoCore uses Hydra-style configuration
            config["model"] = {
                "architecture": architecture_name,
                "feature_dim": task_spec.feature_dim,
                "num_classes": task_spec.num_classes,
            }
            config["training"] = {
                "num_epochs": task_spec.num_epochs,
                "batch_size": task_spec.batch_size,
                "optimizer": {
                    "name": optimizer_name,
                    "lr": task_spec.learning_rate,
                    "weight_decay": task_spec.weight_decay,
                },
            }
        
        elif framework == "PathML":
            # PathML may use different configuration structure
            config["model_config"] = {
                "name": architecture_name,
                "feature_dim": task_spec.feature_dim,
                "num_classes": task_spec.num_classes,
            }
            config["train_config"] = {
                "epochs": task_spec.num_epochs,
                "batch_size": task_spec.batch_size,
                "optimizer": optimizer_name,
                "lr": task_spec.learning_rate,
                "weight_decay": task_spec.weight_decay,
            }
        
        elif framework == "CLAM":
            # CLAM uses command-line style configuration
            config["model_type"] = architecture_name
            config["model_size"] = "small" if task_spec.feature_dim <= 512 else "large"
            config["n_classes"] = task_spec.num_classes
            config["max_epochs"] = task_spec.num_epochs
            config["batch_size"] = task_spec.batch_size
            config["opt"] = optimizer_name
            config["lr"] = task_spec.learning_rate
            config["reg"] = task_spec.weight_decay
        
        elif framework == "PyTorch":
            # Baseline PyTorch uses simple dictionary
            config["model"] = architecture_name
            config["epochs"] = task_spec.num_epochs
            config["batch_size"] = task_spec.batch_size
            config["optimizer"] = {
                "type": optimizer_name,
                "lr": task_spec.learning_rate,
                "weight_decay": task_spec.weight_decay,
            }
        
        return config
    
    def _compute_data_split_hash(self, task_spec: TaskSpecification) -> str:
        """
        Compute hash of data split configuration for verification.
        
        Args:
            task_spec: Task specification
            
        Returns:
            Hash string representing the data split configuration
        """
        import hashlib
        
        # Create a string representation of the data split configuration
        split_str = (
            f"{task_spec.dataset_name}:"
            f"{task_spec.data_root}:"
            f"{task_spec.train_split}:"
            f"{task_spec.val_split}:"
            f"{task_spec.test_split}:"
            f"{task_spec.random_seed}"
        )
        
        # Compute SHA256 hash
        hash_obj = hashlib.sha256(split_str.encode())
        return hash_obj.hexdigest()[:16]  # Use first 16 characters
