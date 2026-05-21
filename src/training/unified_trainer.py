"""
Unified training interface for TransMIL and nnMIL models.

This module provides a unified training interface that supports both
TransMIL and nnMIL architectures through configuration-based selection,
maintaining backward compatibility while enabling new features.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config.nnmil_config import nnMILConfig
from ..models import TransMIL, nnMIL
from .nnmil_trainer import nnMILTrainer


class UnifiedTrainer:
    """
    Unified trainer supporting both TransMIL and nnMIL architectures.

    Provides a single interface for training different MIL models while
    maintaining backward compatibility with existing TransMIL workflows
    and enabling new nnMIL features.

    Args:
        config: Training configuration (nnMILConfig or dict)
        model_type: Model architecture ('nnmil' or 'transmil')
        device: Device for training (default: auto-detect)
        logger: Logger instance (creates new if None)

    Example:
        >>> # nnMIL training
        >>> config = nnMILConfig(
        ...     feature_dim=1024,
        ...     num_classes=2,
        ...     batch_size=32
        ... )
        >>> trainer = UnifiedTrainer(config, model_type='nnmil')
        >>> trainer.train(train_loader, val_loader)
        >>>
        >>> # TransMIL training (backward compatibility)
        >>> config = {
        ...     'feature_dim': 1024,
        ...     'num_classes': 2,
        ...     'batch_size': 16  # TransMIL typically uses smaller batches
        ... }
        >>> trainer = UnifiedTrainer(config, model_type='transmil')
        >>> trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        config: Union[nnMILConfig, Dict[str, Any]],
        model_type: str = "nnmil",
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_type = model_type.lower()
        self.device = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # Validate model type
        if self.model_type not in ["nnmil", "transmil"]:
            raise ValueError(f"model_type must be 'nnmil' or 'transmil', got {self.model_type}")

        # Convert config to nnMILConfig if needed
        if isinstance(config, dict):
            if self.model_type == "transmil":
                # Convert TransMIL config to nnMILConfig format
                config = self._convert_transmil_config(config)
            self.config = nnMILConfig(**config)
        else:
            self.config = config

        # Create model
        self.model = self._create_model()

        # Create trainer
        if self.model_type == "nnmil":
            self.trainer = nnMILTrainer(self.model, self.config, self.device, self.logger)
        else:
            # Use nnMILTrainer for TransMIL as well (unified interface)
            # Adjust config for TransMIL-specific settings
            transmil_config = self._adjust_config_for_transmil()
            self.trainer = nnMILTrainer(self.model, transmil_config, self.device, self.logger)

        self.logger.info(f"UnifiedTrainer initialized with {self.model_type.upper()} model")

    def _convert_transmil_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert TransMIL configuration to nnMILConfig format.

        Args:
            config: TransMIL configuration dictionary

        Returns:
            nnMILConfig-compatible configuration
        """
        # Map TransMIL parameters to nnMIL parameters
        converted = {
            "feature_dim": config.get("feature_dim", 1024),
            "hidden_dim": config.get("hidden_dim", 256),
            "num_classes": config.get("num_classes", 2),
            "dropout": config.get("dropout", 0.25),
            "batch_size": config.get("batch_size", 16),  # TransMIL default
            "learning_rate": config.get("learning_rate", 2e-4),  # TransMIL default
            "weight_decay": config.get("weight_decay", 1e-4),
            "num_epochs": config.get("num_epochs", 100),
            "patience": config.get("patience", 10),
            "task_type": config.get("task_type", "classification"),
            "bag_length": config.get("bag_length", 512),
            "sampler_type": "balanced",  # TransMIL uses balanced sampling
            "enable_uncertainty": False,  # TransMIL doesn't have uncertainty
            "multi_scale": False,  # TransMIL is single-scale
        }

        # Copy any additional parameters
        for key, value in config.items():
            if key not in converted:
                converted[key] = value

        return converted

    def _adjust_config_for_transmil(self) -> nnMILConfig:
        """
        Adjust configuration for TransMIL-specific training.

        Returns:
            Adjusted nnMILConfig for TransMIL
        """
        # Create a copy of the config
        config_dict = self.config.to_dict()

        # TransMIL-specific adjustments
        config_dict.update(
            {
                "enable_uncertainty": False,  # TransMIL doesn't support uncertainty
                "multi_scale": False,  # TransMIL is single-scale
                "sampler_type": "balanced",  # TransMIL uses balanced sampling
                "window_size": config_dict["bag_length"],  # No sliding window for TransMIL
                "stride": config_dict["bag_length"],  # No overlap for TransMIL
            }
        )

        return nnMILConfig(**config_dict)

    def _create_model(self) -> nn.Module:
        """
        Create model based on model_type.

        Returns:
            Model instance (nnMIL or TransMIL)
        """
        if self.model_type == "nnmil":
            model = nnMIL(
                feature_dim=self.config.feature_dim,
                hidden_dim=self.config.hidden_dim,
                num_classes=self.config.num_classes,
                dropout=self.config.dropout,
                multi_scale=self.config.multi_scale,
                fusion_type=self.config.fusion_type,
            )

        elif self.model_type == "transmil":
            model = TransMIL(
                feature_dim=self.config.feature_dim,
                hidden_dim=self.config.hidden_dim,
                num_classes=self.config.num_classes,
                dropout=self.config.dropout,
            )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return model.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Train the model using the unified interface.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history and statistics
        """
        self.logger.info(f"Starting {self.model_type.upper()} training")

        # Train using the appropriate trainer
        history = self.trainer.train(train_loader, val_loader, checkpoint_dir)

        # Get training statistics
        stats = self.trainer.get_training_stats()

        # Add model-specific information
        result = {
            "model_type": self.model_type,
            "config": self.config.to_dict(),
            "training_history": history,
            "training_stats": stats,
            "model_info": self._get_model_info(),
        }

        self.logger.info(f"{self.model_type.upper()} training completed")

        return result

    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        info = {
            "model_type": self.model_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024**2),  # Assuming float32
        }

        # Add model-specific information
        if self.model_type == "nnmil":
            info.update(
                {
                    "multi_scale": self.config.multi_scale,
                    "fusion_type": self.config.fusion_type,
                    "uncertainty_enabled": self.config.enable_uncertainty,
                    "bag_length": self.config.bag_length,
                    "window_size": self.config.window_size,
                    "stride": self.config.stride,
                }
            )
        elif self.model_type == "transmil":
            info.update(
                {
                    "multi_scale": False,
                    "uncertainty_enabled": False,
                    "bag_length": self.config.bag_length,
                }
            )

        return info

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint information
        """
        return self.trainer.load_checkpoint(checkpoint_path)

    def evaluate(
        self, test_loader: DataLoader, checkpoint_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            test_loader: Test data loader
            checkpoint_path: Path to checkpoint (optional)

        Returns:
            Evaluation results
        """
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()

        all_predictions = []
        all_targets = []
        all_slide_ids = []

        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch = self.trainer._move_batch_to_device(batch)

                # Forward pass
                if hasattr(self.model, "forward_with_attention"):
                    logits, attention_weights = self.model.forward_with_attention(
                        batch.features, batch.masks
                    )
                else:
                    logits = self.model(batch.features, batch.masks)

                all_predictions.append(logits.cpu())
                all_targets.append(batch.labels.cpu())
                all_slide_ids.extend(batch.slide_ids)

        # Concatenate results
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        metrics = self._compute_test_metrics(predictions, targets)

        return {
            "model_type": self.model_type,
            "metrics": metrics,
            "predictions": predictions,
            "targets": targets,
            "slide_ids": all_slide_ids,
            "num_samples": len(targets),
        }

    def _compute_test_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute comprehensive test metrics."""
        metrics = {}

        if self.config.task_type == "classification":
            # Classification metrics
            pred_classes = torch.argmax(predictions, dim=1)
            accuracy = (pred_classes == targets).float().mean().item()
            metrics["accuracy"] = accuracy

            # Per-class metrics
            num_classes = self.config.num_classes
            for class_idx in range(num_classes):
                class_mask = targets == class_idx
                if class_mask.sum() > 0:
                    class_acc = (pred_classes[class_mask] == class_idx).float().mean().item()
                    metrics[f"class_{class_idx}_accuracy"] = class_acc

            # Probabilities for AUC (if binary classification)
            if num_classes == 2:
                probs = torch.softmax(predictions, dim=1)[:, 1]
                try:
                    from sklearn.metrics import roc_auc_score

                    auc = roc_auc_score(targets.numpy(), probs.numpy())
                    metrics["auc"] = auc
                except ImportError:
                    self.logger.warning("sklearn not available for AUC computation")

        elif self.config.task_type == "regression":
            # Regression metrics
            mse = ((predictions.squeeze() - targets) ** 2).mean().item()
            mae = (predictions.squeeze() - targets).abs().mean().item()

            # R²
            ss_res = ((targets - predictions.squeeze()) ** 2).sum()
            ss_tot = ((targets - targets.mean()) ** 2).sum()
            r2 = 1 - (ss_res / (ss_tot + 1e-8))

            metrics.update({"mse": mse, "mae": mae, "r2": r2.item()})

        return metrics

    @classmethod
    def from_config_file(
        cls, config_path: Union[str, Path], model_type: Optional[str] = None, **kwargs
    ) -> "UnifiedTrainer":
        """
        Create UnifiedTrainer from configuration file.

        Args:
            config_path: Path to configuration file
            model_type: Model type override (inferred from config if None)
            **kwargs: Additional arguments for UnifiedTrainer

        Returns:
            UnifiedTrainer instance
        """
        config = nnMILConfig.from_yaml(config_path)

        # Infer model type from config if not specified
        if model_type is None:
            # Check for nnMIL-specific features
            if (
                config.multi_scale
                or config.enable_uncertainty
                or config.window_size != config.bag_length
            ):
                model_type = "nnmil"
            else:
                model_type = "transmil"  # Default to TransMIL for backward compatibility

        return cls(config, model_type, **kwargs)

    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        summary_lines = [
            f"Model Type: {self.model_type.upper()}",
            f"Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}",
            f"Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}",
            f"Feature Dimension: {self.config.feature_dim}",
            f"Hidden Dimension: {self.config.hidden_dim}",
            f"Number of Classes: {self.config.num_classes}",
            f"Dropout Rate: {self.config.dropout}",
        ]

        if self.model_type == "nnmil":
            summary_lines.extend(
                [
                    f"Multi-scale: {self.config.multi_scale}",
                    f"Fusion Type: {self.config.fusion_type}",
                    f"Uncertainty Enabled: {self.config.enable_uncertainty}",
                    f"Window Size: {self.config.window_size}",
                    f"Stride: {self.config.stride}",
                ]
            )

        return "\n".join(summary_lines)
