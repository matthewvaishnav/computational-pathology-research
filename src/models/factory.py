"""
Model factory for creating attention-based MIL models.

This module provides a factory function to create different types of MIL models
from configuration dictionaries. Supports:
- AttentionMIL: Basic attention-weighted pooling
- CLAM: Clustering-Constrained Attention MIL
- TransMIL: Transformer-based MIL
- Baseline pooling models (mean, max)
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

from .attention_mil import AttentionMIL
from .clam import CLAM
from .transmil import TransMIL

logger = logging.getLogger(__name__)


def create_attention_model(config: Dict, feature_dim: int = 1024) -> nn.Module:
    """
    Factory function to create attention-based MIL models from configuration.

    This function reads the model configuration and instantiates the appropriate
    attention-based MIL model (AttentionMIL, CLAM, or TransMIL). It also supports
    baseline pooling models (mean, max) for comparison.

    Args:
        config: Configuration dictionary with model parameters
        feature_dim: Dimension of input patch features (default: 1024)

    Returns:
        Instantiated model (AttentionMIL, CLAM, TransMIL, or baseline)

    Raises:
        ValueError: If model_type is invalid or required config is missing

    Example:
        >>> config = {
        ...     'model_type': 'attention_mil',
        ...     'hidden_dim': 256,
        ...     'num_classes': 2,
        ...     'attention_mil': {
        ...         'gated': True,
        ...         'attention_mode': 'instance'
        ...     }
        ... }
        >>> model = create_attention_model(config, feature_dim=1024)
    """
    model_type = config.get("model_type", "mean")
    hidden_dim = config.get("hidden_dim", 256)
    num_classes = config.get("num_classes", 2)
    dropout = config.get("dropout", 0.1)

    logger.info(
        f"Creating model: type={model_type}, feature_dim={feature_dim}, "
        f"hidden_dim={hidden_dim}, num_classes={num_classes}"
    )

    if model_type == "attention_mil":
        # AttentionMIL configuration
        attention_config = config.get("attention_mil", {})
        model = AttentionMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            gated=attention_config.get("gated", True),
            attention_mode=attention_config.get("attention_mode", "instance"),
        )
        logger.info(
            f"AttentionMIL created: gated={attention_config.get('gated', True)}, "
            f"mode={attention_config.get('attention_mode', 'instance')}"
        )

    elif model_type == "clam":
        # CLAM configuration
        clam_config = config.get("clam", {})
        model = CLAM(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_clusters=clam_config.get("num_clusters", 10),
            dropout=dropout,
            multi_branch=clam_config.get("multi_branch", True),
            instance_loss_weight=clam_config.get("instance_loss_weight", 0.3),
        )
        logger.info(
            f"CLAM created: num_clusters={clam_config.get('num_clusters', 10)}, "
            f"multi_branch={clam_config.get('multi_branch', True)}"
        )

    elif model_type == "transmil":
        # TransMIL configuration
        transmil_config = config.get("transmil", {})
        model = TransMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=transmil_config.get("num_layers", 2),
            num_heads=transmil_config.get("num_heads", 8),
            dropout=dropout,
            use_pos_encoding=transmil_config.get("use_pos_encoding", True),
        )
        logger.info(
            f"TransMIL created: num_layers={transmil_config.get('num_layers', 2)}, "
            f"num_heads={transmil_config.get('num_heads', 8)}"
        )

    elif model_type in ["mean", "max"]:
        # Baseline pooling models
        class SimplePoolingModel(nn.Module):
            """Simple pooling baseline model."""

            def __init__(self, feature_dim, hidden_dim, num_classes, pooling="mean"):
                super().__init__()
                self.pooling = pooling
                self.classifier = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                )

            def forward(self, features, num_patches=None, return_attention=False):
                # Apply pooling
                if self.pooling == "mean":
                    if num_patches is not None:
                        # Masked mean pooling
                        mask = torch.arange(features.size(1), device=features.device).unsqueeze(
                            0
                        ) < num_patches.unsqueeze(1)
                        mask = mask.unsqueeze(-1).float()
                        pooled = (features * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                    else:
                        pooled = features.mean(dim=1)
                elif self.pooling == "max":
                    if num_patches is not None:
                        # Masked max pooling
                        mask = torch.arange(features.size(1), device=features.device).unsqueeze(
                            0
                        ) < num_patches.unsqueeze(1)
                        features_masked = features.clone()
                        features_masked[~mask] = float("-inf")
                        pooled = features_masked.max(dim=1)[0]
                    else:
                        pooled = features.max(dim=1)[0]

                logits = self.classifier(pooled)

                if return_attention:
                    # Return uniform attention for compatibility
                    attention = torch.ones(
                        features.size(0), features.size(1), device=features.device
                    )
                    if num_patches is not None:
                        mask = torch.arange(features.size(1), device=features.device).unsqueeze(
                            0
                        ) < num_patches.unsqueeze(1)
                        attention = attention.masked_fill(~mask, 0.0)
                    attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-8)
                    return logits, attention
                else:
                    return logits

        model = SimplePoolingModel(feature_dim, hidden_dim, num_classes, pooling=model_type)
        logger.info(f"Baseline {model_type} pooling model created")

    else:
        raise ValueError(
            f"Invalid model_type: {model_type}. Must be one of: "
            f"attention_mil, clam, transmil, mean, max"
        )

    return model
