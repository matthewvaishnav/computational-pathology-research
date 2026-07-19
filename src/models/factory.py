"""Factory for attention-based multiple-instance learning models."""

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


TRANSNNMIL_EXPERIMENTAL_TYPES = {
    "transnnmil_branch_attention_experimental": "TransnnMILBranchAttentionExperimental",
    "transnnmil_concat_experimental": "TransnnMILConcatExperimental",
    "transnnmil_gate_experimental": "TransnnMILGateExperimental",
}


def create_attention_model(config: Dict, feature_dim: int = 1024) -> nn.Module:
    """Create an MIL model from a configuration dictionary."""
    model_type = config.get("model_type", "mean")
    hidden_dim = config.get("hidden_dim", 256)
    num_classes = config.get("num_classes", 2)
    dropout = config.get("dropout", 0.1)

    logger.info(
        "Creating model: type=%s, feature_dim=%s, hidden_dim=%s, num_classes=%s",
        model_type,
        feature_dim,
        hidden_dim,
        num_classes,
    )

    if model_type == "attention_mil":
        from src.models.mil.attention_mil import AttentionMIL

        attention_config = config.get("attention_mil", {})
        return AttentionMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            gated=attention_config.get("gated", True),
            attention_mode=attention_config.get("attention_mode", "instance"),
        )

    if model_type == "clam":
        from src.models.mil.clam import CLAM

        clam_config = config.get("clam", {})
        return CLAM(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_clusters=clam_config.get("num_clusters", 10),
            dropout=dropout,
            multi_branch=clam_config.get("multi_branch", True),
            instance_loss_weight=clam_config.get("instance_loss_weight", 0.3),
        )

    if model_type == "transmil":
        from src.models.mil.transmil import TransMIL

        transmil_config = config.get("transmil", {})
        return TransMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=transmil_config.get("num_layers", 2),
            num_heads=transmil_config.get("num_heads", 8),
            dropout=dropout,
            use_pos_encoding=transmil_config.get("use_pos_encoding", True),
        )

    if model_type == "transnnmil":
        from src.models.transnnmil.transnnmil import TransnnMIL

        transnnmil_config = config.get("transnnmil", {})
        return TransnnMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=transnnmil_config.get("num_layers", 2),
            num_heads=transnnmil_config.get("num_heads", 8),
            dropout=dropout,
            use_pos_encoding=transnnmil_config.get("use_pos_encoding", False),
            enable_hierarchical=transnnmil_config.get("enable_hierarchical", False),
            enable_topology=transnnmil_config.get("enable_topology", False),
        )

    if model_type in TRANSNNMIL_EXPERIMENTAL_TYPES:
        from src.models.transnnmil import transnnmil_branch_token as experimental

        transnnmil_config = config.get("transnnmil", {})
        model_class = getattr(experimental, TRANSNNMIL_EXPERIMENTAL_TYPES[model_type])
        return model_class(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=transnnmil_config.get("num_layers", 2),
            num_heads=transnnmil_config.get("num_heads", 8),
            dropout=dropout,
            use_pos_encoding=transnnmil_config.get("use_pos_encoding", False),
            enable_hierarchical=transnnmil_config.get("enable_hierarchical", False),
            enable_topology=False,
        )

    if model_type in ["mean", "max"]:

        class SimplePoolingModel(nn.Module):
            def __init__(self, pooling: str) -> None:
                super().__init__()
                self.pooling = pooling
                self.classifier = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                )

            def forward(self, features, num_patches=None, return_attention=False):
                if num_patches is not None:
                    mask = torch.arange(features.size(1), device=features.device).unsqueeze(
                        0
                    ) < num_patches.unsqueeze(1)
                else:
                    mask = None

                if self.pooling == "mean":
                    if mask is None:
                        pooled = features.mean(dim=1)
                    else:
                        float_mask = mask.unsqueeze(-1).to(features.dtype)
                        pooled = (features * float_mask).sum(dim=1) / float_mask.sum(
                            dim=1
                        ).clamp_min(1.0)
                else:
                    if mask is None:
                        pooled = features.max(dim=1).values
                    else:
                        pooled = (
                            features.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                            .max(dim=1)
                            .values
                        )

                logits = self.classifier(pooled)
                if return_attention:
                    attention = torch.ones(
                        features.size(0), features.size(1), device=features.device
                    )
                    if mask is not None:
                        attention = attention.masked_fill(~mask, 0.0)
                    attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    return logits, attention
                return logits

        return SimplePoolingModel(model_type)

    allowed = [
        "attention_mil",
        "clam",
        "transmil",
        "transnnmil",
        *TRANSNNMIL_EXPERIMENTAL_TYPES.keys(),
        "mean",
        "max",
    ]
    raise ValueError(f"Invalid model_type: {model_type}. Must be one of: {', '.join(allowed)}")
