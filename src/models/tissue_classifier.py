"""
Lightweight tissue classifier for tissue-aware sampling.

Classifies patches into tissue types (tumor, stroma, necrosis, background)
to enable importance-weighted sub-bag sampling in MIL training.
"""

from typing import Optional

import torch
import torch.nn as nn


class TissueClassifier(nn.Module):
    """
    Lightweight tissue type classifier for patch-level scoring.

    Uses a simple 2-layer MLP on foundation model embeddings to predict
    tissue type scores. Designed to be fast and memory-efficient for
    online sampling during training.

    Tissue types:
    - 0: Background (low importance)
    - 1: Stroma (medium importance)
    - 2: Necrosis (medium importance)
    - 3: Tumor (high importance)

    Args:
        feature_dim: Dimension of input patch features (e.g., 1024 for UNI)
        hidden_dim: Hidden dimension for MLP (default: 128)
        num_tissue_types: Number of tissue types to classify (default: 4)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> classifier = TissueClassifier(feature_dim=1024)
        >>> features = torch.randn(100, 1024)  # 100 patches
        >>> scores = classifier(features)
        >>> scores.shape
        torch.Size([100, 4])
        >>>
        >>> # Get importance weights for sampling
        >>> importance = classifier.get_importance_weights(features)
        >>> importance.shape
        torch.Size([100])
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_tissue_types: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_tissue_types <= 0:
            raise ValueError(f"num_tissue_types must be positive, got {num_tissue_types}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_tissue_types = num_tissue_types
        self.dropout = dropout

        # Lightweight 2-layer MLP
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tissue_types),
        )

        # Default importance weights for each tissue type
        # [background, stroma, necrosis, tumor]
        self.register_buffer("importance_weights", torch.tensor([0.1, 0.3, 0.3, 1.0]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get tissue type logits.

        Args:
            features: Patch features [N, D] or [B, N, D]

        Returns:
            logits: Tissue type logits [N, num_tissue_types] or [B, N, num_tissue_types]
        """
        # Handle both 2D and 3D inputs
        if features.dim() == 2:
            # [N, D] -> [N, num_tissue_types]
            return self.classifier(features)
        elif features.dim() == 3:
            # [B, N, D] -> [B, N, num_tissue_types]
            batch_size, num_patches, feature_dim = features.shape
            features_flat = features.view(-1, feature_dim)
            logits_flat = self.classifier(features_flat)
            return logits_flat.view(batch_size, num_patches, self.num_tissue_types)
        else:
            raise ValueError(f"Expected 2D [N, D] or 3D [B, N, D] input, got {features.dim()}D")

    def get_importance_weights(
        self, features: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Get importance weights for tissue-aware sampling.

        Computes tissue type probabilities and weights them by importance.
        Higher weights = more likely to be sampled.

        Args:
            features: Patch features [N, D] or [B, N, D]
            temperature: Temperature for softmax (default: 1.0)
                        Higher = more uniform, lower = more peaked

        Returns:
            importance: Importance weights [N] or [B, N]

        Example:
            >>> classifier = TissueClassifier(feature_dim=1024)
            >>> features = torch.randn(100, 1024)
            >>> importance = classifier.get_importance_weights(features, temperature=0.5)
            >>> # Use for weighted sampling
            >>> probs = importance / importance.sum()
            >>> indices = torch.multinomial(probs, num_samples=50)
        """
        # Get tissue type logits
        logits = self.forward(features)  # [N, 4] or [B, N, 4]

        # Apply temperature scaling
        logits = logits / temperature

        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)  # [N, 4] or [B, N, 4]

        # Weight by importance
        # importance = sum(prob_i * weight_i)
        importance = torch.matmul(probs, self.importance_weights)  # [N] or [B, N]

        return importance

    def set_importance_weights(self, weights: torch.Tensor):
        """
        Set custom importance weights for tissue types.

        Args:
            weights: Importance weights [num_tissue_types]

        Example:
            >>> classifier = TissueClassifier(feature_dim=1024)
            >>> # Emphasize tumor even more
            >>> custom_weights = torch.tensor([0.05, 0.2, 0.2, 2.0])
            >>> classifier.set_importance_weights(custom_weights)
        """
        if weights.shape[0] != self.num_tissue_types:
            raise ValueError(f"Expected {self.num_tissue_types} weights, got {weights.shape[0]}")

        self.importance_weights = weights.to(self.importance_weights.device)

    def predict_tissue_types(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict tissue types (argmax).

        Args:
            features: Patch features [N, D] or [B, N, D]

        Returns:
            tissue_types: Predicted tissue types [N] or [B, N]
        """
        logits = self.forward(features)
        return torch.argmax(logits, dim=-1)

    def get_tissue_distribution(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get tissue type distribution (probabilities).

        Args:
            features: Patch features [N, D] or [B, N, D]

        Returns:
            probs: Tissue type probabilities [N, num_tissue_types] or [B, N, num_tissue_types]
        """
        logits = self.forward(features)
        return torch.softmax(logits, dim=-1)


class PretrainedTissueClassifier(TissueClassifier):
    """
    Tissue classifier with pretrained weights.

    Can be initialized from a checkpoint trained on labeled tissue data.
    Falls back to random initialization if checkpoint not found.

    Args:
        feature_dim: Dimension of input patch features
        checkpoint_path: Path to pretrained weights (optional)
        **kwargs: Additional arguments for TissueClassifier

    Example:
        >>> classifier = PretrainedTissueClassifier(
        ...     feature_dim=1024,
        ...     checkpoint_path='checkpoints/tissue_classifier.pth'
        ... )
    """

    def __init__(self, feature_dim: int, checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__(feature_dim=feature_dim, **kwargs)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

            if "model_state_dict" in checkpoint:
                self.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.load_state_dict(checkpoint)

            print(f"Loaded tissue classifier from {checkpoint_path}")
        except FileNotFoundError:
            print(
                f"Warning: Checkpoint not found at {checkpoint_path}, using random initialization"
            )
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}, using random initialization")
