"""
Task-specific prediction heads for classification and survival analysis.

This module implements prediction heads that transform fused multimodal
representations into task-specific outputs (classification logits, survival
predictions, etc.).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ClassificationHead(nn.Module):
    """
    Classification head for multi-class or multi-label prediction.

    Transforms fused embeddings into class logits through a multi-layer
    perceptron with dropout and layer normalization for stable training.

    Args:
        input_dim: Dimension of input embeddings (default: 256)
        hidden_dim: Dimension of hidden layer (default: 128)
        num_classes: Number of output classes (default: 2)
        dropout: Dropout rate (default: 0.3)
        use_hidden_layer: Whether to use a hidden layer (default: True)

    Example:
        >>> head = ClassificationHead(input_dim=256, num_classes=5)
        >>> embeddings = torch.randn(16, 256)
        >>> logits = head(embeddings)
        >>> logits.shape
        torch.Size([16, 5])
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_hidden_layer: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_hidden_layer = use_hidden_layer

        if use_hidden_layer:
            # Multi-layer classification head
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            # Simple linear classifier
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, num_classes))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute classification logits.

        Args:
            embeddings: Input embeddings [batch_size, input_dim]

        Returns:
            Class logits [batch_size, num_classes]
        """
        logits = self.classifier(embeddings)
        return logits


class SurvivalPredictionHead(nn.Module):
    """
    Survival prediction head for time-to-event analysis.

    Predicts survival-related outputs including risk scores and optionally
    time-dependent hazard rates. Implements a flexible architecture that can
    output either:
    - Risk score (single value per sample)
    - Discrete time hazards (probability distribution over time bins)

    Args:
        input_dim: Dimension of input embeddings (default: 256)
        hidden_dim: Dimension of hidden layer (default: 128)
        num_time_bins: Number of discrete time bins for hazard prediction (default: None)
                       If None, outputs a single risk score
        dropout: Dropout rate (default: 0.3)
        use_hidden_layer: Whether to use a hidden layer (default: True)

    Example:
        >>> # Risk score prediction
        >>> head = SurvivalPredictionHead(input_dim=256)
        >>> embeddings = torch.randn(16, 256)
        >>> risk_scores = head(embeddings)
        >>> risk_scores.shape
        torch.Size([16, 1])

        >>> # Discrete time hazard prediction
        >>> head = SurvivalPredictionHead(input_dim=256, num_time_bins=10)
        >>> embeddings = torch.randn(16, 256)
        >>> hazards = head(embeddings)
        >>> hazards.shape
        torch.Size([16, 10])
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_time_bins: Optional[int] = None,
        dropout: float = 0.3,
        use_hidden_layer: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_time_bins = num_time_bins
        self.use_hidden_layer = use_hidden_layer

        # Determine output dimension
        if num_time_bins is not None:
            # Discrete time hazards
            output_dim = num_time_bins
            self.prediction_mode = "discrete_hazard"
        else:
            # Single risk score
            output_dim = 1
            self.prediction_mode = "risk_score"

        if use_hidden_layer:
            # Multi-layer prediction head
            self.predictor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            # Simple linear predictor
            self.predictor = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, output_dim))

    def forward(self, embeddings: torch.Tensor, return_hazards: bool = False) -> torch.Tensor:
        """
        Compute survival predictions.

        Args:
            embeddings: Input embeddings [batch_size, input_dim]
            return_hazards: If True and using discrete hazards, return hazard probabilities
                           instead of logits (applies sigmoid)

        Returns:
            If prediction_mode is 'risk_score':
                Risk scores [batch_size, 1]
            If prediction_mode is 'discrete_hazard':
                Hazard logits or probabilities [batch_size, num_time_bins]
        """
        output = self.predictor(embeddings)

        # Apply sigmoid for hazard probabilities if requested
        if return_hazards and self.prediction_mode == "discrete_hazard":
            output = torch.sigmoid(output)

        return output

    def compute_survival_curve(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute survival probability curve from discrete hazards.

        Only available when num_time_bins is specified. Computes the survival
        function S(t) = P(T > t) from discrete hazard predictions.

        Args:
            embeddings: Input embeddings [batch_size, input_dim]

        Returns:
            Survival probabilities [batch_size, num_time_bins]

        Raises:
            ValueError: If prediction_mode is not 'discrete_hazard'
        """
        if self.prediction_mode != "discrete_hazard":
            raise ValueError(
                "Survival curve computation requires discrete hazard prediction mode. "
                "Set num_time_bins when initializing the head."
            )

        # Get hazard probabilities
        hazards = self.forward(embeddings, return_hazards=True)  # [B, num_time_bins]

        # Compute survival probabilities: S(t) = prod(1 - h(i)) for i <= t
        # This is the probability of surviving past each time bin
        survival_probs = torch.cumprod(1 - hazards, dim=1)  # [B, num_time_bins]

        return survival_probs


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head combining classification and survival prediction.

    Enables joint training on multiple tasks by sharing the input embedding
    and producing task-specific outputs.

    Args:
        input_dim: Dimension of input embeddings (default: 256)
        classification_config: Configuration dict for classification head
        survival_config: Configuration dict for survival head
        shared_hidden_dim: Optional shared hidden layer dimension (default: None)
        dropout: Dropout rate (default: 0.3)

    Example:
        >>> head = MultiTaskHead(
        ...     input_dim=256,
        ...     classification_config={'num_classes': 5},
        ...     survival_config={'num_time_bins': 10}
        ... )
        >>> embeddings = torch.randn(16, 256)
        >>> class_logits, survival_output = head(embeddings)
        >>> class_logits.shape, survival_output.shape
        (torch.Size([16, 5]), torch.Size([16, 10]))
    """

    def __init__(
        self,
        input_dim: int = 256,
        classification_config: Optional[dict] = None,
        survival_config: Optional[dict] = None,
        shared_hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.shared_hidden_dim = shared_hidden_dim

        # Optional shared hidden layer
        if shared_hidden_dim is not None:
            self.shared_layer = nn.Sequential(
                nn.Linear(input_dim, shared_hidden_dim),
                nn.LayerNorm(shared_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_input_dim = shared_hidden_dim
        else:
            self.shared_layer = None
            head_input_dim = input_dim

        # Default configurations
        if classification_config is None:
            classification_config = {}
        if survival_config is None:
            survival_config = {}

        # Set input_dim for both heads
        classification_config["input_dim"] = head_input_dim
        survival_config["input_dim"] = head_input_dim

        # Initialize task-specific heads
        self.classification_head = ClassificationHead(**classification_config)
        self.survival_head = SurvivalPredictionHead(**survival_config)

    def forward(
        self, embeddings: torch.Tensor, return_survival_hazards: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-task predictions.

        Args:
            embeddings: Input embeddings [batch_size, input_dim]
            return_survival_hazards: If True, return survival hazard probabilities

        Returns:
            Tuple of:
                - Classification logits [batch_size, num_classes]
                - Survival predictions [batch_size, 1 or num_time_bins]
        """
        # Apply shared layer if present
        if self.shared_layer is not None:
            shared_features = self.shared_layer(embeddings)
        else:
            shared_features = embeddings

        # Compute task-specific outputs
        class_logits = self.classification_head(shared_features)
        survival_output = self.survival_head(
            shared_features, return_hazards=return_survival_hazards
        )

        return class_logits, survival_output
