"""
Data models for nnMIL Multiple Instance Learning.

This module defines the core data structures used throughout the nnMIL pipeline:
- Bag: Represents a slide with patches and metadata
- TrainingBatch: Batch of fixed-length bags for training
- InferenceOutput: Model output with predictions and uncertainties
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


@dataclass
class Bag:
    """
    Represents a bag (slide) with patches.

    A bag contains patch features extracted from a whole-slide image,
    along with slide-level labels and metadata.

    Attributes:
        features: Patch features [N, D] where N is number of patches,
                 D is feature dimension from foundation model
        label: Slide-level label (class, target, or survival tuple)
        num_patches: Actual number of patches (before padding)
        slide_id: Unique slide identifier
        metadata: Optional metadata (patient_id, magnification, etc.)

    Example:
        >>> features = torch.randn(100, 1024)  # 100 patches, 1024-dim features
        >>> bag = Bag(
        ...     features=features,
        ...     label=1,  # Binary classification
        ...     num_patches=100,
        ...     slide_id="slide_001",
        ...     metadata={"patient_id": "P001", "magnification": "20x"}
        ... )
    """

    features: torch.Tensor
    label: Union[int, float, Tuple[float, int]]
    num_patches: int
    slide_id: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate bag data after initialization."""
        if self.features.dim() != 2:
            raise ValueError(f"features must be 2D tensor [N, D], got {self.features.dim()}D")

        if self.num_patches <= 0:
            raise ValueError(f"num_patches must be positive, got {self.num_patches}")

        if self.num_patches > self.features.shape[0]:
            raise ValueError(
                f"num_patches ({self.num_patches}) cannot exceed "
                f"features.shape[0] ({self.features.shape[0]})"
            )


@dataclass
class TrainingBatch:
    """
    Batch of fixed-length bags for training.

    Contains multiple bags that have been processed through FixedLengthBagSampler
    to ensure uniform length for efficient batching.

    Attributes:
        features: Fixed-length bag features [B, M, D] where B is batch size,
                 M is bag length, D is feature dimension
        labels: Slide-level labels [B]
        masks: Boolean masks [B, M] where True indicates valid patches,
               False indicates padded positions
        num_patches: Actual patch counts [B] before padding
        slide_ids: Slide identifiers [B]

    Example:
        >>> batch = TrainingBatch(
        ...     features=torch.randn(4, 512, 1024),  # 4 bags, 512 patches each
        ...     labels=torch.tensor([0, 1, 1, 0]),
        ...     masks=torch.ones(4, 512, dtype=torch.bool),
        ...     num_patches=torch.tensor([512, 400, 300, 512]),
        ...     slide_ids=["slide_001", "slide_002", "slide_003", "slide_004"]
        ... )
    """

    features: torch.Tensor
    labels: torch.Tensor
    masks: torch.Tensor
    num_patches: torch.Tensor
    slide_ids: List[str]

    def __post_init__(self):
        """Validate batch data after initialization."""
        if self.features.dim() != 3:
            raise ValueError(f"features must be 3D tensor [B, M, D], got {self.features.dim()}D")

        batch_size = self.features.shape[0]
        bag_length = self.features.shape[1]

        # Validate labels
        if self.labels.shape != (batch_size,):
            raise ValueError(
                f"labels shape {self.labels.shape} does not match " f"batch size {batch_size}"
            )

        # Validate masks
        if self.masks.shape != (batch_size, bag_length):
            raise ValueError(
                f"masks shape {self.masks.shape} does not match "
                f"expected ({batch_size}, {bag_length})"
            )

        if self.masks.dtype != torch.bool:
            raise ValueError(f"masks must have dtype torch.bool, got {self.masks.dtype}")

        # Validate num_patches
        if self.num_patches.shape != (batch_size,):
            raise ValueError(
                f"num_patches shape {self.num_patches.shape} does not match "
                f"batch size {batch_size}"
            )

        # Validate slide_ids
        if len(self.slide_ids) != batch_size:
            raise ValueError(
                f"slide_ids length {len(self.slide_ids)} does not match " f"batch size {batch_size}"
            )

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self.features.shape[0]

    @property
    def bag_length(self) -> int:
        """Return bag length (number of patches per bag)."""
        return self.features.shape[1]

    @property
    def feature_dim(self) -> int:
        """Return feature dimension."""
        return self.features.shape[2]


@dataclass
class InferenceOutput:
    """
    Output from nnMIL inference with uncertainty quantification.

    Contains model predictions along with attention weights and uncertainty
    estimates for clinical decision support.

    Attributes:
        logits: Raw model predictions [B, num_classes] or [B] for regression
        probabilities: Softmax probabilities [B, num_classes] (classification only)
        attention_weights: Attention weights [B, N] showing patch importance
        epistemic_uncertainty: Model uncertainty [B] due to lack of knowledge
        aleatoric_uncertainty: Data uncertainty [B] due to inherent noise
        total_uncertainty: Combined uncertainty [B] = sqrt(epistemic² + aleatoric²)
        slide_ids: Slide identifiers [B]

    Example:
        >>> output = InferenceOutput(
        ...     logits=torch.randn(4, 2),
        ...     probabilities=torch.softmax(torch.randn(4, 2), dim=1),
        ...     attention_weights=torch.rand(4, 100),
        ...     epistemic_uncertainty=torch.rand(4) * 0.1,
        ...     aleatoric_uncertainty=torch.rand(4) * 0.05,
        ...     total_uncertainty=torch.rand(4) * 0.12,
        ...     slide_ids=["slide_001", "slide_002", "slide_003", "slide_004"]
        ... )
    """

    logits: torch.Tensor
    probabilities: Optional[torch.Tensor]
    attention_weights: torch.Tensor
    epistemic_uncertainty: torch.Tensor
    aleatoric_uncertainty: torch.Tensor
    total_uncertainty: torch.Tensor
    slide_ids: List[str]

    def __post_init__(self):
        """Validate inference output after initialization."""
        # Determine batch size from logits
        if self.logits.dim() == 1:
            # Regression case: [B]
            batch_size = self.logits.shape[0]
            num_classes = None
        elif self.logits.dim() == 2:
            # Classification case: [B, num_classes]
            batch_size = self.logits.shape[0]
            num_classes = self.logits.shape[1]
        else:
            raise ValueError(
                f"logits must be 1D [B] or 2D [B, num_classes], " f"got {self.logits.dim()}D"
            )

        # Validate probabilities
        if self.probabilities is not None:
            if num_classes is None:
                raise ValueError("probabilities should be None for regression tasks")

            if self.probabilities.shape != (batch_size, num_classes):
                raise ValueError(
                    f"probabilities shape {self.probabilities.shape} does not match "
                    f"expected ({batch_size}, {num_classes})"
                )

        # Validate attention_weights
        if self.attention_weights.dim() != 2:
            raise ValueError(
                f"attention_weights must be 2D [B, N], got {self.attention_weights.dim()}D"
            )

        if self.attention_weights.shape[0] != batch_size:
            raise ValueError(
                f"attention_weights batch size {self.attention_weights.shape[0]} "
                f"does not match logits batch size {batch_size}"
            )

        # Validate uncertainty tensors
        for name, tensor in [
            ("epistemic_uncertainty", self.epistemic_uncertainty),
            ("aleatoric_uncertainty", self.aleatoric_uncertainty),
            ("total_uncertainty", self.total_uncertainty),
        ]:
            if tensor.shape != (batch_size,):
                raise ValueError(
                    f"{name} shape {tensor.shape} does not match " f"batch size {batch_size}"
                )

        # Validate slide_ids
        if len(self.slide_ids) != batch_size:
            raise ValueError(
                f"slide_ids length {len(self.slide_ids)} does not match " f"batch size {batch_size}"
            )

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return len(self.slide_ids)

    @property
    def num_patches(self) -> int:
        """Return number of patches (from attention weights)."""
        return self.attention_weights.shape[1]

    @property
    def is_classification(self) -> bool:
        """Return True if this is classification output (has probabilities)."""
        return self.probabilities is not None

    @property
    def is_regression(self) -> bool:
        """Return True if this is regression output (no probabilities)."""
        return self.probabilities is None
