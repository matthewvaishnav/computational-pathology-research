"""
Uncertainty estimation for nnMIL Multiple Instance Learning.

This module implements uncertainty quantification methods including:
- Epistemic uncertainty: Model uncertainty due to lack of knowledge
- Aleatoric uncertainty: Data uncertainty due to inherent noise
- Combined uncertainty: Total prediction uncertainty
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.data_models import Bag, InferenceOutput


class UncertaintyEstimator:
    """
    Uncertainty estimation for nnMIL predictions.

    Implements Monte Carlo Dropout for epistemic uncertainty estimation
    and entropy-based aleatoric uncertainty for classification tasks.

    Args:
        model: nnMIL model with dropout layers
        num_samples: Number of MC dropout samples (default: 10)
        device: Device for computation (default: 'cuda' if available)
        normalize: Whether to normalize uncertainties to [0, 1] (default: True)

    Example:
        >>> model = nnMIL(feature_dim=1024, num_classes=2, dropout=0.25)
        >>> estimator = UncertaintyEstimator(model, num_samples=20)
        >>>
        >>> bag = Bag(
        ...     features=torch.randn(100, 1024),
        ...     label=1,
        ...     num_patches=100,
        ...     slide_id="test_slide"
        ... )
        >>>
        >>> output = estimator(bag)
        >>> print(f"Epistemic uncertainty: {output.epistemic_uncertainty:.3f}")
        >>> print(f"Aleatoric uncertainty: {output.aleatoric_uncertainty:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 10,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        self.model = model
        self.num_samples = num_samples
        self.device = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.normalize = normalize

        # Validate parameters
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

        # Move model to device
        self.model = self.model.to(self.device)

    def __call__(
        self,
        bag: Union[Bag, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        slide_id: Optional[str] = None,
    ) -> InferenceOutput:
        """
        Estimate uncertainty for a bag using Monte Carlo Dropout.

        Args:
            bag: Input bag or features tensor
            attention_mask: Attention mask for padded positions
            slide_id: Slide identifier (required if bag is tensor)

        Returns:
            InferenceOutput with uncertainty estimates
        """
        # Extract features and slide_id
        if isinstance(bag, Bag):
            features = bag.features
            slide_id = bag.slide_id
            num_patches = bag.num_patches
        else:
            features = bag
            if slide_id is None:
                raise ValueError("slide_id required when bag is tensor")
            num_patches = features.shape[0]

        # Move to device
        features = features.to(self.device)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(features.shape[0], dtype=torch.bool, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        # Add batch dimension
        batch_features = features.unsqueeze(0)  # [1, N, D]
        batch_mask = attention_mask.unsqueeze(0)  # [1, N]

        # Collect predictions with dropout enabled
        all_logits = []
        all_attention_weights = []

        # Enable dropout for uncertainty estimation
        self.model.train()

        with torch.no_grad():
            for _ in range(self.num_samples):
                # Forward pass with dropout
                if hasattr(self.model, "forward_with_attention"):
                    logits, attention_weights = self.model.forward_with_attention(
                        batch_features, batch_mask
                    )
                else:
                    logits = self.model(batch_features, batch_mask)
                    attention_weights = torch.ones(1, features.shape[0], device=self.device)

                all_logits.append(logits.squeeze(0))
                all_attention_weights.append(attention_weights.squeeze(0))

        # Return to eval mode
        self.model.eval()

        # Stack predictions
        stacked_logits = torch.stack(
            all_logits, dim=0
        )  # [num_samples, num_classes] or [num_samples]
        stacked_attention = torch.stack(all_attention_weights, dim=0)  # [num_samples, N]

        # Compute mean predictions
        mean_logits = stacked_logits.mean(dim=0)
        mean_attention = stacked_attention.mean(dim=0)

        # Compute probabilities for classification
        probabilities = None
        aleatoric_uncertainty = torch.zeros(1, device=self.device)

        if mean_logits.dim() > 0 and mean_logits.shape[-1] > 1:  # Classification
            # Mean probabilities
            all_probabilities = torch.softmax(stacked_logits, dim=-1)
            probabilities = all_probabilities.mean(dim=0)

            # Aleatoric uncertainty = mean entropy across samples
            log_probs = torch.log_softmax(stacked_logits, dim=-1)
            entropies = -(all_probabilities * log_probs).sum(dim=-1)
            aleatoric_uncertainty = entropies.mean(dim=0)

        # Epistemic uncertainty = variance across samples
        if mean_logits.dim() == 0 or mean_logits.shape[-1] == 1:  # Regression
            epistemic_uncertainty = stacked_logits.var(dim=0)
        else:  # Classification
            # Use variance of mean probabilities
            epistemic_uncertainty = all_probabilities.var(dim=0).mean()

        # Normalize uncertainties if requested
        if self.normalize:
            epistemic_uncertainty = self._normalize_uncertainty(
                epistemic_uncertainty, uncertainty_type="epistemic"
            )
            aleatoric_uncertainty = self._normalize_uncertainty(
                aleatoric_uncertainty, uncertainty_type="aleatoric"
            )

        # Combined uncertainty
        total_uncertainty = torch.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)

        return InferenceOutput(
            logits=mean_logits,
            probabilities=probabilities,
            attention_weights=mean_attention,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            slide_ids=[slide_id],
        )

    def _normalize_uncertainty(
        self, uncertainty: torch.Tensor, uncertainty_type: str
    ) -> torch.Tensor:
        """
        Normalize uncertainty to [0, 1] range.

        Args:
            uncertainty: Raw uncertainty values
            uncertainty_type: Type of uncertainty ('epistemic' or 'aleatoric')

        Returns:
            Normalized uncertainty in [0, 1]
        """
        if uncertainty_type == "epistemic":
            # Epistemic uncertainty normalization (model-dependent)
            # Use sigmoid to map to [0, 1]
            return torch.sigmoid(uncertainty)

        elif uncertainty_type == "aleatoric":
            # Aleatoric uncertainty normalization
            # For classification: entropy is already in reasonable range
            # For regression: use sigmoid
            if uncertainty.numel() == 1:
                return torch.sigmoid(uncertainty)
            else:
                # Classification entropy: normalize by log(num_classes)
                max_entropy = torch.log(torch.tensor(uncertainty.numel(), dtype=uncertainty.dtype))
                return uncertainty / max_entropy

        else:
            raise ValueError(f"Unknown uncertainty_type: {uncertainty_type}")

    def estimate_batch_uncertainty(
        self, features: torch.Tensor, attention_masks: torch.Tensor, slide_ids: List[str]
    ) -> InferenceOutput:
        """
        Estimate uncertainty for a batch of bags.

        Args:
            features: Batch features [B, N, D]
            attention_masks: Attention masks [B, N]
            slide_ids: Slide identifiers [B]

        Returns:
            InferenceOutput with batch uncertainty estimates
        """
        batch_size = features.shape[0]

        # Move to device
        features = features.to(self.device)
        attention_masks = attention_masks.to(self.device)

        # Collect predictions with dropout enabled
        all_logits = []
        all_attention_weights = []

        # Enable dropout for uncertainty estimation
        self.model.train()

        with torch.no_grad():
            for _ in range(self.num_samples):
                # Forward pass with dropout
                if hasattr(self.model, "forward_with_attention"):
                    logits, attention_weights = self.model.forward_with_attention(
                        features, attention_masks
                    )
                else:
                    logits = self.model(features, attention_masks)
                    attention_weights = torch.ones_like(attention_masks, dtype=torch.float)

                all_logits.append(logits)
                all_attention_weights.append(attention_weights)

        # Return to eval mode
        self.model.eval()

        # Stack predictions
        stacked_logits = torch.stack(
            all_logits, dim=0
        )  # [num_samples, B, num_classes] or [num_samples, B]
        stacked_attention = torch.stack(all_attention_weights, dim=0)  # [num_samples, B, N]

        # Compute mean predictions
        mean_logits = stacked_logits.mean(dim=0)  # [B, num_classes] or [B]
        mean_attention = stacked_attention.mean(dim=0)  # [B, N]

        # Compute probabilities and uncertainties
        probabilities = None
        aleatoric_uncertainties = torch.zeros(batch_size, device=self.device)

        if mean_logits.dim() > 1 and mean_logits.shape[-1] > 1:  # Classification
            # Mean probabilities
            all_probabilities = torch.softmax(stacked_logits, dim=-1)
            probabilities = all_probabilities.mean(dim=0)

            # Aleatoric uncertainty = mean entropy across samples
            log_probs = torch.log_softmax(stacked_logits, dim=-1)
            entropies = -(all_probabilities * log_probs).sum(dim=-1)
            aleatoric_uncertainties = entropies.mean(dim=0)

            # Epistemic uncertainty = variance of mean probabilities
            epistemic_uncertainties = all_probabilities.var(dim=0).mean(dim=-1)

        else:  # Regression
            # Epistemic uncertainty = variance across samples
            epistemic_uncertainties = stacked_logits.var(dim=0)
            if epistemic_uncertainties.dim() > 1:
                epistemic_uncertainties = epistemic_uncertainties.squeeze(-1)

        # Normalize uncertainties if requested
        if self.normalize:
            epistemic_uncertainties = torch.sigmoid(epistemic_uncertainties)
            if probabilities is not None:
                # Classification: normalize by log(num_classes)
                max_entropy = torch.log(
                    torch.tensor(probabilities.shape[-1], dtype=aleatoric_uncertainties.dtype)
                )
                aleatoric_uncertainties = aleatoric_uncertainties / max_entropy
            else:
                # Regression: use sigmoid
                aleatoric_uncertainties = torch.sigmoid(aleatoric_uncertainties)

        # Combined uncertainty
        total_uncertainties = torch.sqrt(epistemic_uncertainties**2 + aleatoric_uncertainties**2)

        return InferenceOutput(
            logits=mean_logits,
            probabilities=probabilities,
            attention_weights=mean_attention,
            epistemic_uncertainty=epistemic_uncertainties,
            aleatoric_uncertainty=aleatoric_uncertainties,
            total_uncertainty=total_uncertainties,
            slide_ids=slide_ids,
        )

    def get_uncertainty_stats(self, outputs: List[InferenceOutput]) -> Dict[str, Dict[str, float]]:
        """
        Compute uncertainty statistics across multiple outputs.

        Args:
            outputs: List of InferenceOutput objects

        Returns:
            Dictionary with uncertainty statistics:
            - epistemic: {mean, std, min, max}
            - aleatoric: {mean, std, min, max}
            - total: {mean, std, min, max}
        """
        # Collect all uncertainties
        epistemic_values = []
        aleatoric_values = []
        total_values = []

        for output in outputs:
            if output.epistemic_uncertainty.dim() == 0:
                epistemic_values.append(output.epistemic_uncertainty.item())
                aleatoric_values.append(output.aleatoric_uncertainty.item())
                total_values.append(output.total_uncertainty.item())
            else:
                epistemic_values.extend(output.epistemic_uncertainty.tolist())
                aleatoric_values.extend(output.aleatoric_uncertainty.tolist())
                total_values.extend(output.total_uncertainty.tolist())

        # Convert to tensors for statistics
        epistemic_tensor = torch.tensor(epistemic_values)
        aleatoric_tensor = torch.tensor(aleatoric_values)
        total_tensor = torch.tensor(total_values)

        return {
            "epistemic": {
                "mean": epistemic_tensor.mean().item(),
                "std": epistemic_tensor.std().item(),
                "min": epistemic_tensor.min().item(),
                "max": epistemic_tensor.max().item(),
            },
            "aleatoric": {
                "mean": aleatoric_tensor.mean().item(),
                "std": aleatoric_tensor.std().item(),
                "min": aleatoric_tensor.min().item(),
                "max": aleatoric_tensor.max().item(),
            },
            "total": {
                "mean": total_tensor.mean().item(),
                "std": total_tensor.std().item(),
                "min": total_tensor.min().item(),
                "max": total_tensor.max().item(),
            },
        }
