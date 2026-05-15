"""
Sliding window inference for nnMIL Multiple Instance Learning.

This module implements sliding window inference to handle large bags that exceed
the fixed bag length during inference. It divides bags into overlapping windows,
processes each independently, and aggregates predictions.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..data.data_models import Bag, InferenceOutput


class SlidingWindowInference:
    """
    Sliding window inference for processing large bags.

    Divides D-dimensional bags into overlapping H-dimensional chunks,
    processes each window through the model independently, and aggregates
    predictions via mean pooling. Computes epistemic uncertainty from
    variance across windows.

    Args:
        model: nnMIL model for inference
        window_size: Size of each window (H)
        stride: Step size between windows (default: window_size // 4)
        device: Device for computation (default: 'cuda' if available)
        enable_uncertainty: Whether to compute uncertainty (default: True)

    Example:
        >>> model = nnMIL(feature_dim=1024, num_classes=2)
        >>> inference = SlidingWindowInference(model, window_size=512)
        >>>
        >>> # Large bag with 2000 patches
        >>> bag = Bag(
        ...     features=torch.randn(2000, 1024),
        ...     label=1,
        ...     num_patches=2000,
        ...     slide_id="large_slide"
        ... )
        >>>
        >>> output = inference(bag)
        >>> print(f"Logits: {output.logits}")
        >>> print(f"Epistemic uncertainty: {output.epistemic_uncertainty}")
    """

    def __init__(
        self,
        model: nn.Module,
        window_size: int,
        stride: Optional[int] = None,
        device: Optional[str] = None,
        enable_uncertainty: bool = True,
        use_flash_attention: bool = True,
    ):
        self.model = model
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size // 4
        self.device = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.enable_uncertainty = enable_uncertainty
        self.use_flash_attention = use_flash_attention and hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

        # Validate parameters
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")

        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")

        if self.stride > self.window_size:
            raise ValueError(
                f"stride ({self.stride}) cannot exceed window_size ({self.window_size})"
            )

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(
        self, bag: Union[Bag, torch.Tensor], slide_id: Optional[str] = None
    ) -> InferenceOutput:
        """
        Perform sliding window inference on a bag.

        Args:
            bag: Input bag or features tensor
            slide_id: Slide identifier (required if bag is tensor)

        Returns:
            InferenceOutput with aggregated predictions and uncertainties
        """
        # Extract features and slide_id
        if isinstance(bag, Bag):
            features = bag.features
            slide_id = bag.slide_id
        else:
            features = bag
            if slide_id is None:
                raise ValueError("slide_id required when bag is tensor")

        # Move features to device
        features = features.to(self.device)

        # Check if sliding window is needed
        num_patches = features.shape[0]
        if num_patches <= self.window_size:
            # Single window inference
            return self._single_window_inference(features, slide_id)

        # Multi-window inference
        return self._multi_window_inference(features, slide_id)

    def _single_window_inference(self, features: torch.Tensor, slide_id: str) -> InferenceOutput:
        """Process single window (no sliding needed)."""
        # Pad if necessary
        num_patches = features.shape[0]
        if num_patches < self.window_size:
            padding = torch.zeros(
                self.window_size - num_patches,
                features.shape[1],
                device=features.device,
                dtype=features.dtype,
            )
            padded_features = torch.cat([features, padding], dim=0)

            # Create attention mask
            mask = torch.zeros(self.window_size, dtype=torch.bool, device=features.device)
            mask[:num_patches] = True
        else:
            padded_features = features[: self.window_size]
            mask = torch.ones(self.window_size, dtype=torch.bool, device=features.device)

        # Add batch dimension
        batch_features = padded_features.unsqueeze(0)  # [1, H, D]
        batch_mask = mask.unsqueeze(0)  # [1, H]

        with torch.no_grad():
            # Forward pass
            if hasattr(self.model, "forward_with_attention"):
                logits, attention_weights = self.model.forward_with_attention(
                    batch_features, batch_mask
                )
            else:
                logits = self.model(batch_features, return_attention=True)
                if isinstance(logits, tuple):
                    logits, attention_weights = logits
                else:
                    attention_weights = torch.ones(1, self.window_size, device=features.device)

        # Compute probabilities for classification
        probabilities = None
        aleatoric_uncertainty = torch.zeros(1, device=features.device)

        if logits.shape[1] > 1:  # Classification
            probabilities = torch.softmax(logits, dim=1)
            # Aleatoric uncertainty = entropy
            log_probs = torch.log_softmax(logits, dim=1)
            aleatoric_uncertainty = -(probabilities * log_probs).sum(dim=1)

        # No epistemic uncertainty for single window
        epistemic_uncertainty = torch.zeros(1, device=features.device)
        total_uncertainty = aleatoric_uncertainty

        return InferenceOutput(
            logits=logits.squeeze(0),
            probabilities=probabilities.squeeze(0) if probabilities is not None else None,
            attention_weights=attention_weights.squeeze(0)[:num_patches],
            epistemic_uncertainty=epistemic_uncertainty.squeeze(0),
            aleatoric_uncertainty=aleatoric_uncertainty.squeeze(0),
            total_uncertainty=total_uncertainty.squeeze(0),
            slide_ids=[slide_id],
        )

    def _multi_window_inference(self, features: torch.Tensor, slide_id: str) -> InferenceOutput:
        """Process multiple overlapping windows."""
        num_patches = features.shape[0]

        # Generate window positions
        window_starts = list(range(0, num_patches - self.window_size + 1, self.stride))
        if window_starts[-1] + self.window_size < num_patches:
            # Add final window to cover remaining patches
            window_starts.append(num_patches - self.window_size)

        # Process each window
        all_logits = []
        all_probabilities = []
        all_attention_weights = []
        all_aleatoric_uncertainties = []

        with torch.no_grad():
            for start_idx in window_starts:
                end_idx = start_idx + self.window_size
                window_features = features[start_idx:end_idx]

                # Add batch dimension
                batch_features = window_features.unsqueeze(0)  # [1, H, D]
                batch_mask = torch.ones(
                    1, self.window_size, dtype=torch.bool, device=features.device
                )

                # Forward pass
                if hasattr(self.model, "forward_with_attention"):
                    logits, attention_weights = self.model.forward_with_attention(
                        batch_features, batch_mask
                    )
                else:
                    logits = self.model(batch_features, return_attention=True)
                    if isinstance(logits, tuple):
                        logits, attention_weights = logits
                    else:
                        attention_weights = torch.ones(1, self.window_size, device=features.device)

                all_logits.append(logits.squeeze(0))
                all_attention_weights.append(attention_weights.squeeze(0))

                # Compute probabilities and aleatoric uncertainty
                if logits.shape[1] > 1:  # Classification
                    probs = torch.softmax(logits, dim=1)
                    all_probabilities.append(probs.squeeze(0))

                    # Aleatoric uncertainty = entropy
                    log_probs = torch.log_softmax(logits, dim=1)
                    aleatoric = -(probs * log_probs).sum(dim=1)
                    all_aleatoric_uncertainties.append(aleatoric.squeeze(0))
                else:  # Regression
                    all_aleatoric_uncertainties.append(torch.zeros(1, device=features.device))

        # Aggregate predictions
        stacked_logits = torch.stack(all_logits, dim=0)  # [K, num_classes] or [K]
        mean_logits = stacked_logits.mean(dim=0)

        # Aggregate probabilities
        mean_probabilities = None
        if all_probabilities:
            stacked_probabilities = torch.stack(all_probabilities, dim=0)
            mean_probabilities = stacked_probabilities.mean(dim=0)

        # Aggregate attention weights (mean pooling)
        stacked_attention = torch.stack(all_attention_weights, dim=0)  # [K, H]
        mean_attention = stacked_attention.mean(dim=0)

        # Compute uncertainties
        if self.enable_uncertainty:
            # Epistemic uncertainty = variance across windows
            if stacked_logits.dim() == 1:  # Regression
                epistemic_uncertainty = stacked_logits.var(dim=0)
            else:  # Classification
                epistemic_uncertainty = stacked_logits.var(dim=0).mean()

            # Aleatoric uncertainty = mean across windows
            mean_aleatoric = torch.stack(all_aleatoric_uncertainties, dim=0).mean(dim=0)

            # Combined uncertainty
            total_uncertainty = torch.sqrt(epistemic_uncertainty**2 + mean_aleatoric**2)
        else:
            epistemic_uncertainty = torch.zeros(1, device=features.device)
            mean_aleatoric = torch.zeros(1, device=features.device)
            total_uncertainty = torch.zeros(1, device=features.device)

        return InferenceOutput(
            logits=mean_logits,
            probabilities=mean_probabilities,
            attention_weights=mean_attention,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=mean_aleatoric,
            total_uncertainty=total_uncertainty,
            slide_ids=[slide_id],
        )

    def get_window_info(self, num_patches: int) -> Dict[str, Union[int, List[Tuple[int, int]]]]:
        """
        Get information about windows for a given number of patches.

        Args:
            num_patches: Number of patches in the bag

        Returns:
            Dictionary with window information:
            - num_windows: Number of windows
            - window_positions: List of (start, end) positions
            - overlap_ratio: Fraction of overlap between adjacent windows
        """
        if num_patches <= self.window_size:
            return {
                "num_windows": 1,
                "window_positions": [(0, min(num_patches, self.window_size))],
                "overlap_ratio": 0.0,
            }

        # Generate window positions
        window_starts = list(range(0, num_patches - self.window_size + 1, self.stride))
        if window_starts[-1] + self.window_size < num_patches:
            window_starts.append(num_patches - self.window_size)

        window_positions = [(start, start + self.window_size) for start in window_starts]

        # Calculate overlap ratio
        if len(window_positions) > 1:
            overlap_size = self.window_size - self.stride
            overlap_ratio = overlap_size / self.window_size
        else:
            overlap_ratio = 0.0

        return {
            "num_windows": len(window_positions),
            "window_positions": window_positions,
            "overlap_ratio": overlap_ratio,
        }
