"""
Monte Carlo Dropout for uncertainty quantification in nnMIL.

This module implements MC Dropout to estimate epistemic uncertainty
by performing multiple forward passes with dropout enabled at inference time.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


class MCDropoutInference:
    """
    Monte Carlo Dropout inference for uncertainty quantification.
    
    Performs K forward passes with dropout enabled to estimate epistemic
    uncertainty. Provides better calibration than single-pass inference.
    
    Args:
        model: nnMIL model for inference
        num_samples: Number of MC samples (default: 30)
        dropout_rate: Dropout rate during inference (default: 0.25)
        device: Device for computation (default: 'cuda' if available)
    
    Example:
        >>> model = nnMIL(feature_dim=1024, num_classes=2)
        >>> mc_inference = MCDropoutInference(model, num_samples=30)
        >>> 
        >>> features = torch.randn(1, 100, 1024)
        >>> num_patches = torch.tensor([100])
        >>> 
        >>> result = mc_inference(features, num_patches)
        >>> print(f"Mean prediction: {result['mean_logits']}")
        >>> print(f"Epistemic uncertainty: {result['epistemic_uncertainty']}")
        >>> print(f"Predictive entropy: {result['predictive_entropy']}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 30,
        dropout_rate: Optional[float] = None,
        device: Optional[str] = None
    ):
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Enable dropout at inference
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                if self.dropout_rate is not None:
                    module.p = self.dropout_rate
    
    def __call__(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        return_all_samples: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Perform MC Dropout inference.
        
        Args:
            features: Input features [B, N, D]
            num_patches: Actual patch counts [B]
            return_all_samples: Return all K samples (default: False)
        
        Returns:
            Dictionary with:
            - mean_logits: Mean prediction [B, num_classes]
            - epistemic_uncertainty: Variance across samples [B]
            - predictive_entropy: Entropy of mean prediction [B]
            - aleatoric_uncertainty: Mean entropy across samples [B]
            - total_uncertainty: Combined uncertainty [B]
            - all_samples: (optional) All K predictions [K, B, num_classes]
        """
        # Move to device
        features = features.to(self.device)
        if num_patches is not None:
            num_patches = num_patches.to(self.device)
        
        # Collect predictions from K forward passes
        all_logits = []
        all_probs = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                # Forward pass with dropout enabled
                logits = self.model(features, num_patches)
                all_logits.append(logits)
                
                # Compute probabilities for classification
                if logits.shape[-1] > 1:
                    probs = torch.softmax(logits, dim=-1)
                    all_probs.append(probs)
        
        # Stack predictions
        all_logits = torch.stack(all_logits, dim=0)  # [K, B, num_classes] or [K, B]
        
        # Compute mean prediction
        mean_logits = all_logits.mean(dim=0)  # [B, num_classes] or [B]
        
        # Compute epistemic uncertainty (variance across samples)
        if all_logits.dim() == 2:  # Regression
            epistemic_uncertainty = all_logits.var(dim=0)  # [B]
        else:  # Classification
            # Variance of logits (averaged across classes)
            epistemic_uncertainty = all_logits.var(dim=0).mean(dim=-1)  # [B]
        
        # Compute predictive entropy (uncertainty of mean prediction)
        if len(all_probs) > 0:
            # Classification task
            mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)  # [B, num_classes]
            log_mean_probs = torch.log(mean_probs + 1e-10)
            predictive_entropy = -(mean_probs * log_mean_probs).sum(dim=-1)  # [B]
            
            # Aleatoric uncertainty (mean entropy across samples)
            all_probs_stacked = torch.stack(all_probs, dim=0)  # [K, B, num_classes]
            log_probs = torch.log(all_probs_stacked + 1e-10)
            sample_entropies = -(all_probs_stacked * log_probs).sum(dim=-1)  # [K, B]
            aleatoric_uncertainty = sample_entropies.mean(dim=0)  # [B]
        else:
            # Regression task
            predictive_entropy = torch.zeros_like(epistemic_uncertainty)
            aleatoric_uncertainty = torch.zeros_like(epistemic_uncertainty)
        
        # Combined uncertainty
        total_uncertainty = torch.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        result = {
            'mean_logits': mean_logits,
            'epistemic_uncertainty': epistemic_uncertainty,
            'predictive_entropy': predictive_entropy,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty
        }
        
        if return_all_samples:
            result['all_samples'] = all_logits
        
        return result
    
    def calibrate(
        self,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        val_num_patches: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute calibration metrics on validation set.
        
        Args:
            val_features: Validation features [N, M, D]
            val_labels: Validation labels [N]
            val_num_patches: Validation patch counts [N]
        
        Returns:
            Dictionary with calibration metrics:
            - ece: Expected Calibration Error
            - mce: Maximum Calibration Error
            - brier_score: Brier score
        """
        # Get predictions
        result = self(val_features, val_num_patches)
        mean_logits = result['mean_logits']
        
        # Compute probabilities
        if mean_logits.dim() == 1:
            # Regression - skip calibration
            return {'ece': 0.0, 'mce': 0.0, 'brier_score': 0.0}
        
        probs = torch.softmax(mean_logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        
        # Move to CPU for numpy operations
        confidences = confidences.cpu().numpy()
        predictions = predictions.cpu().numpy()
        labels = val_labels.cpu().numpy()
        probs = probs.cpu().numpy()
        
        # Compute ECE (Expected Calibration Error)
        num_bins = 10
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                # Average confidence in this bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # ECE contribution
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                # MCE
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        # Compute Brier score
        num_classes = probs.shape[1]
        one_hot_labels = np.eye(num_classes)[labels]
        brier_score = np.mean(np.sum((probs - one_hot_labels)**2, axis=1))
        
        return {
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(brier_score)
        }
    
    def get_confidence_intervals(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        confidence_level: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """
        Compute confidence intervals for predictions.
        
        Args:
            features: Input features [B, N, D]
            num_patches: Actual patch counts [B]
            confidence_level: Confidence level (default: 0.95)
        
        Returns:
            Dictionary with:
            - mean: Mean prediction [B, num_classes]
            - lower: Lower bound [B, num_classes]
            - upper: Upper bound [B, num_classes]
        """
        # Get all samples
        result = self(features, num_patches, return_all_samples=True)
        all_samples = result['all_samples']  # [K, B, num_classes]
        
        # Compute percentiles
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = torch.quantile(all_samples, lower_percentile / 100, dim=0)
        upper = torch.quantile(all_samples, upper_percentile / 100, dim=0)
        mean = result['mean_logits']
        
        return {
            'mean': mean,
            'lower': lower,
            'upper': upper
        }
