"""
Training Utilities

Helper functions to reduce complexity in training loops.
"""

from typing import Dict, Any
import torch
import torch.nn as nn


class TrainingMetrics:
    """Track training metrics."""
    
    def __init__(self):
        self.total_loss = 0.0
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.num_valid_batches = 0
        self.num_skipped_batches = 0
    
    def update(self, loss: float, preds: torch.Tensor, labels: torch.Tensor, probs: torch.Tensor):
        """Update metrics with batch results."""
        self.total_loss += loss
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        self.all_probs.extend(probs.cpu().numpy())
        self.num_valid_batches += 1
    
    def skip_batch(self):
        """Increment skipped batch counter."""
        self.num_skipped_batches += 1
    
    def get_average_loss(self) -> float:
        """Get average loss."""
        return self.total_loss / max(1, self.num_valid_batches)


class NaNDetector:
    """Detect and track NaN occurrences."""
    
    def __init__(self, threshold: int = 3):
        self.consecutive_count = 0
        self.threshold = threshold
        self.detected_in_epoch = False
    
    def check(self, loss: torch.Tensor) -> bool:
        """Check if loss is NaN or too small."""
        if torch.isnan(loss) or loss < 1e-7:
            self.consecutive_count += 1
            self.detected_in_epoch = True
            return True
        else:
            self.consecutive_count = 0
            return False
    
    def is_cascading(self) -> bool:
        """Check if NaN is cascading."""
        return self.consecutive_count >= self.threshold
    
    def reset(self):
        """Reset detector."""
        self.consecutive_count = 0


def prepare_batch(batch: Dict, device: str, channels_last: bool = False) -> tuple:
    """Prepare batch for training.
    
    Args:
        batch: Batch dictionary
        device: Device to move tensors to
        channels_last: Whether to use channels_last memory format
        
    Returns:
        Tuple of (images, labels)
    """
    images = batch["image"].to(device)
    labels = batch["label"].to(device).float().unsqueeze(1)
    
    if channels_last and device.startswith("cuda"):
        images = images.to(memory_format=torch.channels_last)
    
    return images, labels


def forward_pass(
    images: torch.Tensor,
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module
) -> torch.Tensor:
    """Execute forward pass through model.
    
    Args:
        images: Input images
        feature_extractor: Feature extraction model
        encoder: Encoding model
        head: Classification head
        
    Returns:
        Logits tensor
    """
    # Extract features
    features = feature_extractor(images)
    
    # Add sequence dimension
    features = features.unsqueeze(1)
    
    # Encode
    encoded = encoder(features)
    
    # Classify
    logits = head(encoded)
    
    return logits


def compute_predictions(logits: torch.Tensor) -> tuple:
    """Compute predictions from logits.
    
    Args:
        logits: Model logits
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return preds, probs
