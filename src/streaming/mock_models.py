"""
Mock Models for Testing Real-Time WSI Streaming

Provides dummy CNN encoder and attention models for testing the streaming pipeline
without requiring trained models.

Author: Matthew Vaishnav
Date: 2026-04-28
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class MockCNNEncoder(nn.Module):
    """
    Mock CNN encoder that simulates feature extraction.
    
    Returns random features with realistic dimensions for testing.
    """
    
    def __init__(self, feature_dim: int = 512):
        """
        Initialize mock CNN encoder.
        
        Args:
            feature_dim: Output feature dimension
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # Simple conv layers for realistic behavior
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(64, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from patches.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
        
        Returns:
            Features tensor [batch_size, feature_dim]
        """
        # Simple forward pass
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        
        # Add some realistic variation
        features = features + torch.randn_like(features) * 0.1
        
        return features


class MockAttentionMIL(nn.Module):
    """
    Mock attention-based MIL model for testing.
    
    Simulates attention weight computation and classification.
    """
    
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2):
        """
        Initialize mock attention MIL model.
        
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with attention.
        
        Args:
            features: Input features [batch_size, num_patches, feature_dim]
            return_attention: Whether to return attention weights
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
            attention_weights: Attention weights [batch_size, num_patches] (if return_attention=True)
        """
        # Compute attention weights
        attention_scores = self.attention(features)  # [batch_size, num_patches, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # Normalize across patches
        
        # Weighted aggregation
        weighted_features = torch.sum(features * attention_weights, dim=1)  # [batch_size, feature_dim]
        
        # Classification
        logits = self.classifier(weighted_features)  # [batch_size, num_classes]
        
        if return_attention:
            return logits, attention_weights.squeeze(-1)
        else:
            return logits


def create_mock_cnn_encoder(feature_dim: int = 512, device: str = 'cpu') -> nn.Module:
    """
    Create a mock CNN encoder for testing.
    
    Args:
        feature_dim: Output feature dimension
        device: Device to place model on
    
    Returns:
        Mock CNN encoder model
    """
    model = MockCNNEncoder(feature_dim=feature_dim)
    model = model.to(device)
    model.eval()
    
    return model


def create_mock_attention_model(
    feature_dim: int = 512,
    hidden_dim: int = 256,
    num_classes: int = 2,
    device: str = 'cpu'
) -> nn.Module:
    """
    Create a mock attention MIL model for testing.
    
    Args:
        feature_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        device: Device to place model on
    
    Returns:
        Mock attention MIL model
    """
    model = MockAttentionMIL(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    model = model.to(device)
    model.eval()
    
    return model


def create_mock_models(
    feature_dim: int = 512,
    device: str = 'cpu'
) -> Tuple[nn.Module, nn.Module]:
    """
    Create both mock models for testing.
    
    Args:
        feature_dim: Feature dimension
        device: Device to place models on
    
    Returns:
        Tuple of (cnn_encoder, attention_model)
    """
    cnn_encoder = create_mock_cnn_encoder(feature_dim=feature_dim, device=device)
    attention_model = create_mock_attention_model(feature_dim=feature_dim, device=device)
    
    return cnn_encoder, attention_model


# For backward compatibility with attention_aggregator.py
class AttentionMIL(MockAttentionMIL):
    """Alias for MockAttentionMIL for compatibility."""
    pass


if __name__ == "__main__":
    # Test mock models
    print("Testing Mock Models")
    print("=" * 80)
    
    # Create models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    cnn_encoder, attention_model = create_mock_models(feature_dim=512, device=device)
    
    # Test CNN encoder
    print("\nTesting CNN Encoder:")
    batch_size = 32
    test_patches = torch.randn(batch_size, 3, 224, 224).to(device)
    features = cnn_encoder(test_patches)
    print(f"  Input shape: {test_patches.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
    
    # Test attention model
    print("\nTesting Attention Model:")
    num_patches = 100
    test_features = torch.randn(1, num_patches, 512).to(device)
    logits, attention_weights = attention_model(test_features, return_attention=True)
    print(f"  Input shape: {test_features.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Attention weights shape: {attention_weights.shape}")
    print(f"  Attention sum: {attention_weights.sum():.6f} (should be ~1.0)")
    print(f"  Prediction: {torch.argmax(logits, dim=1).item()}")
    print(f"  Confidence: {torch.softmax(logits, dim=1).max().item():.3f}")
    
    print("\n" + "=" * 80)
    print("Mock models working correctly!")
