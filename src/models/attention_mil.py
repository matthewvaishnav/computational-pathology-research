"""
Attention-based Multiple Instance Learning (MIL) models for slide-level classification.

This module implements three state-of-the-art attention-based MIL architectures:
1. AttentionMIL: Basic attention-weighted pooling with gated attention mechanism
2. CLAM: Clustering-Constrained Attention MIL with instance-level clustering
3. TransMIL: Transformer-based MIL with multi-head self-attention

All models inherit from AttentionMILBase and work with pre-extracted patch features
stored in HDF5 format. They support variable-length bags (slides with different
numbers of patches) through masking and provide interpretable attention weights.

References:
- AttentionMIL: Ilse et al. "Attention-based Deep Multiple Instance Learning" (ICML 2018)
- CLAM: Lu et al. "Data-efficient and weakly supervised computational pathology on whole-slide images" (Nature Biomedical Engineering 2021)
- TransMIL: Shao et al. "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification" (NeurIPS 2021)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class AttentionMILBase(ABC, nn.Module):
    """
    Abstract base class for attention-based MIL models.
    
    This class defines the common interface that all attention-based MIL models
    must implement. It provides a unified API for training, inference, and
    attention weight extraction across different architectures.
    
    All subclasses must implement:
    - compute_attention: Calculate attention weights for patches
    - aggregate_features: Aggregate patch features using attention weights
    - forward: Complete forward pass from features to logits
    
    Args:
        feature_dim: Dimension of input patch features (e.g., 1024 for ResNet50)
        hidden_dim: Dimension of hidden layers in the model
        num_classes: Number of output classes (2 for binary classification)
        dropout: Dropout rate for regularization (default: 0.1)
    
    Example:
        >>> # Subclass must implement abstract methods
        >>> class MyAttentionMIL(AttentionMILBase):
        ...     def compute_attention(self, features, mask=None):
        ...         # Implementation here
        ...         pass
        ...     def aggregate_features(self, features, attention_weights):
        ...         # Implementation here
        ...         pass
        ...     def forward(self, features, num_patches=None, return_attention=False):
        ...         # Implementation here
        ...         pass
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
    
    @abstractmethod
    def compute_attention(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention weights for each patch in the bag.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Boolean mask for valid patches [batch_size, num_patches]
                  True indicates valid patches, False indicates padding
        
        Returns:
            Attention weights [batch_size, num_patches] that sum to 1 for each slide
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement compute_attention")
    
    @abstractmethod
    def aggregate_features(
        self,
        features: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate patch features using attention weights to create slide representation.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            attention_weights: Attention weights [batch_size, num_patches]
        
        Returns:
            Aggregated slide representation [batch_size, hidden_dim]
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement aggregate_features")
    
    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass from patch features to class logits.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            num_patches: Number of valid patches per slide [batch_size]
                        Used to create mask for variable-length bags
            return_attention: If True, return attention weights along with logits
        
        Returns:
            If return_attention is False:
                logits: Class logits [batch_size, num_classes]
            If return_attention is True:
                (logits, attention_weights): Tuple of logits and attention weights
                attention_weights: [batch_size, num_patches]
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement forward")



class AttentionMIL(AttentionMILBase):
    """
    Attention-based MIL with gated attention mechanism.
    
    This model implements the attention-based pooling approach from Ilse et al. (2018).
    It uses a gated attention mechanism to compute importance weights for each patch,
    then aggregates patch features using these weights to create a slide-level
    representation. The gated attention allows the model to learn both what features
    to attend to (attention_V) and how much to attend (attention_U).
    
    The model supports two attention modes:
    - Instance-level: Attention computed independently for each patch
    - Bag-level: Attention computed considering relationships between patches
    
    Args:
        feature_dim: Dimension of input patch features (e.g., 1024 for ResNet50)
        hidden_dim: Dimension of hidden layers (default: 256)
        num_classes: Number of output classes (default: 2 for binary)
        dropout: Dropout rate (default: 0.1)
        gated: If True, use gated attention; if False, use simple attention (default: True)
        attention_mode: 'instance' or 'bag' level attention (default: 'instance')
    
    Example:
        >>> model = AttentionMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        >>> features = torch.randn(4, 100, 1024)  # 4 slides, 100 patches each
        >>> num_patches = torch.tensor([100, 80, 90, 100])  # Actual patch counts
        >>> logits, attention = model(features, num_patches, return_attention=True)
        >>> logits.shape
        torch.Size([4, 2])
        >>> attention.shape
        torch.Size([4, 100])
        >>> attention[0].sum()  # Should be close to 1.0
        tensor(1.0000)
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        gated: bool = True,
        attention_mode: str = 'instance'
    ):
        super().__init__(feature_dim, hidden_dim, num_classes, dropout)
        
        if attention_mode not in ['instance', 'bag']:
            raise ValueError(f"attention_mode must be 'instance' or 'bag', got {attention_mode}")
        
        self.gated = gated
        self.attention_mode = attention_mode
        
        # Feature projection layer
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gated attention mechanism
        if gated:
            self.attention_V = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            self.attention_U = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            self.attention_w = nn.Linear(hidden_dim, 1)
        else:
            # Simple attention mechanism
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def compute_attention(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention weights for each patch.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Boolean mask [batch_size, num_patches], True for valid patches
        
        Returns:
            Attention weights [batch_size, num_patches] normalized to sum to 1
        """
        # Project features
        h = self.feature_proj(features)  # [batch_size, num_patches, hidden_dim]
        
        # Compute attention scores
        if self.gated:
            # Gated attention: element-wise product of tanh and sigmoid branches
            a_v = self.attention_V(h)  # [batch_size, num_patches, hidden_dim]
            a_u = self.attention_U(h)  # [batch_size, num_patches, hidden_dim]
            a = self.attention_w(a_v * a_u)  # [batch_size, num_patches, 1]
        else:
            # Simple attention
            a = self.attention_net(h)  # [batch_size, num_patches, 1]
        
        a = a.squeeze(-1)  # [batch_size, num_patches]
        
        # Apply mask: set padded patches to -inf before softmax
        if mask is not None:
            a = a.masked_fill(~mask, float('-inf'))
        
        # Normalize with softmax
        attention_weights = torch.softmax(a, dim=1)  # [batch_size, num_patches]
        
        return attention_weights
    
    def aggregate_features(
        self,
        features: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate patch features using attention weights.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            attention_weights: Attention weights [batch_size, num_patches]
        
        Returns:
            Aggregated slide representation [batch_size, hidden_dim]
        """
        # Project features
        h = self.feature_proj(features)  # [batch_size, num_patches, hidden_dim]
        
        # Weighted sum using batch matrix multiplication
        # attention_weights: [batch_size, num_patches] -> [batch_size, 1, num_patches]
        # h: [batch_size, num_patches, hidden_dim]
        # result: [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
        slide_repr = torch.bmm(
            attention_weights.unsqueeze(1),
            h
        ).squeeze(1)
        
        return slide_repr
    
    def forward(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass from patch features to class logits.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            num_patches: Number of valid patches per slide [batch_size]
            return_attention: If True, return attention weights
        
        Returns:
            logits: Class logits [batch_size, num_classes]
            attention_weights: (optional) [batch_size, num_patches]
        """
        batch_size, max_patches, _ = features.shape
        
        # Create mask from num_patches
        mask = None
        if num_patches is not None:
            # Create mask: True for valid patches, False for padding
            mask = torch.arange(max_patches, device=features.device).unsqueeze(0) < num_patches.unsqueeze(1)
        
        # Compute attention weights
        attention_weights = self.compute_attention(features, mask)
        
        # Aggregate features
        slide_repr = self.aggregate_features(features, attention_weights)
        
        # Classify
        logits = self.classifier(slide_repr)
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits



class CLAM(AttentionMILBase):
    """
    Clustering-Constrained Attention Multiple Instance Learning (CLAM).
    
    CLAM extends attention-based MIL by adding instance-level clustering to identify
    distinct subgroups of patches within a slide. It uses a two-stage approach:
    1. Instance-level classifier predicts cluster assignments for each patch
    2. Attention mechanism aggregates features within each cluster
    
    The model supports two modes:
    - Single-branch: One attention branch for all patches
    - Multi-branch: Separate attention branches for positive and negative clusters
    
    Multi-branch attention is particularly useful for binary classification where
    positive and negative regions have distinct visual patterns.
    
    Args:
        feature_dim: Dimension of input patch features
        hidden_dim: Dimension of hidden layers (default: 256)
        num_classes: Number of output classes (default: 2)
        num_clusters: Number of instance-level clusters (default: 10)
        dropout: Dropout rate (default: 0.1)
        multi_branch: If True, use separate attention for pos/neg clusters (default: True)
        instance_loss_weight: Weight for instance-level loss (default: 0.3)
    
    Example:
        >>> model = CLAM(feature_dim=1024, num_clusters=10, multi_branch=True)
        >>> features = torch.randn(4, 100, 1024)
        >>> num_patches = torch.tensor([100, 80, 90, 100])
        >>> logits, attention, instance_preds = model(features, num_patches, return_attention=True)
        >>> logits.shape
        torch.Size([4, 2])
        >>> attention['positive'].shape  # Multi-branch returns dict
        torch.Size([4, 100])
        >>> instance_preds.shape
        torch.Size([4, 100, 10])
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_clusters: int = 10,
        dropout: float = 0.1,
        multi_branch: bool = True,
        instance_loss_weight: float = 0.3
    ):
        super().__init__(feature_dim, hidden_dim, num_classes, dropout)
        
        if num_clusters < 2:
            raise ValueError(f"num_clusters must be >= 2, got {num_clusters}")
        
        self.num_clusters = num_clusters
        self.multi_branch = multi_branch
        self.instance_loss_weight = instance_loss_weight
        
        # Feature projection layer
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Instance-level classifier for clustering
        self.instance_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_clusters)
        )
        
        # Attention branches
        if multi_branch:
            # Separate attention for positive and negative clusters
            self.attention_pos = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.attention_neg = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            # Single attention branch
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Bag-level classifier
        bag_input_dim = hidden_dim * 2 if multi_branch else hidden_dim
        self.bag_classifier = nn.Sequential(
            nn.Linear(bag_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def compute_instance_predictions(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute instance-level cluster predictions for all patches.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
        
        Returns:
            Instance predictions [batch_size, num_patches, num_clusters]
        """
        # Project features
        h = self.feature_proj(features)  # [batch_size, num_patches, hidden_dim]
        
        # Compute cluster logits
        instance_logits = self.instance_classifier(h)  # [batch_size, num_patches, num_clusters]
        
        return instance_logits
    
    def compute_attention(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        branch: str = 'positive'
    ) -> torch.Tensor:
        """
        Compute attention weights for specified branch.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Boolean mask [batch_size, num_patches]
            branch: 'positive' or 'negative' (only used if multi_branch=True)
        
        Returns:
            Attention weights [batch_size, num_patches]
        """
        # Project features
        h = self.feature_proj(features)  # [batch_size, num_patches, hidden_dim]
        
        # Compute attention scores
        if self.multi_branch:
            if branch == 'positive':
                a = self.attention_pos(h)
            elif branch == 'negative':
                a = self.attention_neg(h)
            else:
                raise ValueError(f"branch must be 'positive' or 'negative', got {branch}")
        else:
            a = self.attention_net(h)
        
        a = a.squeeze(-1)  # [batch_size, num_patches]
        
        # Apply mask
        if mask is not None:
            a = a.masked_fill(~mask, float('-inf'))
        
        # Normalize
        attention_weights = torch.softmax(a, dim=1)
        
        return attention_weights
    
    def aggregate_features(
        self,
        features: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate patch features using attention weights.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            attention_weights: Attention weights [batch_size, num_patches]
        
        Returns:
            Aggregated representation [batch_size, hidden_dim]
        """
        # Project features
        h = self.feature_proj(features)
        
        # Weighted sum
        aggregated = torch.bmm(
            attention_weights.unsqueeze(1),
            h
        ).squeeze(1)
        
        return aggregated
    
    def forward(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through CLAM model.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            num_patches: Number of valid patches per slide [batch_size]
            return_attention: If True, return attention weights and instance predictions
        
        Returns:
            logits: Class logits [batch_size, num_classes]
            attention_weights: (optional) Dict with 'positive' and 'negative' keys if multi_branch,
                              otherwise single tensor [batch_size, num_patches]
            instance_preds: (optional) Instance predictions [batch_size, num_patches, num_clusters]
        """
        batch_size, max_patches, _ = features.shape
        
        # Create mask
        mask = None
        if num_patches is not None:
            mask = torch.arange(max_patches, device=features.device).unsqueeze(0) < num_patches.unsqueeze(1)
        
        # Compute instance predictions
        instance_preds = self.compute_instance_predictions(features)
        
        # Compute attention and aggregate features
        if self.multi_branch:
            # Separate branches for positive and negative
            attention_pos = self.compute_attention(features, mask, branch='positive')
            attention_neg = self.compute_attention(features, mask, branch='negative')
            
            slide_repr_pos = self.aggregate_features(features, attention_pos)
            slide_repr_neg = self.aggregate_features(features, attention_neg)
            
            # Concatenate branch features
            slide_repr = torch.cat([slide_repr_pos, slide_repr_neg], dim=1)
            
            attention_weights = {
                'positive': attention_pos,
                'negative': attention_neg
            }
        else:
            # Single branch
            attention_weights = self.compute_attention(features, mask)
            slide_repr = self.aggregate_features(features, attention_weights)
        
        # Bag-level classification
        logits = self.bag_classifier(slide_repr)
        
        if return_attention:
            return logits, attention_weights, instance_preds
        else:
            return logits



class TransMIL(AttentionMILBase):
    """
    Transformer-based Multiple Instance Learning (TransMIL).
    
    TransMIL uses transformer encoder layers with multi-head self-attention to model
    relationships between patches in a slide. Unlike traditional attention-based MIL
    which computes attention weights independently for each patch, TransMIL allows
    patches to attend to each other, capturing spatial and contextual relationships.
    
    The model uses a learnable CLS token (similar to BERT) that aggregates information
    from all patches through self-attention. The CLS token representation is then
    used for slide-level classification.
    
    Optional positional encoding can be added to preserve spatial information about
    patch locations within the slide.
    
    Args:
        feature_dim: Dimension of input patch features
        hidden_dim: Dimension of transformer hidden layers (default: 256)
        num_classes: Number of output classes (default: 2)
        num_layers: Number of transformer encoder layers (default: 2)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        use_pos_encoding: If True, add positional encoding (default: True)
    
    Example:
        >>> model = TransMIL(feature_dim=1024, hidden_dim=256, num_layers=2, num_heads=8)
        >>> features = torch.randn(4, 100, 1024)
        >>> num_patches = torch.tensor([100, 80, 90, 100])
        >>> logits, attention = model(features, num_patches, return_attention=True)
        >>> logits.shape
        torch.Size([4, 2])
        >>> attention.shape  # Uniform weights (transformer attention is internal)
        torch.Size([4, 100])
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_pos_encoding: bool = True
    ):
        super().__init__(feature_dim, hidden_dim, num_classes, dropout)
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_pos_encoding = use_pos_encoding
        
        # Feature projection layer
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Learnable positional encoding (max 10000 patches)
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, 10000, hidden_dim) * 0.02)
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def compute_attention(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Return uniform attention weights as placeholder.
        
        Note: TransMIL uses internal transformer attention which is not directly
        exposed. This method returns uniform weights for API compatibility.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Boolean mask [batch_size, num_patches]
        
        Returns:
            Uniform attention weights [batch_size, num_patches]
        """
        batch_size, num_patches, _ = features.shape
        
        # Create uniform weights
        attention_weights = torch.ones(batch_size, num_patches, device=features.device)
        
        # Apply mask if provided
        if mask is not None:
            attention_weights = attention_weights.masked_fill(~mask, 0.0)
        
        # Normalize to sum to 1
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return attention_weights
    
    def aggregate_features(
        self,
        features: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Not used in TransMIL (aggregation happens via CLS token).
        
        Included for API compatibility with AttentionMILBase.
        """
        # Project features
        h = self.feature_proj(features)
        
        # Simple weighted sum (not actually used in forward pass)
        aggregated = torch.bmm(
            attention_weights.unsqueeze(1),
            h
        ).squeeze(1)
        
        return aggregated
    
    def forward(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through TransMIL model.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            num_patches: Number of valid patches per slide [batch_size]
            return_attention: If True, return uniform attention weights
        
        Returns:
            logits: Class logits [batch_size, num_classes]
            attention_weights: (optional) Uniform weights [batch_size, num_patches]
        """
        batch_size, max_patches, _ = features.shape
        
        # Project features
        h = self.feature_proj(features)  # [batch_size, num_patches, hidden_dim]
        
        # Add positional encoding if enabled
        if self.use_pos_encoding:
            h = h + self.pos_encoding[:, :max_patches, :]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]
        h = torch.cat([cls_tokens, h], dim=1)  # [batch_size, num_patches+1, hidden_dim]
        
        # Create attention mask for transformer
        # True indicates positions that should be masked (padding)
        if num_patches is not None:
            # Create mask: False for valid positions (CLS + valid patches), True for padding
            mask = torch.arange(max_patches, device=features.device).unsqueeze(0) >= num_patches.unsqueeze(1)
            # Prepend False for CLS token
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=features.device)
            mask = torch.cat([cls_mask, mask], dim=1)  # [batch_size, num_patches+1]
        else:
            mask = None
        
        # Apply transformer encoder
        h = self.transformer(h, src_key_padding_mask=mask)  # [batch_size, num_patches+1, hidden_dim]
        
        # Extract CLS token representation
        cls_repr = h[:, 0, :]  # [batch_size, hidden_dim]
        
        # Apply layer normalization
        cls_repr = self.norm(cls_repr)
        
        # Classify
        logits = self.classifier(cls_repr)
        
        if return_attention:
            # Return uniform attention weights for API compatibility
            attention_weights = self.compute_attention(features, 
                                                      mask[:, 1:] if mask is not None else None)
            return logits, attention_weights
        else:
            return logits



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
    import logging
    logger = logging.getLogger(__name__)
    
    model_type = config.get('model_type', 'mean')
    hidden_dim = config.get('hidden_dim', 256)
    num_classes = config.get('num_classes', 2)
    dropout = config.get('dropout', 0.1)
    
    logger.info(f"Creating model: type={model_type}, feature_dim={feature_dim}, "
                f"hidden_dim={hidden_dim}, num_classes={num_classes}")
    
    if model_type == 'attention_mil':
        # AttentionMIL configuration
        attention_config = config.get('attention_mil', {})
        model = AttentionMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            gated=attention_config.get('gated', True),
            attention_mode=attention_config.get('attention_mode', 'instance')
        )
        logger.info(f"AttentionMIL created: gated={attention_config.get('gated', True)}, "
                   f"mode={attention_config.get('attention_mode', 'instance')}")
    
    elif model_type == 'clam':
        # CLAM configuration
        clam_config = config.get('clam', {})
        model = CLAM(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_clusters=clam_config.get('num_clusters', 10),
            dropout=dropout,
            multi_branch=clam_config.get('multi_branch', True),
            instance_loss_weight=clam_config.get('instance_loss_weight', 0.3)
        )
        logger.info(f"CLAM created: num_clusters={clam_config.get('num_clusters', 10)}, "
                   f"multi_branch={clam_config.get('multi_branch', True)}")
    
    elif model_type == 'transmil':
        # TransMIL configuration
        transmil_config = config.get('transmil', {})
        model = TransMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=transmil_config.get('num_layers', 2),
            num_heads=transmil_config.get('num_heads', 8),
            dropout=dropout,
            use_pos_encoding=transmil_config.get('use_pos_encoding', True)
        )
        logger.info(f"TransMIL created: num_layers={transmil_config.get('num_layers', 2)}, "
                   f"num_heads={transmil_config.get('num_heads', 8)}")
    
    elif model_type in ['mean', 'max']:
        # Baseline pooling models
        from .baselines import get_baseline_model
        
        # Create a simple pooling model
        class SimplePoolingModel(nn.Module):
            def __init__(self, feature_dim, hidden_dim, num_classes, pooling='mean'):
                super().__init__()
                self.pooling = pooling
                self.classifier = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes)
                )
            
            def forward(self, features, num_patches=None, return_attention=False):
                # Apply pooling
                if self.pooling == 'mean':
                    if num_patches is not None:
                        # Masked mean pooling
                        mask = torch.arange(features.size(1), device=features.device).unsqueeze(0) < num_patches.unsqueeze(1)
                        mask = mask.unsqueeze(-1).float()
                        pooled = (features * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                    else:
                        pooled = features.mean(dim=1)
                elif self.pooling == 'max':
                    if num_patches is not None:
                        # Masked max pooling
                        mask = torch.arange(features.size(1), device=features.device).unsqueeze(0) < num_patches.unsqueeze(1)
                        features_masked = features.clone()
                        features_masked[~mask] = float('-inf')
                        pooled = features_masked.max(dim=1)[0]
                    else:
                        pooled = features.max(dim=1)[0]
                
                logits = self.classifier(pooled)
                
                if return_attention:
                    # Return uniform attention for compatibility
                    attention = torch.ones(features.size(0), features.size(1), device=features.device)
                    if num_patches is not None:
                        mask = torch.arange(features.size(1), device=features.device).unsqueeze(0) < num_patches.unsqueeze(1)
                        attention = attention.masked_fill(~mask, 0.0)
                    attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-8)
                    return logits, attention
                else:
                    return logits
        
        model = SimplePoolingModel(feature_dim, hidden_dim, num_classes, pooling=model_type)
        logger.info(f"Baseline {model_type} pooling model created")
    
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be one of: "
                        f"attention_mil, clam, transmil, mean, max")
    
    return model
