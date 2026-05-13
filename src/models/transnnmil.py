"""
TransnnMIL: Fusion of TransMIL and nnMIL

This module implements TransnnMIL, a novel dual-branch MIL architecture that combines:
- Branch A: TransMIL (transformer-based correlator with self-attention)
- Branch B: nnMIL (lightweight gated attention aggregator)

The two branches are fused using a learnable scalar gate parameter that balances
their contributions. This design leverages both the global context modeling of
transformers and the efficiency of gated attention mechanisms.

Key features:
- Dual-branch architecture with learnable fusion gate
- TransMIL branch captures long-range patch dependencies via self-attention
- nnMIL branch provides efficient attention-weighted aggregation
- Positional encoding disabled in TransMIL for random sub-bag compatibility
- Compatible with existing training infrastructure (nnMILTrainer, samplers)
- Supports uncertainty estimation via sliding window inference

Architecture:
    Input: Bag of patch embeddings [B, K, D]
    ├─ Branch A (TransMIL): Transformer → CLS token → MLP → logits_A
    ├─ Branch B (nnMIL): Gated attention → Weighted sum → MLP → logits_B
    └─ Fusion: gate * logits_A + (1 - gate) * logits_B → final logits

The gate parameter is initialized to 0.0 (sigmoid(0) = 0.5), giving equal weight
to both branches initially. During training, the model learns the optimal balance.

Reference:
- TransMIL: Shao et al., "TransMIL: Transformer based Correlated Multiple Instance Learning"
- nnMIL: Stanford/NIH (2024), "No-New-Net Multiple Instance Learning"
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .nnmil import nnMIL
from .transmil import TransMIL


class TransnnMIL(nn.Module):
    """
    TransnnMIL: Dual-branch MIL with learnable fusion.
    
    Combines TransMIL (transformer-based) and nnMIL (gated attention) through
    a learnable scalar gate. The gate parameter controls the contribution of
    each branch to the final prediction.
    
    Args:
        feature_dim: Dimension of input patch features (e.g., 1024 for UNI)
        hidden_dim: Hidden dimension for both branches (default: 256)
        num_classes: Number of output classes (default: 2)
        num_layers: Number of transformer layers in Branch A (default: 2)
        num_heads: Number of attention heads in Branch A (default: 8)
        dropout: Dropout rate (default: 0.1 for TransMIL, 0.25 for nnMIL)
        use_pos_encoding: Enable positional encoding in TransMIL (default: False)
                         Set to False for random sub-bag sampling compatibility
    
    Example:
        >>> # Create TransnnMIL model
        >>> model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        >>> 
        >>> # Forward pass with fixed-length bags
        >>> features = torch.randn(4, 100, 1024)  # [batch, patches, features]
        >>> num_patches = torch.tensor([100, 80, 90, 100])  # actual patch counts
        >>> 
        >>> # Get predictions
        >>> logits = model(features, num_patches)
        >>> logits.shape
        torch.Size([4, 2])
        >>> 
        >>> # Get predictions with attention weights
        >>> logits, attention = model(features, num_patches, return_attention=True)
        >>> attention.shape  # From Branch A (TransMIL)
        torch.Size([4, 100])
        >>> 
        >>> # Check learned gate value
        >>> gate_value = torch.sigmoid(model.gate_param)
        >>> print(f"Branch A weight: {gate_value.item():.3f}")
        >>> print(f"Branch B weight: {1 - gate_value.item():.3f}")
    
    Notes:
        - The gate parameter is initialized to 0.0, giving equal weight (0.5) to both branches
        - During training, the gate learns to balance the branches based on the task
        - For uncertainty estimation, run the model on multiple sliding windows and
          compute variance across predictions
        - Both branches process the same input bag in parallel for efficiency
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_pos_encoding: bool = False,
    ):
        super().__init__()
        
        # Validate inputs
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_pos_encoding = use_pos_encoding
        
        # Branch A: TransMIL (Transformer-based correlator)
        # Disable positional encoding for random sub-bag compatibility
        self.branch_a = TransMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
        )
        
        # Branch B: nnMIL (Lightweight gated attention aggregator)
        # Use higher dropout (0.25) as per nnMIL paper
        self.branch_b = nnMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=0.25,  # nnMIL uses higher dropout than TransMIL
        )
        
        # Learnable fusion gate
        # Initialized to 0.0 → sigmoid(0) = 0.5 (equal weight to both branches)
        # During training, the model learns the optimal balance
        self.gate_param = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through TransnnMIL.
        
        Processes the input bag through both branches in parallel and fuses
        their outputs using the learnable gate parameter.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            num_patches: Actual patch counts [batch_size] for masking padded patches
            return_attention: If True, return attention weights from Branch A
        
        Returns:
            logits: Class predictions [batch_size, num_classes]
            attention_weights: (optional) Attention from Branch A [batch_size, num_patches]
        
        Notes:
            - Both branches process the same input in parallel
            - The gate parameter controls the contribution of each branch
            - Attention weights are returned from Branch A (TransMIL) for interpretability
            - Branch B attention weights are not returned but could be accessed separately
        """
        # Branch A: TransMIL forward pass
        if return_attention:
            logits_a, attention_a = self.branch_a(
                features, num_patches, return_attention=True
            )
        else:
            logits_a = self.branch_a(features, num_patches, return_attention=False)
        
        # Branch B: nnMIL forward pass
        logits_b = self.branch_b(features, num_patches, return_attention=False)
        
        # Compute fusion gate
        # gate ∈ (0, 1) via sigmoid activation
        gate = torch.sigmoid(self.gate_param)
        
        # Fuse logits from both branches
        # output = gate * logits_A + (1 - gate) * logits_B
        logits = gate * logits_a + (1 - gate) * logits_b
        
        if return_attention:
            # Return attention weights from Branch A (TransMIL)
            # These provide interpretability via transformer attention patterns
            return logits, attention_a
        else:
            return logits
    
    def get_gate_value(self) -> float:
        """
        Get the current fusion gate value.
        
        Returns:
            gate_value: Weight given to Branch A (TransMIL), in range (0, 1)
                       Branch B (nnMIL) receives weight (1 - gate_value)
        
        Example:
            >>> model = TransnnMIL(feature_dim=1024)
            >>> gate = model.get_gate_value()
            >>> print(f"TransMIL weight: {gate:.3f}, nnMIL weight: {1-gate:.3f}")
        """
        with torch.no_grad():
            return torch.sigmoid(self.gate_param).item()
    
    def get_branch_outputs(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get outputs from both branches separately (for analysis/debugging).
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            num_patches: Actual patch counts [batch_size]
        
        Returns:
            logits_a: Predictions from Branch A (TransMIL) [batch_size, num_classes]
            logits_b: Predictions from Branch B (nnMIL) [batch_size, num_classes]
            logits_fused: Final fused predictions [batch_size, num_classes]
        
        Example:
            >>> model = TransnnMIL(feature_dim=1024)
            >>> features = torch.randn(4, 100, 1024)
            >>> logits_a, logits_b, logits_fused = model.get_branch_outputs(features)
            >>> 
            >>> # Analyze branch agreement
            >>> preds_a = logits_a.argmax(dim=1)
            >>> preds_b = logits_b.argmax(dim=1)
            >>> agreement = (preds_a == preds_b).float().mean()
            >>> print(f"Branch agreement: {agreement:.2%}")
        """
        with torch.no_grad():
            # Get outputs from both branches
            logits_a = self.branch_a(features, num_patches, return_attention=False)
            logits_b = self.branch_b(features, num_patches, return_attention=False)
            
            # Compute fused output
            gate = torch.sigmoid(self.gate_param)
            logits_fused = gate * logits_a + (1 - gate) * logits_b
            
            return logits_a, logits_b, logits_fused

