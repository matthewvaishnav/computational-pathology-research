"""
Foundation model compatibility adapter for nnMIL.

This module provides adapters to integrate various foundation models
(UNI, CONCH, Phikon, ResNet50) with nnMIL architecture through
automatic dimension detection and adaptive projection.
"""

from typing import Dict, Union

import torch
import torch.nn as nn


class FoundationModelAdapter(nn.Module):
    """
    Adapter for foundation model compatibility with nnMIL.

    Automatically detects feature dimensions from input tensors and applies
    learned projection if needed. Supports weight freezing and fine-tuning
    with configurable learning rate multipliers.

    Supported foundation models:
    - UNI (1024-dim): Pathology foundation model
    - CONCH (512-dim): Contrastive learning model
    - Phikon (768-dim): Vision transformer for pathology
    - ResNet50 (2048-dim): Traditional CNN features

    Args:
        target_dim: Target feature dimension for nnMIL (default: 256)
        freeze_projection: Whether to freeze projection weights (default: False)
        lr_multiplier: Learning rate multiplier for adapter (default: 1.0)
        dropout: Dropout rate for projection (default: 0.1)

    Example:
        >>> adapter = FoundationModelAdapter(target_dim=256)
        >>>
        >>> # UNI features (1024-dim)
        >>> uni_features = torch.randn(32, 100, 1024)
        >>> adapted_features = adapter(uni_features)
        >>> print(adapted_features.shape)  # [32, 100, 256]
        >>>
        >>> # CONCH features (512-dim) - automatic adaptation
        >>> conch_features = torch.randn(32, 100, 512)
        >>> adapted_features = adapter(conch_features)
        >>> print(adapted_features.shape)  # [32, 100, 256]
    """

    def __init__(
        self,
        target_dim: int = 256,
        freeze_projection: bool = False,
        lr_multiplier: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.target_dim = target_dim
        self.freeze_projection = freeze_projection
        self.lr_multiplier = lr_multiplier
        self.dropout_rate = dropout

        # Dictionary to store projection layers for different input dimensions
        self.projections = nn.ModuleDict()

        # Common foundation model dimensions
        self.known_dimensions = {1024: "UNI", 512: "CONCH", 768: "Phikon", 2048: "ResNet50"}

        # Pre-create projections for known dimensions
        for dim, model_name in self.known_dimensions.items():
            if dim != target_dim:
                self._create_projection(dim, model_name)

        # Track which projection was last used (for logging)
        self.last_used_projection = None

    def _create_projection(self, input_dim: int, model_name: str = "Unknown"):
        """
        Create projection layer for specific input dimension.

        Args:
            input_dim: Input feature dimension
            model_name: Name of the foundation model (for logging)
        """
        projection_key = str(input_dim)

        if projection_key not in self.projections:
            # Create projection layer with residual connection if dimensions are close
            if abs(input_dim - self.target_dim) <= 64:
                # Simple linear projection for small differences
                projection = nn.Sequential(
                    nn.Linear(input_dim, self.target_dim),
                    nn.LayerNorm(self.target_dim),
                    nn.Dropout(self.dropout_rate),
                )
            else:
                # Two-layer projection for large differences
                hidden_dim = (input_dim + self.target_dim) // 2
                projection = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(hidden_dim, self.target_dim),
                    nn.LayerNorm(self.target_dim),
                    nn.Dropout(self.dropout_rate),
                )

            # Initialize weights
            self._initialize_projection(projection)

            # Freeze if requested
            if self.freeze_projection:
                for param in projection.parameters():
                    param.requires_grad = False

            self.projections[projection_key] = projection

            print(f"Created projection for {model_name} ({input_dim}D → {self.target_dim}D)")

    def _initialize_projection(self, projection: nn.Module):
        """Initialize projection weights."""
        for module in projection.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic dimension detection and projection.

        Args:
            features: Input features [B, N, D] or [N, D]

        Returns:
            Projected features [B, N, target_dim] or [N, target_dim]
        """
        # Handle different input shapes
        features.shape
        if features.dim() == 2:
            # [N, D] -> [1, N, D]
            features = features.unsqueeze(0)
            squeeze_output = True
        elif features.dim() == 3:
            # [B, N, D]
            squeeze_output = False
        else:
            raise ValueError(f"Expected 2D [N, D] or 3D [B, N, D] input, got {features.dim()}D")

        batch_size, num_patches, input_dim = features.shape

        # Check if projection is needed
        if input_dim == self.target_dim:
            # No projection needed
            self.last_used_projection = None
            output = features
        else:
            # Get or create projection
            projection_key = str(input_dim)

            if projection_key not in self.projections:
                # Create projection for unknown dimension
                model_name = self.known_dimensions.get(input_dim, f"Unknown_{input_dim}D")
                self._create_projection(input_dim, model_name)

            # Apply projection
            projection = self.projections[projection_key]
            self.last_used_projection = projection_key

            # Reshape for projection: [B, N, D] -> [B*N, D]
            features_flat = features.view(-1, input_dim)

            # Apply projection
            projected_flat = projection(features_flat)

            # Reshape back: [B*N, target_dim] -> [B, N, target_dim]
            output = projected_flat.view(batch_size, num_patches, self.target_dim)

        # Restore original shape if needed
        if squeeze_output:
            output = output.squeeze(0)

        return output

    def get_projection_info(self) -> Dict[str, Dict[str, Union[str, int, bool]]]:
        """
        Get information about available projections.

        Returns:
            Dictionary with projection information
        """
        info = {}

        for dim_str, projection in self.projections.items():
            input_dim = int(dim_str)
            model_name = self.known_dimensions.get(input_dim, f"Unknown_{input_dim}D")

            # Count parameters
            num_params = sum(p.numel() for p in projection.parameters())
            trainable_params = sum(p.numel() for p in projection.parameters() if p.requires_grad)

            info[dim_str] = {
                "model_name": model_name,
                "input_dim": input_dim,
                "output_dim": self.target_dim,
                "num_parameters": num_params,
                "trainable_parameters": trainable_params,
                "frozen": trainable_params == 0,
            }

        return info

    def freeze_projections(self):
        """Freeze all projection layers."""
        for projection in self.projections.values():
            for param in projection.parameters():
                param.requires_grad = False

        self.freeze_projection = True
        print("All projections frozen")

    def unfreeze_projections(self):
        """Unfreeze all projection layers."""
        for projection in self.projections.values():
            for param in projection.parameters():
                param.requires_grad = True

        self.freeze_projection = False
        print("All projections unfrozen")

    def get_lr_multiplier_params(self) -> Dict[str, float]:
        """
        Get learning rate multiplier for optimizer parameter groups.

        Returns:
            Dictionary mapping parameter names to LR multipliers
        """
        lr_params = {}

        for name, param in self.named_parameters():
            if param.requires_grad:
                lr_params[name] = self.lr_multiplier

        return lr_params

    def adapt_features_batch(
        self, features_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Adapt a batch of features from different foundation models.

        Args:
            features_dict: Dictionary mapping model names to feature tensors

        Returns:
            Dictionary with adapted features (all same dimension)

        Example:
            >>> features = {
            ...     'UNI': torch.randn(32, 100, 1024),
            ...     'CONCH': torch.randn(32, 100, 512),
            ...     'Phikon': torch.randn(32, 100, 768)
            ... }
            >>> adapted = adapter.adapt_features_batch(features)
            >>> # All outputs will be [32, 100, 256]
        """
        adapted_features = {}

        for model_name, features in features_dict.items():
            adapted_features[model_name] = self.forward(features)

        return adapted_features

    def benchmark_projections(
        self,
        batch_size: int = 32,
        num_patches: int = 100,
        num_iterations: int = 100,
        device: str = "cuda",
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark projection performance for different input dimensions.

        Args:
            batch_size: Batch size for benchmarking
            num_patches: Number of patches per bag
            num_iterations: Number of iterations to average
            device: Device for benchmarking

        Returns:
            Dictionary with timing results
        """
        import time

        self.to(device)
        self.eval()

        results = {}

        for dim_str in self.projections:
            input_dim = int(dim_str)
            model_name = self.known_dimensions.get(input_dim, f"Unknown_{input_dim}D")

            # Create test data
            test_features = torch.randn(batch_size, num_patches, input_dim, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = self.forward(test_features)

            # Benchmark
            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.time()

            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = self.forward(test_features)

            torch.cuda.synchronize() if device == "cuda" else None
            end_time = time.time()

            # Calculate metrics
            total_time = end_time - start_time
            avg_time_per_batch = total_time / num_iterations
            throughput = (batch_size * num_iterations) / total_time  # bags/second

            results[dim_str] = {
                "model_name": model_name,
                "input_dim": input_dim,
                "avg_time_ms": avg_time_per_batch * 1000,
                "throughput_bags_per_sec": throughput,
                "total_time_sec": total_time,
            }

        return results

    def extra_repr(self) -> str:
        """String representation of the adapter."""
        return (
            f"target_dim={self.target_dim}, "
            f"freeze_projection={self.freeze_projection}, "
            f"lr_multiplier={self.lr_multiplier}, "
            f"num_projections={len(self.projections)}"
        )
