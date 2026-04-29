"""
Checkpoint Loader for Real-Time WSI Streaming

This module provides utilities to load trained model checkpoints from PCam training
and adapt them for use in the real-time streaming pipeline.

Author: Matthew Vaishnav
Date: 2026-04-28
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointLoader:
    """
    Utility class for loading trained model checkpoints.

    Supports loading checkpoints from PCam training that contain:
    - feature_extractor_state_dict: CNN encoder (ResNet or foundation model)
    - encoder_state_dict: WSI encoder
    - head_state_dict: Classification head

    Example:
        >>> loader = CheckpointLoader("checkpoints/pcam_real/best_model.pth")
        >>> cnn_encoder, attention_model = loader.load_for_streaming()
        >>> # Use in streaming pipeline
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize checkpoint loader.

        Args:
            checkpoint_path: Path to checkpoint file (.pth)
            device: Device to load models on ('cuda' or 'cpu')
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device if torch.cuda.is_available() else "cpu"

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Initializing checkpoint loader for: {checkpoint_path}")
        logger.info(f"Target device: {self.device}")

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load checkpoint file.

        Returns:
            Dictionary containing checkpoint data

        Raises:
            RuntimeError: If checkpoint loading fails
        """
        try:
            logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # Validate checkpoint structure
            required_keys = [
                "feature_extractor_state_dict",
                "encoder_state_dict",
                "head_state_dict",
            ]

            missing_keys = [k for k in required_keys if k not in checkpoint]
            if missing_keys:
                raise ValueError(
                    f"Checkpoint missing required keys: {missing_keys}. "
                    f"Available keys: {list(checkpoint.keys())}"
                )

            # Log checkpoint info
            if "epoch" in checkpoint:
                logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
            if "metrics" in checkpoint:
                metrics = checkpoint["metrics"]
                logger.info(f"Checkpoint metrics: {metrics}")

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            raise RuntimeError(f"Checkpoint loading failed: {e}") from e

    def load_feature_extractor(
        self, checkpoint: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """
        Load feature extractor (CNN encoder) from checkpoint.

        Args:
            checkpoint: Loaded checkpoint dictionary
            config: Optional config to reconstruct model architecture

        Returns:
            Feature extractor model
        """
        try:
            # Get state dict
            state_dict = checkpoint["feature_extractor_state_dict"]

            # Use config from checkpoint if not provided
            if config is None and "config" in checkpoint:
                config = checkpoint["config"]

            # Try to infer architecture from state dict or config
            # Check if it's a ResNet-based extractor
            if any("resnet" in k.lower() or "layer" in k.lower() for k in state_dict.keys()):
                logger.info("Detected ResNet-based feature extractor")
                from src.models.feature_extractors import ResNetFeatureExtractor

                # Get config if available
                if config:
                    fe_config = config.get("model", {}).get("feature_extractor", {})
                    model_name = fe_config.get("model", "resnet50")
                    feature_dim = fe_config.get("feature_dim", 512)
                    pretrained = fe_config.get(
                        "pretrained", False
                    )  # Don't use pretrained when loading from checkpoint
                else:
                    # Infer from state dict
                    model_name = "resnet50"
                    feature_dim = self._infer_feature_dim(state_dict)
                    pretrained = False

                # Create model
                model = ResNetFeatureExtractor(
                    model_name=model_name, pretrained=pretrained, feature_dim=feature_dim
                )

            # Check if it's a foundation model
            elif any(
                "foundation" in k.lower() or "encoder" in k.lower() for k in state_dict.keys()
            ):
                logger.info("Detected foundation model feature extractor")
                # For foundation models, we need the config
                if config is None:
                    raise ValueError("Config required for foundation model loading")

                foundation_config = config.get("model", {}).get("foundation", {})
                model_name = foundation_config.get("model_name", "phikon")

                from src.models.foundation import load_foundation_model
                from src.models.foundation.projector import FeatureProjector

                # Load foundation encoder
                foundation_encoder = load_foundation_model(
                    model_name=model_name, freeze=foundation_config.get("freeze", True)
                )

                # Create projector
                projector_config = foundation_config.get("projector", {})
                projector = FeatureProjector(
                    input_dim=foundation_encoder.feature_dim,
                    output_dim=projector_config.get("output_dim", 256),
                    dropout=projector_config.get("dropout", 0.1),
                )

                # Combine into wrapper
                class FoundationFeatureExtractor(nn.Module):
                    def __init__(self, encoder, projector):
                        super().__init__()
                        self.encoder = encoder
                        self.projector = projector

                    def forward(self, x):
                        features = self.encoder(x)
                        return self.projector(features)

                model = FoundationFeatureExtractor(foundation_encoder, projector)

            else:
                # Generic feature extractor - try to reconstruct from state dict
                logger.warning(
                    "Unknown feature extractor architecture, attempting to reconstruct from state dict"
                )
                from src.models.feature_extractors import ResNetFeatureExtractor

                # Default to ResNet50
                feature_dim = self._infer_feature_dim(state_dict)
                model = ResNetFeatureExtractor(
                    model_name="resnet50", pretrained=False, feature_dim=feature_dim
                )

            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(self.device)

            logger.info(f"Feature extractor loaded successfully on {self.device}")
            return model

        except Exception as e:
            logger.error(f"Failed to load feature extractor: {e}", exc_info=True)
            raise RuntimeError(f"Feature extractor loading failed: {e}") from e

    def load_attention_model(
        self, checkpoint: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """
        Load attention model from checkpoint (encoder + head combined).

        The streaming pipeline expects a single attention model that takes features
        and returns logits + attention weights. This combines the WSI encoder and
        classification head from the checkpoint.

        Args:
            checkpoint: Loaded checkpoint dictionary
            config: Optional config to reconstruct model architecture

        Returns:
            Combined attention model
        """
        try:
            # Get state dicts
            encoder_state_dict = checkpoint["encoder_state_dict"]
            head_state_dict = checkpoint["head_state_dict"]

            # Use config from checkpoint if not provided
            if config is None and "config" in checkpoint:
                config = checkpoint["config"]

            # Infer dimensions from state dicts
            input_dim = self._infer_encoder_input_dim(encoder_state_dict)
            hidden_dim = self._infer_encoder_hidden_dim(encoder_state_dict)
            num_classes = self._infer_num_classes(head_state_dict)

            logger.info(
                f"Inferred dimensions: input_dim={input_dim}, "
                f"hidden_dim={hidden_dim}, num_classes={num_classes}"
            )

            # Create encoder
            from src.models.encoders import WSIEncoder

            # Get encoder config if available
            if config:
                wsi_config = config.get("model", {}).get("wsi", {})
                embed_dim = config.get("model", {}).get("embed_dim", 256)
                encoder = WSIEncoder(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=embed_dim,
                    num_heads=wsi_config.get("num_heads", 8),
                    num_layers=wsi_config.get("num_layers", 2),
                    pooling=wsi_config.get("pooling", "mean"),
                    dropout=config.get("training", {}).get("dropout", 0.1),
                )
            else:
                # Use defaults with inferred dimensions
                encoder = WSIEncoder(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=256,
                    num_heads=8,
                    num_layers=2,
                    pooling="mean",
                    dropout=0.1,
                )

            # Create head
            from src.models.heads import ClassificationHead

            if config:
                classification_config = config.get("task", {}).get("classification", {})
                hidden_dims = classification_config.get("hidden_dims", [128])
                use_hidden_layer = len(hidden_dims) > 0
                head_hidden_dim = hidden_dims[0] if use_hidden_layer else 128

                head = ClassificationHead(
                    input_dim=encoder.output_dim,
                    hidden_dim=head_hidden_dim,
                    num_classes=num_classes,
                    dropout=classification_config.get("dropout", 0.1),
                    use_hidden_layer=use_hidden_layer,
                )
            else:
                head = ClassificationHead(
                    input_dim=256,
                    hidden_dim=128,
                    num_classes=num_classes,
                    dropout=0.1,
                    use_hidden_layer=True,
                )

            # Load state dicts
            encoder.load_state_dict(encoder_state_dict)
            head.load_state_dict(head_state_dict)

            # Combine into a single model for streaming
            class StreamingAttentionModel(nn.Module):
                """
                Combined encoder + head model for streaming pipeline.

                Provides the interface expected by StreamingAttentionAggregator:
                - forward(features, num_patches, return_attention) -> (logits, attention)
                """

                def __init__(self, encoder, head):
                    super().__init__()
                    self.encoder = encoder
                    self.head = head

                def forward(
                    self,
                    features: torch.Tensor,
                    num_patches: Optional[torch.Tensor] = None,
                    return_attention: bool = False,
                ):
                    """
                    Forward pass.

                    Args:
                        features: [batch_size, num_patches, feature_dim]
                        num_patches: [batch_size] number of valid patches
                        return_attention: If True, return attention weights

                    Returns:
                        logits: [batch_size, num_classes]
                        attention: [batch_size, num_patches] (if return_attention=True)
                    """
                    # Encode features
                    encoded = self.encoder(features)  # [batch_size, output_dim]

                    # Classify
                    logits = self.head(encoded)  # [batch_size, num_classes]

                    if return_attention:
                        # For simple encoder without attention mechanism,
                        # return uniform attention weights
                        batch_size, num_patches_dim, _ = features.shape
                        attention = (
                            torch.ones(batch_size, num_patches_dim, device=features.device)
                            / num_patches_dim
                        )
                        return logits, attention
                    else:
                        return logits

            model = StreamingAttentionModel(encoder, head)
            model.eval()
            model.to(self.device)

            logger.info(f"Attention model loaded successfully on {self.device}")
            return model

        except Exception as e:
            logger.error(f"Failed to load attention model: {e}", exc_info=True)
            raise RuntimeError(f"Attention model loading failed: {e}") from e

    def load_for_streaming(
        self, config: Optional[Dict[str, Any]] = None
    ) -> Tuple[nn.Module, nn.Module]:
        """
        Load both feature extractor and attention model for streaming pipeline.

        This is the main entry point for loading trained models into the
        real-time streaming system.

        Args:
            config: Optional config dictionary from checkpoint

        Returns:
            Tuple of (feature_extractor, attention_model)

        Example:
            >>> loader = CheckpointLoader("checkpoints/pcam_real/best_model.pth")
            >>> cnn_encoder, attention_model = loader.load_for_streaming()
            >>>
            >>> # Use in streaming config
            >>> from src.streaming import StreamingConfig, RealTimeWSIProcessor
            >>> config = StreamingConfig()
            >>> processor = RealTimeWSIProcessor(config)
            >>> processor._cnn_encoder = cnn_encoder
            >>> processor._attention_model = attention_model
        """
        # Load checkpoint
        checkpoint = self.load_checkpoint()

        # Use config from checkpoint if not provided
        if config is None and "config" in checkpoint:
            config = checkpoint["config"]

        # Load models
        feature_extractor = self.load_feature_extractor(checkpoint, config)
        attention_model = self.load_attention_model(checkpoint, config)

        logger.info("Successfully loaded both models for streaming")
        return feature_extractor, attention_model

    # Helper methods for dimension inference

    def _infer_feature_dim(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Infer feature dimension from feature extractor state dict."""
        # Look for final layer output dimension
        for key in state_dict.keys():
            if "fc" in key.lower() or "linear" in key.lower():
                if "weight" in key:
                    return state_dict[key].shape[0]
        return 512  # Default

    def _infer_encoder_input_dim(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Infer input dimension from encoder state dict."""
        # Look for input_proj layer (first layer in WSIEncoder)
        if "input_proj.0.weight" in state_dict:
            # Shape is [hidden_dim, input_dim]
            return state_dict["input_proj.0.weight"].shape[1]

        # Fallback: look for first layer input dimension
        for key in sorted(state_dict.keys()):
            if "weight" in key and len(state_dict[key].shape) >= 2:
                return state_dict[key].shape[1]
        return 512  # Default

    def _infer_encoder_hidden_dim(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Infer hidden dimension from encoder state dict."""
        # Look for input_proj layer (first layer in WSIEncoder)
        if "input_proj.0.weight" in state_dict:
            # Shape is [hidden_dim, input_dim]
            return state_dict["input_proj.0.weight"].shape[0]

        # Fallback: look for transformer hidden dimension
        for key in state_dict.keys():
            if "transformer" in key.lower() and "weight" in key:
                return state_dict[key].shape[0]
        return 256  # Default

    def _infer_num_classes(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Infer number of classes from head state dict."""
        # Look for final layer output dimension
        for key in reversed(list(state_dict.keys())):
            if "weight" in key:
                return state_dict[key].shape[0]
        return 2  # Default for binary classification


def load_checkpoint_for_streaming(
    checkpoint_path: str, device: str = "cuda"
) -> Tuple[nn.Module, nn.Module]:
    """
    Convenience function to load checkpoint for streaming.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on ('cuda' or 'cpu')

    Returns:
        Tuple of (feature_extractor, attention_model)

    Example:
        >>> cnn_encoder, attention_model = load_checkpoint_for_streaming(
        ...     "checkpoints/pcam_real/best_model.pth"
        ... )
    """
    loader = CheckpointLoader(checkpoint_path, device)
    return loader.load_for_streaming()
