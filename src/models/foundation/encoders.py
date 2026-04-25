"""Foundation model encoder implementations.

Provides unified interface for histopathology foundation models:
- Phikon (Owkin): ViT-B/16, Apache 2.0, 768-dim
- UNI (Mahmood Lab): ViT-L/16, CC-BY-NC-ND 4.0, 1024-dim
- CONCH (Mahmood Lab): ViT-B/16, CC-BY-NC-ND 4.0, 512-dim

All encoders are drop-in replacements for ResNetFeatureExtractor.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class FoundationModelEncoder(ABC, nn.Module):
    """Base class for all foundation model encoders.

    Drop-in replacement for ResNetFeatureExtractor with unified interface.
    All subclasses output (batch_size, feature_dim) tensors.

    Args:
        freeze: Whether to freeze encoder weights (default: True)
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.freeze = freeze
        self._feature_dim: int = 0

    @property
    def feature_dim(self) -> int:
        """Feature dimension of encoder output."""
        return self._feature_dim

    @abstractmethod
    def _build_model(self) -> None:
        """Build the underlying model architecture."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from image patches.

        Args:
            x: [B, 3, H, W] normalized image patches (ImageNet stats)

        Returns:
            [B, feature_dim] patch embeddings
        """
        if self.freeze:
            with torch.no_grad():
                return self._encode(x)
        return self._encode(x)

    @abstractmethod
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Internal encoding implementation."""
        pass

    # Legacy interface compatibility
    def extract_features(self, patches: torch.Tensor) -> torch.Tensor:
        """Legacy interface for compatibility with existing code."""
        return self.forward(patches)


class PhikonEncoder(FoundationModelEncoder):
    """Owkin Phikon encoder.

    ViT-B/16 pretrained on 40K TCGA WSIs via iBOT (masked image modeling).

    - License: Apache 2.0 (commercial use allowed)
    - Feature dim: 768
    - Input: 224x224 patches @ 20x magnification
    - HuggingFace: owkin/phikon (no access gate)

    Args:
        freeze: Whether to freeze encoder weights (default: True)
    """

    def __init__(self, freeze: bool = True):
        super().__init__(freeze=freeze)
        self._feature_dim = 768
        self._build_model()

    def _build_model(self) -> None:
        """Load Phikon model from HuggingFace Hub using transformers."""
        try:
            from transformers import ViTModel
        except ImportError:
            raise ImportError(
                "transformers required for Phikon. "
                "Install with: pip install transformers>=4.37.0"
            )

        # Load Phikon ViT-B/16 model from HuggingFace
        self.model = ViTModel.from_pretrained(
            "owkin/phikon",
            add_pooling_layer=False,  # We'll use CLS token directly
        )

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode patches to 768-dim features."""
        # HuggingFace ViT returns last_hidden_state [B, num_patches+1, 768]
        # Extract CLS token (first token) as patch embedding
        outputs = self.model(x)
        return outputs.last_hidden_state[:, 0, :]  # [B, 768]


class UNIEncoder(FoundationModelEncoder):
    """UNI (Universal Network for Imaging) encoder.

    ViT-L/16 pretrained on 100K+ WSIs via DINOv2.

    - License: CC-BY-NC-ND 4.0 (non-commercial only)
    - Feature dim: 1024
    - Input: 224x224 patches @ 20x/40x magnification
    - HuggingFace: MahmoodLab/UNI (gated access - request required)

    Requires: huggingface-cli login + access approval

    Args:
        freeze: Whether to freeze encoder weights (default: True)
        local_dir: Optional path to cached checkpoint
    """

    def __init__(self, freeze: bool = True, local_dir: Optional[str] = None):
        super().__init__(freeze=freeze)
        self._feature_dim = 1024
        self.local_dir = local_dir
        self._build_model()

    def _build_model(self) -> None:
        """Load UNI model from HuggingFace Hub."""
        try:
            import timm
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "timm and huggingface_hub required for UNI. "
                "Install with: pip install timm>=0.9.12 huggingface_hub>=0.20.0"
            )

        import os

        if self.local_dir:
            # Use cached local checkpoint
            checkpoint_path = os.path.join(self.local_dir, "pytorch_model.bin")
        else:
            # Download from HuggingFace (requires authentication)
            checkpoint_path = hf_hub_download(
                "MahmoodLab/UNI",
                filename="pytorch_model.bin",
                use_auth_token=True,
            )

        # Create ViT-L/16 architecture
        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )

        # Load pretrained weights
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode patches to 1024-dim features."""
        return self.model(x)  # [B, 1024]


class CONCHEncoder(FoundationModelEncoder):
    """CONCH (Contrastive Learning from Captions about Histopathology) encoder.

    ViT-B/16 vision encoder trained on 1.17M image-caption pairs.

    - License: CC-BY-NC-ND 4.0 (non-commercial only)
    - Feature dim: 512
    - Input: 224x224 patches @ 20x magnification
    - HuggingFace: MahmoodLab/CONCH (gated access - request required)
    - Unique capability: Zero-shot classification via text prompts

    Requires: huggingface-cli login + access approval

    Args:
        freeze: Whether to freeze encoder weights (default: True)
    """

    def __init__(self, freeze: bool = True):
        super().__init__(freeze=freeze)
        self._feature_dim = 512
        self._build_model()

    def _build_model(self) -> None:
        """Load CONCH model from HuggingFace Hub."""
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers required for CONCH. " "Install with: pip install transformers>=4.37.0"
            )

        self.model = AutoModel.from_pretrained(
            "MahmoodLab/CONCH",
            trust_remote_code=True,
        )
        self.vision_encoder = self.model.visual

        if self.freeze:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode patches to 512-dim features."""
        return self.vision_encoder(x)  # [B, 512]

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode diagnostic text prompts for zero-shot classification.

        Args:
            texts: List of text prompts (e.g., ["tumor tissue", "normal tissue"])

        Returns:
            [len(texts), 512] text embeddings
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("transformers required for text encoding")

        tokenizer = AutoTokenizer.from_pretrained("MahmoodLab/CONCH")
        tokens = tokenizer(texts, return_tensors="pt", padding=True)
        return self.model.encode_text(tokens)


# Model registry
_REGISTRY = {
    "phikon": PhikonEncoder,
    "uni": UNIEncoder,
    "conch": CONCHEncoder,
}


def load_foundation_model(
    model_name: str,
    freeze: bool = True,
    **kwargs,
) -> FoundationModelEncoder:
    """Factory function to load foundation model encoders.

    Args:
        model_name: One of 'phikon', 'uni', 'conch'
        freeze: Whether to freeze encoder weights (default: True)
        **kwargs: Model-specific kwargs (e.g., local_dir for UNI)

    Returns:
        Configured FoundationModelEncoder ready for feature extraction

    Example:
        >>> encoder = load_foundation_model('phikon', freeze=True)
        >>> features = encoder.extract_features(patches)  # [B, 768]

    Raises:
        ValueError: If model_name is not recognized
        ImportError: If required dependencies are missing
    """
    if model_name not in _REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_REGISTRY.keys())}")

    return _REGISTRY[model_name](freeze=freeze, **kwargs)
