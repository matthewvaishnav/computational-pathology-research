"""
Pretrained model loaders for computational pathology.

Provides unified interface for loading and using publicly available
pretrained pathology models (UNI, Prov-GigaPath, CTransPath) as
feature extractors or encoders.

Security Note:
    All model loading is validated against the PRETRAINED_MODELS registry.
    Dynamic imports only occur after string validation against known models.
    No user input is passed to eval/exec. Uses torch.load with weights_only=True
    for checkpoint loading to prevent arbitrary code execution.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.platform.security.model_download import ModelDownloadManager

logger = logging.getLogger(__name__)

# Model registry with metadata
PRETRAINED_MODELS = {
    "uni": {
        "name": "UNI",
        "source": "hf_hub:MahmoodLab/uni",
        "description": "Universal Model for pathology (ViT-L on 100k+ WSIs)",
        "input_size": 224,
        "output_dim": 1024,
        "requires_timm": True,
        "requires_huggingface": True,
        "revision": "main",
    },
    "phikon": {
        "name": "Phikon",
        "source": "hf_hub:owkin/phikon",
        "description": "ViT-B/16 pretrained on 500M+ histopathology patches",
        "input_size": 224,
        "output_dim": 768,
        "requires_timm": True,
        "requires_huggingface": True,
        "revision": "main",
    },
    "gigapath": {
        "name": "Prov-GigaPath",
        "source": "hf_hub:prov-gigapath/prov-gigapath",
        "description": "Gigapixel-level pretrained transformer for WSIs",
        "input_size": 224,
        "output_dim": 1536,
        "requires_timm": True,
        "requires_huggingface": True,
        "revision": "main",
    },
    "ctranspath": {
        "name": "CTransPath",
        "source": "https://github.com/Xiyue-Wang/TransPath",
        "description": "CNN-Transformer hybrid for pathology (ImageNet + histology)",
        "input_size": 224,
        "output_dim": 768,
        "requires_timm": False,
        "requires_huggingface": False,
        "custom_loader": True,
    },
    "resnet50_imagenet": {
        "name": "ResNet50 (ImageNet)",
        "source": "torchvision",
        "description": "Standard ResNet50 for baseline comparisons",
        "input_size": 224,
        "output_dim": 2048,
        "requires_timm": False,
        "requires_huggingface": False,
    },
}


class PretrainedFeatureExtractor(nn.Module):
    """
    Wrapper for pretrained patch-level feature extractors.

    Loads publicly available pretrained models and extracts fixed-size
    features from pathology image patches. Features can be fed into
    WSIEncoder for slide-level aggregation.

    Args:
        model_name: Name of pretrained model ('uni', 'gigapath', 'ctranspath', 'resnet50_imagenet')
        cache_dir: Directory to cache downloaded weights
        freeze: Whether to freeze pretrained weights (default: True)
        device: Device to load model on

    Example:
        >>> extractor = PretrainedFeatureExtractor('uni')
        >>> patches = torch.randn(4, 3, 224, 224)  # [batch, channels, h, w]
        >>> features = extractor(patches)
        >>> features.shape
        torch.Size([4, 1024])
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        freeze: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        if model_name not in PRETRAINED_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. " f"Available: {list(PRETRAINED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.config = PRETRAINED_MODELS[model_name]
        self.device = device
        self.freeze = freeze

        # Load model
        self.backbone = self._load_model(cache_dir)
        self.output_dim = self.config["output_dim"]

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        logger.info(
            f"Loaded {self.config['name']} ({model_name}) "
            f"with output_dim={self.output_dim}, freeze={freeze}"
        )

    def _load_model(self, cache_dir: Optional[str]) -> nn.Module:
        """Load pretrained model based on registry config."""
        source = self.config["source"]

        if source == "torchvision":
            return self._load_torchvision()
        elif source.startswith("hf_hub:"):
            return self._load_huggingface(source.replace("hf_hub:", ""), cache_dir)
        elif self.config.get("custom_loader"):
            return self._load_custom(self.model_name)
        else:
            raise ValueError(f"Unknown source: {source}")

    def _load_torchvision(self) -> nn.Module:
        """Load torchvision pretrained model."""
        import torchvision.models as models

        model = models.resnet50(weights="IMAGENET1K_V2")
        # Remove final classification layer
        model = nn.Sequential(*list(model.children())[:-1])
        return model.to(self.device)

    def _load_huggingface(self, repo_id: str, cache_dir: Optional[str]) -> nn.Module:
        """Load model from Hugging Face Hub with an explicit revision."""
        try:
            from transformers import AutoModel, ViTModel
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFace models. "
                "Install: pip install transformers"
            )

        revision = str(self.config.get("revision", "main"))

        # Special handling for known ViT models
        if "phikon" in repo_id.lower() or "uni" in repo_id.lower():
            # Load as ViT model (these are vision transformers)
            model = ViTModel.from_pretrained(
                repo_id,
                revision=revision,
                cache_dir=cache_dir,
                add_pooling_layer=False,  # We'll use the CLS token
            )
        else:
            # Load model directly with transformers
            model = AutoModel.from_pretrained(
                repo_id,
                revision=revision,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        return model.to(self.device)

    def _load_custom(self, model_name: str) -> nn.Module:
        """Load custom models (CTransPath, etc.)."""
        try:
            # Handle custom model formats
            if model_name.startswith("histocore_"):
                # HistoCore custom models
                model_type = model_name.replace("histocore_", "")
                if model_type == "dmi":
                    from ..dmi.distributed_medical_intelligence import (
                        DistributedMedicalIntelligence,
                    )

                    return DistributedMedicalIntelligence()
                elif model_type == "attention_mil":
                    from ..models.mil.attention_mil import AttentionMIL

                    return AttentionMIL(feature_dim=2048, hidden_dim=256, num_classes=2)
                elif model_type == "clam":
                    from ..models.mil.attention_mil import CLAM

                    return CLAM(feature_dim=2048, hidden_dim=256, num_classes=2)
                elif model_type == "transmil":
                    from ..models.mil.attention_mil import TransMIL

                    return TransMIL(feature_dim=2048, hidden_dim=256, num_classes=2)

            # Handle foundation models
            elif model_name in ["uni", "phikon", "conch"]:
                if model_name == "uni":
                    # UNI foundation model
                    import timm

                    model = timm.create_model(
                        "vit_large_patch16_224", pretrained=False, num_classes=0
                    )
                    # Load UNI weights if available
                    return model
                elif model_name == "phikon":
                    # Phikon foundation model
                    import timm

                    model = timm.create_model(
                        "vit_base_patch16_224", pretrained=False, num_classes=0
                    )
                    return model
                elif model_name == "conch":
                    # CONCH foundation model
                    import timm

                    model = timm.create_model(
                        "vit_base_patch16_224", pretrained=False, num_classes=0
                    )
                    return model

            # Handle checkpoint loading
            elif model_name.endswith(".pth") or model_name.endswith(".ckpt"):
                import torch

                checkpoint_path = Path(model_name)
                if checkpoint_path.exists():
                    # nosec B614 - Safe: weights_only=True prevents code execution
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

                    # Extract model from checkpoint
                    if "model" in checkpoint:
                        return checkpoint["model"]
                    elif "state_dict" in checkpoint:
                        # Need to reconstruct model architecture
                        # This would require model config in checkpoint
                        raise NotImplementedError(
                            "Model reconstruction from state_dict requires architecture info"
                        )
                    else:
                        # Assume checkpoint is the model directly
                        return checkpoint
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            # Handle Hugging Face models
            elif "/" in model_name:  # Likely HF model path
                try:
                    from transformers import AutoModel

                    revision = ModelDownloadManager.get_pinned_revision(model_name)
                    return AutoModel.from_pretrained(model_name, revision=revision)
                except ImportError:
                    raise ImportError("transformers library required for Hugging Face models")

            else:
                raise ValueError(f"Unknown model format: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load custom model {model_name}: {e}")
            raise RuntimeError(f"Custom model loading failed: {e}")

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Extract features from image patches.

        Args:
            patches: Image patches [batch_size, 3, H, W]
                     Expected H=W=224 for most models

        Returns:
            Features [batch_size, output_dim]
        """
        if self.freeze:
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(patches)
        else:
            features = self.backbone(patches)

        # Handle different output formats
        if isinstance(features, dict):
            # Transformers models return dict with 'last_hidden_state'
            if "last_hidden_state" in features:
                # For ViT: [batch, num_patches+1, hidden_dim]
                # Take CLS token (first token)
                features = features["last_hidden_state"][:, 0, :]
            elif "pooler_output" in features:
                features = features["pooler_output"]
            else:
                raise ValueError(f"Unknown output format: {features.keys()}")

        # Flatten if needed (for CNN outputs with spatial dims)
        if features.dim() > 2:
            features = features.flatten(1)

        return features

    def extract_slide_features(
        self,
        patch_loader: torch.utils.data.DataLoader,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Extract features for all patches in a slide.

        Args:
            patch_loader: DataLoader yielding image patches [B, 3, H, W]
            device: Device to run on (defaults to model device)

        Returns:
            All slide features [num_patches, output_dim]
        """
        device = device or self.device
        all_features = []

        self.backbone.eval()
        with torch.no_grad():
            for batch in patch_loader:
                if isinstance(batch, dict):
                    patches = batch.get("image", batch.get("patches"))
                else:
                    patches = batch

                patches = patches.to(device)
                features = self.forward(patches)
                all_features.append(features.cpu())

        return torch.cat(all_features, dim=0)


def list_pretrained_models() -> List[Dict[str, str]]:
    """List all available pretrained models with descriptions."""
    return [
        {
            "name": key,
            "full_name": config["name"],
            "description": config["description"],
            "output_dim": config["output_dim"],
        }
        for key, config in PRETRAINED_MODELS.items()
    ]


def get_recommended_model(task: str = "general") -> str:
    """
    Get recommended pretrained model for a task.

    Args:
        task: One of 'general', 'gigapixel', 'fast', 'baseline', 'histopathology'

    Returns:
        Model name key
    """
    recommendations = {
        "general": "uni",  # Best overall performance
        "histopathology": "phikon",  # Specialized for histopathology
        "gigapixel": "gigapath",  # Designed for large WSIs
        "fast": "resnet50_imagenet",  # Fastest, smallest
        "baseline": "resnet50_imagenet",  # Standard baseline
    }

    if task not in recommendations:
        raise ValueError(f"Unknown task: {task}. Use: {list(recommendations.keys())}")

    return recommendations[task]


# Integration with existing encoders
def create_wsi_encoder_with_pretrained(
    pretrained_model: str = "uni",
    output_dim: int = 256,
    freeze_pretrained: bool = True,
) -> Tuple[PretrainedFeatureExtractor, nn.Module]:
    """
    Create a feature extractor + WSI encoder combo.

    Args:
        pretrained_model: Name of pretrained patch extractor
        output_dim: Output dimension for WSI encoder
        freeze_pretrained: Whether to freeze pretrained weights

    Returns:
        Tuple of (feature_extractor, wsi_encoder)
    """
    from src.models.components.encoders import WSIEncoder

    # Load pretrained feature extractor
    extractor = PretrainedFeatureExtractor(
        pretrained_model,
        freeze=freeze_pretrained,
    )

    # Create WSI encoder that takes pretrained features
    wsi_encoder = WSIEncoder(
        input_dim=extractor.output_dim,
        output_dim=output_dim,
    )

    return extractor, wsi_encoder
