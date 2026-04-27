"""
Pretrained model loaders for computational pathology.

Provides unified interface for loading and using publicly available
pretrained pathology models (UNI, Prov-GigaPath, CTransPath) as
feature extractors or encoders.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

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
    },
    "gigapath": {
        "name": "Prov-GigaPath",
        "source": "hf_hub:prov-gigapath/prov-gigapath",
        "description": "Gigapixel-level pretrained transformer for WSIs",
        "input_size": 224,
        "output_dim": 1536,
        "requires_timm": True,
        "requires_huggingface": True,
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
            return self._load_custom()
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
        """Load model from HuggingFace Hub via timm."""
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for HuggingFace models. " "Install: pip install timm"
            )

        model = timm.create_model(
            f"hf_hub:{repo_id}",
            pretrained=True,
            cache_dir=cache_dir,
        )
        return model.to(self.device)

    def _load_custom(self) -> nn.Module:
        """Load custom models (CTransPath, Phikon, UNI, etc.)."""
        
        if self.model_name.lower() == 'ctranspath':
            return self._load_ctranspath()
        elif self.model_name.lower() == 'phikon':
            return self._load_phikon()
        elif self.model_name.lower() == 'uni':
            return self._load_uni()
        elif self.model_name.lower() == 'conch':
            return self._load_conch()
        else:
            raise NotImplementedError(
                f"Custom loader for {self.model_name} not yet implemented. "
                "Supported custom models: ctranspath, phikon, uni, conch"
            )
    
    def _load_ctranspath(self) -> nn.Module:
        """Load CTransPath model."""
        try:
            import timm
            
            # CTransPath is based on Swin Transformer
            model = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=False,
                num_classes=0,  # Remove classification head
                global_pool=''  # Remove global pooling
            )
            
            # Try to load CTransPath weights if available
            weights_path = f"{self.cache_dir}/ctranspath_weights.pth"
            if os.path.exists(weights_path):
                logger.info(f"Loading CTransPath weights from {weights_path}")
                checkpoint = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(checkpoint, strict=False)
            else:
                logger.warning(
                    f"CTransPath weights not found at {weights_path}. "
                    "Download from: https://github.com/Xiyue-Wang/TransPath"
                )
            
            return model
            
        except ImportError:
            raise ImportError("CTransPath requires timm. Install with: pip install timm")
    
    def _load_phikon(self) -> nn.Module:
        """Load Phikon model."""
        try:
            import timm
            
            # Phikon is based on ViT
            model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=0,
                global_pool=''
            )
            
            # Try to load Phikon weights if available
            weights_path = f"{self.cache_dir}/phikon_weights.pth"
            if os.path.exists(weights_path):
                logger.info(f"Loading Phikon weights from {weights_path}")
                checkpoint = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(checkpoint, strict=False)
            else:
                logger.warning(
                    f"Phikon weights not found at {weights_path}. "
                    "Download from: https://huggingface.co/owkin/phikon"
                )
            
            return model
            
        except ImportError:
            raise ImportError("Phikon requires timm. Install with: pip install timm")
    
    def _load_uni(self) -> nn.Module:
        """Load UNI (Universal Pathology Foundation Model)."""
        try:
            import timm
            
            # UNI is based on DINOv2 ViT
            model = timm.create_model(
                'vit_large_patch16_224',
                pretrained=False,
                num_classes=0,
                global_pool=''
            )
            
            # Try to load UNI weights if available
            weights_path = f"{self.cache_dir}/uni_weights.pth"
            if os.path.exists(weights_path):
                logger.info(f"Loading UNI weights from {weights_path}")
                checkpoint = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(checkpoint, strict=False)
            else:
                logger.warning(
                    f"UNI weights not found at {weights_path}. "
                    "Download from: https://huggingface.co/MahmoodLab/UNI"
                )
            
            return model
            
        except ImportError:
            raise ImportError("UNI requires timm. Install with: pip install timm")
    
    def _load_conch(self) -> nn.Module:
        """Load CONCH model."""
        try:
            import timm
            
            # CONCH is based on ViT with contrastive learning
            model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=0,
                global_pool=''
            )
            
            # Try to load CONCH weights if available
            weights_path = f"{self.cache_dir}/conch_weights.pth"
            if os.path.exists(weights_path):
                logger.info(f"Loading CONCH weights from {weights_path}")
                checkpoint = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(checkpoint, strict=False)
            else:
                logger.warning(
                    f"CONCH weights not found at {weights_path}. "
                    "Download from: https://huggingface.co/MahmoodLab/CONCH"
                )
            
            return model
            
        except ImportError:
            raise ImportError("CONCH requires timm. Install with: pip install timm")

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
        task: One of 'general', 'gigapixel', 'fast', 'baseline'

    Returns:
        Model name key
    """
    recommendations = {
        "general": "uni",  # Best overall performance
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
    from .encoders import WSIEncoder

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
