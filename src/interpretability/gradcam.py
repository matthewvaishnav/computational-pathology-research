"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Reference:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization" (ICCV 2017)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm

from .utils import get_device, normalize_array, to_numpy

logger = logging.getLogger(__name__)


class GradCAMGenerator:
    """Generate Grad-CAM visualizations for CNN models.

    Supports ResNet, DenseNet, and EfficientNet architectures.
    Computes gradient-weighted activations at specified convolutional layers.

    Attributes:
        model: CNN feature extractor
        target_layers: List of layer names to generate CAMs for
        device: Device for computation ('cuda' or 'cpu')
        activations: Dictionary storing forward activations
        gradients: Dictionary storing backward gradients
        hooks: List of registered hooks

    Examples:
        >>> model = torchvision.models.resnet18(pretrained=True)
        >>> generator = GradCAMGenerator(model, target_layers=['layer4'])
        >>> images = torch.randn(4, 3, 224, 224)
        >>> heatmaps = generator.generate(images)
        >>> print(heatmaps['layer4'].shape)  # [4, 224, 224]
    """

    def __init__(self, model: nn.Module, target_layers: List[str], device: Optional[str] = None):
        """Initialize Grad-CAM generator.

        Args:
            model: CNN feature extractor (ResNet, DenseNet, EfficientNet)
            target_layers: List of layer names to generate CAMs for
            device: Device for computation ('cuda', 'cpu', or None for auto-detection)

        Raises:
            ValueError: If target_layers is empty or contains invalid layer names
        """
        if not target_layers:
            raise ValueError("target_layers cannot be empty")

        self.model = model
        self.target_layers = target_layers
        self.device = get_device(device)
        self.model.to(self.device)
        self.model.eval()

        # Storage for activations and gradients
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.hooks: List = []

        # Register hooks for target layers
        self._register_hooks()

        logger.info(
            f"GradCAMGenerator initialized with {len(target_layers)} target layers on {self.device}"
        )

    def _register_hooks(self):
        """Register forward and backward hooks on target layers.

        Raises:
            ValueError: If a target layer is not found in the model
        """
        for layer_name in self.target_layers:
            # Find the layer in the model
            layer = self._get_layer_by_name(layer_name)
            if layer is None:
                available_layers = self._get_available_layers()
                raise ValueError(
                    f"Layer '{layer_name}' not found in model. "
                    f"Available layers: {available_layers}"
                )

            # Register forward hook to capture activations
            def forward_hook(module, input, output, name=layer_name):
                self.activations[name] = output.detach()

            # Register backward hook to capture gradients
            def backward_hook(module, grad_input, grad_output, name=layer_name):
                self.gradients[name] = grad_output[0].detach()

            forward_handle = layer.register_forward_hook(forward_hook)
            backward_handle = layer.register_full_backward_hook(backward_hook)

            self.hooks.extend([forward_handle, backward_handle])
            logger.debug(f"Registered hooks for layer: {layer_name}")

    def _get_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """Get layer module by name.

        Args:
            layer_name: Name of the layer (e.g., 'layer4', 'features.denseblock4')

        Returns:
            Layer module or None if not found
        """
        # Handle nested layer names (e.g., 'features.denseblock4')
        parts = layer_name.split(".")
        module = self.model

        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None

        return module

    def _get_available_layers(self) -> List[str]:
        """Get list of available layer names in the model.

        Returns:
            List of layer names
        """
        layers = []
        for name, _ in self.model.named_modules():
            if name:  # Skip empty names (root module)
                layers.append(name)
        return layers[:20]  # Return first 20 for brevity

    def generate(
        self, images: torch.Tensor, class_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate Grad-CAM heatmaps for input images.

        Args:
            images: [batch, 3, H, W] input patches
            class_idx: Target class index (None for predicted class)

        Returns:
            Dictionary mapping layer names to heatmaps [batch, H, W]

        Examples:
            >>> images = torch.randn(4, 3, 224, 224)
            >>> heatmaps = generator.generate(images)
            >>> heatmaps = generator.generate(images, class_idx=1)  # Target class 1
        """
        self.model.eval()
        images = images.to(self.device)
        batch_size = images.size(0)

        # Clear previous activations and gradients
        self.activations.clear()
        self.gradients.clear()

        # Forward pass
        images.requires_grad = True
        outputs = self.model(images)

        # Determine target class
        if class_idx is None:
            class_idx = outputs.argmax(dim=1)
        elif isinstance(class_idx, int):
            class_idx = torch.full((batch_size,), class_idx, device=self.device)

        # Backward pass for each sample
        self.model.zero_grad()

        # Create one-hot encoding for target classes
        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, class_idx.unsqueeze(1), 1.0)

        # Backward pass
        outputs.backward(gradient=one_hot, retain_graph=False)

        # Generate heatmaps for each target layer
        heatmaps = {}
        for layer_name in self.target_layers:
            if layer_name in self.activations and layer_name in self.gradients:
                heatmap = self._compute_cam(
                    self.activations[layer_name],
                    self.gradients[layer_name],
                    images.size()[-2:],  # (H, W)
                )
                heatmaps[layer_name] = heatmap
            else:
                logger.warning(f"Missing activations or gradients for layer: {layer_name}")

        return heatmaps

    def _compute_cam(
        self, activations: torch.Tensor, gradients: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Compute Class Activation Map from activations and gradients.

        Args:
            activations: Forward activations [batch, channels, h, w]
            gradients: Backward gradients [batch, channels, h, w]
            target_size: Target size (H, W) for upsampling

        Returns:
            Heatmap [batch, H, W] normalized to [0, 1]
        """
        # Global average pooling of gradients to get weights
        # α_k = (1/Z) * Σ(∂y/∂A_k) where Z is spatial size
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [batch, channels, 1, 1]

        # Weighted combination of activations
        # CAM = ReLU(Σ(α_k * A_k))
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [batch, 1, h, w]
        cam = F.relu(cam)  # Apply ReLU to focus on positive contributions

        # Upsample to input resolution
        cam = F.interpolate(
            cam, size=target_size, mode="bilinear", align_corners=False
        )  # [batch, 1, H, W]

        # Remove channel dimension
        cam = cam.squeeze(1)  # [batch, H, W]

        # Normalize each heatmap in batch to [0, 1]
        batch_size = cam.size(0)
        cam = cam.view(batch_size, -1)  # [batch, H*W]
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]

        # Avoid division by zero
        cam_range = cam_max - cam_min
        cam_range = torch.where(cam_range == 0, torch.ones_like(cam_range), cam_range)

        cam = (cam - cam_min) / cam_range
        cam = cam.view(batch_size, *target_size)  # [batch, H, W]

        return cam

    def overlay_heatmap(
        self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap: str = "jet"
    ) -> np.ndarray:
        """Overlay heatmap on original image.

        Args:
            image: Original image [H, W, 3] in range [0, 1] or [0, 255]
            heatmap: Grad-CAM heatmap [H, W] in range [0, 1]
            alpha: Transparency (0=transparent heatmap, 1=opaque heatmap)
            colormap: Matplotlib colormap name ('jet', 'viridis', 'plasma', etc.)

        Returns:
            Overlaid image [H, W, 3] in range [0, 1]

        Examples:
            >>> image = np.random.rand(224, 224, 3)
            >>> heatmap = np.random.rand(224, 224)
            >>> overlaid = generator.overlay_heatmap(image, heatmap, alpha=0.5)
        """
        # Ensure image is in [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0

        # Ensure heatmap is 2D
        if heatmap.ndim != 2:
            raise ValueError(f"Heatmap must be 2D, got shape {heatmap.shape}")

        # Ensure image and heatmap have same spatial dimensions
        if image.shape[:2] != heatmap.shape:
            raise ValueError(
                f"Image shape {image.shape[:2]} does not match heatmap shape {heatmap.shape}"
            )

        # Apply colormap to heatmap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # [H, W, 3], drop alpha channel

        # Blend image and heatmap
        overlaid = (1 - alpha) * image + alpha * heatmap_colored

        # Clip to [0, 1] range
        overlaid = np.clip(overlaid, 0, 1)

        return overlaid.astype(np.float32)

    def save_visualization(
        self, image: np.ndarray, heatmap: np.ndarray, output_path: Path, dpi: int = 300
    ) -> Path:
        """Save Grad-CAM visualization to file.

        Args:
            image: Original image [H, W, 3]
            heatmap: Grad-CAM heatmap [H, W]
            output_path: Output file path
            dpi: Resolution (default 300 for publication quality)

        Returns:
            Path to saved visualization

        Examples:
            >>> output_path = Path('gradcam_visualization.png')
            >>> saved_path = generator.save_visualization(image, heatmap, output_path)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create figure with original image and overlaid heatmap
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original image
        axes[0].imshow(image if image.max() <= 1.0 else image / 255.0)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Overlaid heatmap
        overlaid = self.overlay_heatmap(image, heatmap)
        axes[1].imshow(overlaid)
        axes[1].set_title("Grad-CAM Overlay")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved Grad-CAM visualization to {output_path}")
        return output_path

    def __del__(self):
        """Remove hooks when object is destroyed."""
        for hook in self.hooks:
            hook.remove()
