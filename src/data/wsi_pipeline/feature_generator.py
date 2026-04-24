"""
Feature Generator for WSI processing.

This module provides feature extraction functionality using pretrained CNN encoders.
Supports multiple architectures (ResNet, DenseNet, EfficientNet) with automatic
GPU/CPU device selection, memory-efficient streaming extraction, and speed optimizations
including mixed precision training and optimized preprocessing.
"""

import logging
from typing import Iterator, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from .exceptions import ProcessingError, ResourceError

logger = logging.getLogger(__name__)

# Check for mixed precision support
try:
    from torch.cuda.amp import autocast

    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    autocast = None


class FeatureGenerator:
    """
    Generate feature embeddings from patches using pretrained CNN encoders.

    Supports multiple pretrained architectures with automatic device selection
    and GPU memory management. Features are extracted from the penultimate layer
    (before classification head) for use in MIL models.

    Args:
        encoder_name: Name of pretrained encoder ("resnet50", "densenet121", "efficientnet_b0")
        pretrained: Whether to use pretrained weights
        device: Device to use ("auto", "cuda", "cpu")
        batch_size: Batch size for feature extraction

    Example:
        >>> generator = FeatureGenerator(encoder_name="resnet50", device="auto")
        >>> features = generator.extract_features(patches)
        >>> print(f"Feature dimension: {generator.feature_dim}")
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        pretrained: bool = True,
        device: str = "auto",
        batch_size: int = 32,
        use_mixed_precision: bool = True,
        compile_model: bool = True,
    ):
        """
        Initialize feature generator with pretrained encoder.

        Args:
            encoder_name: Encoder architecture name
            pretrained: Use pretrained ImageNet weights
            device: Device selection ("auto", "cuda", "cpu")
            batch_size: Batch size for GPU processing
            use_mixed_precision: Use FP16 for faster inference (GPU only)
            compile_model: Use torch.compile for optimization (PyTorch 2.0+)

        Raises:
            ValueError: If encoder_name is not supported
            ProcessingError: If encoder initialization fails
        """
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision and AMP_AVAILABLE
        self.compile_model = compile_model

        # Determine device
        self.device = self._select_device(device)

        # Initialize encoder
        self.encoder = self._load_encoder()
        self.encoder.eval()  # Set to evaluation mode
        self.encoder.to(self.device)

        # Apply optimizations
        self._apply_optimizations()

        # Setup preprocessing transforms (optimized)
        self._setup_transforms()

        # Determine feature dimension
        self._feature_dim = self._get_feature_dimension()

        logger.info(
            f"Initialized FeatureGenerator: encoder={encoder_name}, "
            f"device={self.device}, feature_dim={self._feature_dim}, "
            f"batch_size={batch_size}, mixed_precision={self.use_mixed_precision}, "
            f"compiled={self.compile_model}"
        )

    def _select_device(self, device: str) -> torch.device:
        """
        Select computation device.

        Args:
            device: Device string ("auto", "cuda", "cpu")

        Returns:
            torch.device object
        """
        if device == "auto":
            if torch.cuda.is_available():
                selected_device = torch.device("cuda")
                logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                selected_device = torch.device("cpu")
                logger.info("Auto-selected CPU device (CUDA not available)")
        elif device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                selected_device = torch.device("cpu")
            else:
                selected_device = torch.device("cuda")
        else:
            selected_device = torch.device("cpu")

        return selected_device

    def _apply_optimizations(self) -> None:
        """Apply performance optimizations to the model."""
        # Enable mixed precision if available and requested
        if self.use_mixed_precision and self.device.type == "cuda":
            logger.info("Enabled mixed precision (FP16) for faster inference")

        # Compile model for optimization (PyTorch 2.0+)
        if self.compile_model:
            try:
                if hasattr(torch, "compile"):
                    # Check if we're on Windows and skip compilation if compiler issues
                    import platform

                    if platform.system() == "Windows":
                        logger.info("Skipping torch.compile on Windows to avoid compiler issues")
                    else:
                        self.encoder = torch.compile(self.encoder, mode="reduce-overhead")
                        logger.info("Model compiled with torch.compile for optimization")
                else:
                    logger.debug("torch.compile not available, skipping compilation")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}, continuing without compilation")

        # Set optimal settings for inference
        if self.device.type == "cuda":
            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            # Enable TensorFloat-32 for faster computation on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.debug("Enabled cuDNN optimizations")

    def _setup_transforms(self) -> None:
        """Setup optimized preprocessing transforms."""
        # Use optimized transforms for faster preprocessing
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Pre-compute normalization values for faster processing
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def _load_encoder(self) -> nn.Module:
        """
        Load pretrained encoder and remove classification head.

        Returns:
            Encoder model without classification head

        Raises:
            ValueError: If encoder_name is not supported
            ProcessingError: If encoder loading fails
        """
        try:
            # Determine weights parameter for torchvision models
            weights = "DEFAULT" if self.pretrained else None

            if self.encoder_name == "resnet50":
                model = models.resnet50(weights=weights)
                # Remove final classification layer
                encoder = nn.Sequential(*list(model.children())[:-1])

            elif self.encoder_name == "densenet121":
                model = models.densenet121(weights=weights)
                # Remove final classification layer
                encoder = nn.Sequential(
                    model.features,
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )

            elif self.encoder_name == "efficientnet_b0":
                model = models.efficientnet_b0(weights=weights)
                # Remove final classification layer
                encoder = nn.Sequential(
                    model.features,
                    model.avgpool,
                )

            else:
                # Try to load from torchvision or timm
                encoder = self._load_custom_encoder()

            return encoder

        except Exception as e:
            raise ProcessingError(f"Failed to load encoder '{self.encoder_name}': {e}")

    def _load_custom_encoder(self) -> nn.Module:
        """
        Load custom encoder from torchvision or timm.

        Returns:
            Custom encoder model

        Raises:
            ValueError: If encoder is not found
        """
        # Try torchvision first
        try:
            model_fn = getattr(models, self.encoder_name, None)
            if model_fn is not None:
                weights = "DEFAULT" if self.pretrained else None
                model = model_fn(weights=weights)
                # Remove classification head (last layer)
                encoder = nn.Sequential(*list(model.children())[:-1])
                logger.info(f"Loaded custom encoder from torchvision: {self.encoder_name}")
                return encoder
        except Exception as e:
            logger.debug(f"Failed to load from torchvision: {e}")

        # Try timm
        try:
            import timm

            model = timm.create_model(
                self.encoder_name,
                pretrained=self.pretrained,
                num_classes=0,  # Remove classification head
            )
            logger.info(f"Loaded custom encoder from timm: {self.encoder_name}")
            return model
        except ImportError:
            raise ValueError(
                f"Encoder '{self.encoder_name}' not found in torchvision. "
                f"Install timm for additional encoders: pip install timm"
            )
        except Exception as e:
            raise ValueError(f"Encoder '{self.encoder_name}' not found in torchvision or timm: {e}")

    def _get_feature_dimension(self) -> int:
        """
        Determine feature dimension by running a test forward pass.

        Returns:
            Feature dimension (number of features per patch)
        """
        try:
            # Create dummy input (batch_size=1, channels=3, height=224, width=224)
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

            with torch.no_grad():
                features = self.encoder(dummy_input)

            # Flatten if needed
            if features.dim() > 2:
                features = features.flatten(1)

            return features.shape[1]

        except Exception as e:
            logger.warning(f"Failed to determine feature dimension: {e}")
            # Return default dimensions for known encoders
            default_dims = {
                "resnet50": 2048,
                "densenet121": 1024,
                "efficientnet_b0": 1280,
            }
            return default_dims.get(self.encoder_name, 2048)

    @property
    def feature_dim(self) -> int:
        """
        Get feature dimension of encoder.

        Returns:
            Number of features per patch
        """
        return self._feature_dim

    def extract_features(
        self,
        patches: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """
        Extract features from batch of patches with speed optimizations.

        Uses mixed precision (FP16) when available for faster inference,
        optimized preprocessing, and efficient memory management.

        Args:
            patches: Batch of patches as tensor (N, C, H, W) or numpy array (N, H, W, C)

        Returns:
            Feature tensor of shape (N, feature_dim) as float32

        Raises:
            ProcessingError: If feature extraction fails
            ResourceError: If GPU memory is exhausted

        Example:
            >>> patches = torch.randn(32, 3, 224, 224)
            >>> features = generator.extract_features(patches)
            >>> features.shape
            torch.Size([32, 2048])
        """
        try:
            # Convert numpy to tensor if needed (optimized preprocessing)
            if isinstance(patches, np.ndarray):
                patches = self._preprocess_numpy_patches_optimized(patches)
            else:
                patches = patches.to(self.device, non_blocking=True)

            # Extract features with mixed precision if available
            with torch.no_grad():
                if self.use_mixed_precision and self.device.type == "cuda":
                    with autocast():
                        features = self.encoder(patches)
                else:
                    features = self.encoder(patches)

            # Flatten if needed
            if features.dim() > 2:
                features = features.flatten(1)

            # Convert to float32 (ensure consistent output type)
            features = features.float()

            return features

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # GPU OOM - try to recover
                logger.warning(f"GPU OOM during feature extraction: {e}")
                torch.cuda.empty_cache()
                raise ResourceError(
                    f"GPU memory exhausted. Try reducing batch_size (current: {self.batch_size})"
                )
            else:
                raise ProcessingError(f"Feature extraction failed: {e}")

        except Exception as e:
            raise ProcessingError(f"Feature extraction failed: {e}")

    def _preprocess_numpy_patches_optimized(self, patches: np.ndarray) -> torch.Tensor:
        """
        Optimized preprocessing of numpy patches for feature extraction.

        Uses vectorized operations and pre-computed normalization values
        for faster preprocessing.

        Args:
            patches: Numpy array of shape (N, H, W, C) with values in [0, 255]

        Returns:
            Preprocessed tensor of shape (N, C, H, W) with normalized values
        """
        # Convert to float and normalize to [0, 1] in one operation
        patches = patches.astype(np.float32, copy=False) * (1.0 / 255.0)

        # Convert from (N, H, W, C) to (N, C, H, W)
        patches = np.transpose(patches, (0, 3, 1, 2))

        # Convert to tensor with non-blocking transfer
        patches_tensor = torch.from_numpy(patches).to(self.device, non_blocking=True)

        # Use pre-computed normalization values for faster processing
        patches_tensor = (patches_tensor - self.norm_mean) / self.norm_std

        return patches_tensor

    def extract_features_streaming(
        self,
        patch_iterator: Iterator[np.ndarray],
    ) -> Iterator[torch.Tensor]:
        """
        Stream feature extraction for memory efficiency.

        Processes patches in batches without accumulating all patches in memory.

        Args:
            patch_iterator: Iterator yielding individual patches (H, W, C)

        Yields:
            Feature tensors of shape (feature_dim,) as float32

        Example:
            >>> for patch, coord in extractor.extract_patches_streaming(reader, coords):
            ...     for features in generator.extract_features_streaming(iter([patch])):
            ...         save_features(features, coord)
        """
        batch = []

        for patch in patch_iterator:
            batch.append(patch)

            # Process batch when full
            if len(batch) >= self.batch_size:
                batch_array = np.stack(batch, axis=0)
                batch_features = self.extract_features(batch_array)

                # Yield individual features
                for i in range(batch_features.shape[0]):
                    yield batch_features[i]

                # Clear batch
                batch = []

                # Clear GPU cache periodically
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Process remaining patches
        if batch:
            batch_array = np.stack(batch, axis=0)
            batch_features = self.extract_features(batch_array)

            for i in range(batch_features.shape[0]):
                yield batch_features[i]

            # Final GPU cache clear
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def reduce_batch_size(self, factor: float = 0.5) -> None:
        """
        Reduce batch size to handle GPU memory constraints.

        Args:
            factor: Reduction factor (e.g., 0.5 to halve batch size)

        Example:
            >>> try:
            ...     features = generator.extract_features(large_batch)
            ... except ResourceError:
            ...     generator.reduce_batch_size(0.5)
            ...     features = generator.extract_features(large_batch)
        """
        new_batch_size = max(1, int(self.batch_size * factor))
        logger.warning(f"Reducing batch size from {self.batch_size} to {new_batch_size}")
        self.batch_size = new_batch_size

    def fallback_to_cpu(self) -> None:
        """
        Fallback to CPU processing when GPU memory is exhausted.

        Example:
            >>> try:
            ...     features = generator.extract_features(patches)
            ... except ResourceError:
            ...     generator.fallback_to_cpu()
            ...     features = generator.extract_features(patches)
        """
        if self.device.type == "cuda":
            logger.warning("Falling back to CPU processing due to GPU memory issues")
            self.device = torch.device("cpu")
            self.encoder.to(self.device)

            # Clear GPU cache
            torch.cuda.empty_cache()
        else:
            logger.info("Already using CPU device")

    def clear_gpu_cache(self) -> None:
        """
        Clear GPU cache to free memory.

        Should be called after processing batches to prevent memory accumulation.

        Example:
            >>> for batch in batches:
            ...     features = generator.extract_features(batch)
            ...     save_features(features)
            ...     generator.clear_gpu_cache()
        """
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("Cleared GPU cache")
