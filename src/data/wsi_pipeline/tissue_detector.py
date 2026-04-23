"""
Tissue Detector for WSI processing.

This module provides tissue detection functionality to filter out background
regions and focus processing on tissue-containing patches. Supports Otsu
thresholding for efficient tissue segmentation and optional deep learning
tissue detection using pretrained segmentation models.
"""

import logging
from typing import Optional, Tuple, Union
import warnings

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk
from skimage.transform import resize

from .exceptions import ProcessingError
from .reader import WSIReader

# Optional deep learning dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torchvision.models import resnet18
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    transforms = None
    resnet18 = None

logger = logging.getLogger(__name__)


class SimpleTissueSegmentationModel(nn.Module):
    """
    Simple CNN-based tissue segmentation model.
    
    This is a lightweight model for tissue detection that can be used
    when deep learning tissue detection is enabled. Based on ResNet-18
    with a segmentation head.
    """
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize tissue segmentation model.
        
        Args:
            num_classes: Number of classes (2 for tissue/background)
        """
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for deep learning tissue detection. "
                "Install with: pip install torch torchvision"
            )
        
        # Use ResNet-18 as backbone
        backbone = resnet18(pretrained=True)
        
        # Remove final layers
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # Add segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )
        
    def forward(self, x):
        """Forward pass."""
        # Extract features
        features = self.features(x)
        
        # Generate segmentation map
        seg_map = self.segmentation_head(features)
        
        # Upsample to input size
        seg_map = F.interpolate(
            seg_map, 
            size=x.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        return seg_map


class TissueDetector:
    """
    Detect tissue regions and filter background patches.

    Supports multiple detection methods:
    - "otsu": Fast Otsu thresholding (default)
    - "deep_learning": CNN-based tissue segmentation
    - "hybrid": Otsu for initial filtering, DL for refinement

    The detector generates a binary tissue mask at a low resolution (thumbnail)
    for efficiency, then uses this mask to determine if individual patches
    contain sufficient tissue.

    Args:
        method: Tissue detection method ("otsu", "deep_learning", "hybrid")
        tissue_threshold: Minimum fraction of tissue pixels required (0.0-1.0)
        thumbnail_level: Pyramid level for tissue mask generation (-1 = lowest)
        model_path: Path to pretrained tissue segmentation model (for DL methods)
        device: Device for DL inference ("auto", "cuda", "cpu")

    Example:
        >>> # Otsu thresholding (fast)
        >>> detector = TissueDetector(method="otsu", tissue_threshold=0.5)
        >>> 
        >>> # Deep learning (more accurate)
        >>> detector = TissueDetector(
        ...     method="deep_learning", 
        ...     model_path="tissue_model.pth"
        ... )
        >>> 
        >>> tissue_mask = detector.generate_tissue_mask(reader)
        >>> is_tissue = detector.is_tissue_patch(patch)
    """

    def __init__(
        self,
        method: str = "otsu",
        tissue_threshold: float = 0.5,
        thumbnail_level: int = -1,
        model_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize tissue detector.

        Args:
            method: Detection method ("otsu", "deep_learning", "hybrid")
            tissue_threshold: Minimum tissue fraction (0.0-1.0) for patch to be
                            considered tissue
            thumbnail_level: Pyramid level for mask generation (-1 = lowest level)
            model_path: Path to pretrained model file (required for DL methods)
            device: Device for DL inference ("auto", "cuda", "cpu")

        Raises:
            ValueError: If tissue_threshold is not in valid range
            ImportError: If PyTorch is required but not available
        """
        if not 0.0 <= tissue_threshold <= 1.0:
            raise ValueError(
                f"tissue_threshold must be between 0.0 and 1.0, got {tissue_threshold}"
            )

        # Validate method
        supported_methods = ["otsu", "deep_learning", "hybrid"]
        if method not in supported_methods:
            logger.warning(
                f"Method '{method}' not supported, using 'otsu'. "
                f"Supported methods: {supported_methods}"
            )
            method = "otsu"

        # Check PyTorch availability for DL methods
        if method in ["deep_learning", "hybrid"] and not TORCH_AVAILABLE:
            raise ImportError(
                f"PyTorch is required for method '{method}'. "
                "Install with: pip install torch torchvision"
            )

        self.method = method
        self.tissue_threshold = tissue_threshold
        self.thumbnail_level = thumbnail_level
        self.model_path = model_path

        # Setup device for DL methods
        if method in ["deep_learning", "hybrid"]:
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            # Load model
            self._load_model()
        else:
            self.device = None
            self.model = None

        # Cache for tissue masks (keyed by slide path)
        self._mask_cache = {}

        logger.debug(
            f"Initialized TissueDetector: method={method}, "
            f"threshold={tissue_threshold}, thumbnail_level={thumbnail_level}, "
            f"device={self.device}"
        )

    def _load_model(self) -> None:
        """Load pretrained tissue segmentation model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning tissue detection")

        try:
            if self.model_path and self.model_path != "default":
                # Load custom model
                logger.info(f"Loading custom tissue model from {self.model_path}")
                self.model = torch.load(self.model_path, map_location=self.device)
            else:
                # Use default simple model (randomly initialized)
                logger.info("Using default tissue segmentation model (randomly initialized)")
                self.model = SimpleTissueSegmentationModel(num_classes=2)
                
            self.model.to(self.device)
            self.model.eval()
            
            # Setup preprocessing transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info(f"Loaded tissue segmentation model on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load tissue segmentation model: {e}")
            # Fallback to Otsu method
            logger.warning("Falling back to Otsu thresholding")
            self.method = "otsu"
            self.model = None

    def generate_tissue_mask(
        self,
        reader: WSIReader,
        thumbnail_size: Tuple[int, int] = (2048, 2048),
    ) -> np.ndarray:
        """
        Generate binary tissue mask at thumbnail resolution.

        The mask is generated at low resolution for efficiency, then cached to
        avoid recomputation. Supports multiple detection methods:
        - Otsu: Fast automatic thresholding
        - Deep Learning: CNN-based segmentation
        - Hybrid: Otsu + DL refinement

        Args:
            reader: WSIReader instance for the slide
            thumbnail_size: Maximum size for thumbnail (width, height)

        Returns:
            Binary mask (height, width) where True = tissue, False = background

        Example:
            >>> with WSIReader("slide.svs") as reader:
            ...     mask = detector.generate_tissue_mask(reader)
            ...     tissue_coverage = mask.sum() / mask.size
        """
        # Check cache first
        slide_path = str(reader.wsi_path)
        cache_key = f"{slide_path}_{self.method}_{thumbnail_size}"
        if cache_key in self._mask_cache:
            logger.debug(f"Using cached tissue mask for {reader.wsi_path.name}")
            return self._mask_cache[cache_key]

        try:
            # Get thumbnail
            thumbnail = reader.get_thumbnail(size=thumbnail_size)
            thumbnail_array = np.array(thumbnail)

            # Generate mask based on method
            if self.method == "otsu":
                mask = self._otsu_tissue_detection(thumbnail_array)
            elif self.method == "deep_learning":
                mask = self._dl_tissue_detection(thumbnail_array)
            elif self.method == "hybrid":
                mask = self._hybrid_tissue_detection(thumbnail_array)
            else:
                raise ProcessingError(f"Unsupported tissue detection method: {self.method}")

            # Cache the mask
            self._mask_cache[cache_key] = mask

            tissue_pixels = mask.sum()
            total_pixels = mask.size
            coverage = tissue_pixels / total_pixels if total_pixels > 0 else 0.0

            logger.info(
                f"Generated tissue mask for {reader.wsi_path.name} using {self.method}: "
                f"{tissue_pixels}/{total_pixels} pixels ({coverage:.1%} tissue)"
            )

            return mask

        except Exception as e:
            raise ProcessingError(f"Failed to generate tissue mask: {e}")

    def _otsu_tissue_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Otsu thresholding for tissue detection.

        Args:
            image: RGB image array (height, width, 3)

        Returns:
            Binary mask where True = tissue, False = background
        """
        # Convert to grayscale
        if image.ndim == 3:
            gray = rgb2gray(image)
        else:
            gray = image

        # Invert grayscale (tissue is darker than background)
        # This makes tissue pixels have higher values
        gray_inverted = 1.0 - gray

        # Check if image is uniform (no variation)
        if gray_inverted.std() < 1e-6:
            # Uniform image - decide based on mean intensity
            # If dark (mean < 0.5 after inversion), it's tissue
            # If bright (mean >= 0.5 after inversion), it's background
            mean_val = gray_inverted.mean()
            tissue_mask = np.ones_like(gray_inverted, dtype=bool) if mean_val > 0.5 else np.zeros_like(gray_inverted, dtype=bool)
            logger.debug(f"Uniform image detected (std={gray_inverted.std():.6f}), using mean-based classification")
            return tissue_mask

        # Apply Otsu thresholding
        try:
            threshold = threshold_otsu(gray_inverted)
            tissue_mask = gray_inverted > threshold
        except Exception as e:
            logger.warning(f"Otsu thresholding failed: {e}, using fallback threshold")
            # Fallback: use fixed threshold
            threshold = 0.5
            tissue_mask = gray_inverted > threshold

        # Apply morphological closing to fill small holes
        # Use a small disk structuring element
        try:
            selem = disk(5)
            tissue_mask = closing(tissue_mask, selem)
        except Exception as e:
            logger.warning(f"Morphological closing failed: {e}, skipping")

        return tissue_mask

    def _dl_tissue_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Apply deep learning-based tissue detection.

        Args:
            image: RGB image array (height, width, 3)

        Returns:
            Binary mask where True = tissue, False = background
        """
        if self.model is None:
            logger.warning("DL model not available, falling back to Otsu")
            return self._otsu_tissue_detection(image)

        try:
            # Preprocess image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                
            # Get predictions (class 1 = tissue)
            predictions = torch.softmax(output, dim=1)
            tissue_prob = predictions[0, 1].cpu().numpy()
            
            # Resize back to original size
            original_size = image.shape[:2]
            tissue_prob_resized = resize(
                tissue_prob, 
                original_size, 
                preserve_range=True,
                anti_aliasing=True
            )
            
            # Threshold to get binary mask
            tissue_mask = tissue_prob_resized > 0.5
            
            # Apply morphological closing
            try:
                selem = disk(5)
                tissue_mask = closing(tissue_mask, selem)
            except Exception as e:
                logger.warning(f"Morphological closing failed: {e}, skipping")
            
            return tissue_mask
            
        except Exception as e:
            logger.warning(f"DL tissue detection failed: {e}, falling back to Otsu")
            return self._otsu_tissue_detection(image)

    def _hybrid_tissue_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Apply hybrid tissue detection (Otsu + DL refinement).

        Uses Otsu for initial coarse segmentation, then applies DL model
        for refinement in regions of uncertainty.

        Args:
            image: RGB image array (height, width, 3)

        Returns:
            Binary mask where True = tissue, False = background
        """
        try:
            # Step 1: Get Otsu mask
            otsu_mask = self._otsu_tissue_detection(image)
            
            # Step 2: Get DL mask
            if self.model is not None:
                dl_mask = self._dl_tissue_detection(image)
                
                # Step 3: Combine masks
                # Use DL mask where Otsu is uncertain (near boundaries)
                # Create uncertainty map by dilating Otsu boundaries
                from skimage.morphology import binary_dilation, binary_erosion
                
                # Find boundary regions
                dilated = binary_dilation(otsu_mask, disk(10))
                eroded = binary_erosion(otsu_mask, disk(10))
                boundary_region = dilated & ~eroded
                
                # Use DL predictions in boundary regions, Otsu elsewhere
                combined_mask = otsu_mask.copy()
                combined_mask[boundary_region] = dl_mask[boundary_region]
                
                return combined_mask
            else:
                logger.warning("DL model not available for hybrid mode, using Otsu only")
                return otsu_mask
                
        except Exception as e:
            logger.warning(f"Hybrid tissue detection failed: {e}, falling back to Otsu")
            return self._otsu_tissue_detection(image)

    def calculate_tissue_percentage(self, patch: np.ndarray) -> float:
        """
        Calculate percentage of tissue pixels in a patch.

        Uses a fast approximation for performance. For Otsu method,
        uses a simplified grayscale threshold instead of full Otsu calculation.

        Args:
            patch: RGB patch array (height, width, 3)

        Returns:
            Tissue percentage as float between 0.0 and 1.0

        Example:
            >>> patch = reader.read_region((1000, 1000), level=0, size=(256, 256))
            >>> tissue_pct = detector.calculate_tissue_percentage(patch)
            >>> print(f"Patch contains {tissue_pct:.1%} tissue")
        """
        try:
            # Fast approximation for performance
            if self.method == "otsu":
                # Fast grayscale threshold instead of full Otsu
                if patch.ndim == 3:
                    gray = np.mean(patch, axis=2)  # Fast RGB to grayscale
                else:
                    gray = patch
                
                # Normalize to 0-1
                gray = gray / 255.0 if gray.max() > 1.0 else gray
                
                # Simple threshold (tissue is darker)
                # Use a fixed threshold for speed instead of Otsu calculation
                tissue_mask = gray < 0.8  # Tissue is typically darker than 0.8
                
                tissue_pixels = tissue_mask.sum()
                total_pixels = tissue_mask.size
                
                return tissue_pixels / total_pixels if total_pixels > 0 else 0.0
            
            else:
                # For DL methods, fall back to full calculation
                if self.method == "deep_learning":
                    patch_mask = self._dl_tissue_detection(patch)
                elif self.method == "hybrid":
                    patch_mask = self._hybrid_tissue_detection(patch)
                else:
                    # Fallback to fast Otsu
                    if patch.ndim == 3:
                        gray = np.mean(patch, axis=2)
                    else:
                        gray = patch
                    gray = gray / 255.0 if gray.max() > 1.0 else gray
                    patch_mask = gray < 0.8

                tissue_pixels = patch_mask.sum()
                total_pixels = patch_mask.size
                return tissue_pixels / total_pixels if total_pixels > 0 else 0.0

        except Exception as e:
            logger.warning(f"Failed to calculate tissue percentage: {e}")
            return 0.0

    def is_tissue_patch(
        self,
        patch: np.ndarray,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Determine if a patch contains sufficient tissue.

        Args:
            patch: RGB patch array (height, width, 3)
            threshold: Custom tissue threshold (uses self.tissue_threshold if None)

        Returns:
            True if patch contains sufficient tissue, False otherwise

        Example:
            >>> for patch, coord in extractor.extract_patches_streaming(reader, coords):
            ...     if detector.is_tissue_patch(patch):
            ...         features = model(patch)
            ...         save_features(features, coord)
        """
        if threshold is None:
            threshold = self.tissue_threshold

        tissue_pct = self.calculate_tissue_percentage(patch)
        return tissue_pct >= threshold

    def clear_cache(self) -> None:
        """
        Clear the tissue mask cache.

        Useful when processing many slides to free memory.

        Example:
            >>> for slide_path in slide_paths:
            ...     with WSIReader(slide_path) as reader:
            ...         mask = detector.generate_tissue_mask(reader)
            ...         process_slide(reader, mask)
            ...     detector.clear_cache()  # Free memory after each slide
        """
        self._mask_cache.clear()
        logger.debug("Cleared tissue mask cache")

    def get_cache_size(self) -> int:
        """
        Get number of cached tissue masks.

        Returns:
            Number of slides with cached masks
        """
        return len(self._mask_cache)
