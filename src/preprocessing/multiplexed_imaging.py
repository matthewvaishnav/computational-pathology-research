"""
Multiplexed Imaging Support

CODEX, Vectra, and multi-channel immunofluorescence processing.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    from skimage import exposure, filters
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not available")


class MultiplexedImageProcessor:
    """
    Processor for multiplexed immunofluorescence images.
    
    Handles CODEX, Vectra, and multi-channel IF data.
    """
    
    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        normalize_per_channel: bool = True
    ):
        """
        Initialize multiplexed image processor.
        
        Args:
            channel_names: Names of channels (e.g., ["DAPI", "CD3", "CD8"])
            normalize_per_channel: Whether to normalize each channel independently
        """
        self.channel_names = channel_names or []
        self.normalize_per_channel = normalize_per_channel
        self._channel_stats = {}
    
    def normalize_channel(
        self,
        channel: np.ndarray,
        method: str = "percentile",
        percentile_range: Tuple[float, float] = (1, 99)
    ) -> np.ndarray:
        """
        Normalize single channel.
        
        Args:
            channel: Channel data (H, W)
            method: Normalization method ("percentile", "minmax", "zscore")
            percentile_range: Percentile range for clipping
        
        Returns:
            Normalized channel (H, W)
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required for normalization")
        
        if method == "percentile":
            p_low, p_high = percentile_range
            v_low, v_high = np.percentile(channel, [p_low, p_high])
            normalized = np.clip((channel - v_low) / (v_high - v_low + 1e-8), 0, 1)
        
        elif method == "minmax":
            v_min, v_max = channel.min(), channel.max()
            normalized = (channel - v_min) / (v_max - v_min + 1e-8)
        
        elif method == "zscore":
            mean, std = channel.mean(), channel.std()
            normalized = (channel - mean) / (std + 1e-8)
            # Clip to reasonable range
            normalized = np.clip(normalized, -3, 3)
            normalized = (normalized + 3) / 6  # Scale to [0, 1]
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized.astype(np.float32)
    
    def process_multiplexed_image(
        self,
        image: np.ndarray,
        channel_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Process multiplexed image.
        
        Args:
            image: Multi-channel image (H, W, C) or (C, H, W)
            channel_names: Optional channel names
        
        Returns:
            Dict mapping channel names to normalized images
        """
        # Handle channel dimension
        if image.ndim == 3:
            if image.shape[2] < image.shape[0]:
                # Likely (C, H, W) - transpose to (H, W, C)
                image = np.transpose(image, (1, 2, 0))
        
        num_channels = image.shape[2]
        names = channel_names or self.channel_names or [f"Channel_{i}" for i in range(num_channels)]
        
        if len(names) != num_channels:
            raise ValueError(f"Channel names ({len(names)}) != image channels ({num_channels})")
        
        processed = {}
        for i, name in enumerate(names):
            channel = image[:, :, i]
            
            if self.normalize_per_channel:
                channel = self.normalize_channel(channel)
            
            processed[name] = channel
            
            # Store stats
            self._channel_stats[name] = {
                'mean': float(channel.mean()),
                'std': float(channel.std()),
                'min': float(channel.min()),
                'max': float(channel.max())
            }
        
        return processed
    
    def create_composite(
        self,
        channels: Dict[str, np.ndarray],
        channel_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Create RGB composite from multiple channels.
        
        Args:
            channels: Dict mapping channel names to images
            channel_colors: Dict mapping channel names to RGB colors
        
        Returns:
            RGB composite image (H, W, 3)
        """
        if not channels:
            raise ValueError("No channels provided")
        
        # Get image shape
        first_channel = next(iter(channels.values()))
        h, w = first_channel.shape
        
        # Initialize composite
        composite = np.zeros((h, w, 3), dtype=np.float32)
        
        # Default colors (if not provided)
        default_colors = {
            'DAPI': (0, 0, 255),      # Blue
            'CD3': (0, 255, 0),        # Green
            'CD8': (255, 0, 0),        # Red
            'CD4': (255, 255, 0),      # Yellow
            'CD20': (255, 0, 255),     # Magenta
            'PanCK': (0, 255, 255),    # Cyan
        }
        
        colors = channel_colors or {}
        
        for name, channel in channels.items():
            # Get color for this channel
            if name in colors:
                color = colors[name]
            elif name in default_colors:
                color = default_colors[name]
            else:
                # Random color
                color = tuple(np.random.randint(0, 256, 3))
            
            # Normalize color to [0, 1]
            color = np.array(color) / 255.0
            
            # Add weighted channel to composite
            for c in range(3):
                composite[:, :, c] += channel * color[c]
        
        # Clip to valid range
        composite = np.clip(composite, 0, 1)
        
        return composite
    
    def extract_marker_positive_regions(
        self,
        channel: np.ndarray,
        threshold: Optional[float] = None,
        method: str = "otsu"
    ) -> np.ndarray:
        """
        Extract marker-positive regions.
        
        Args:
            channel: Channel data (H, W)
            threshold: Manual threshold (if None, auto-compute)
            method: Thresholding method ("otsu", "li", "yen")
        
        Returns:
            Binary mask (H, W)
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required for thresholding")
        
        if threshold is None:
            # Auto-threshold
            if method == "otsu":
                threshold = filters.threshold_otsu(channel)
            elif method == "li":
                threshold = filters.threshold_li(channel)
            elif method == "yen":
                threshold = filters.threshold_yen(channel)
            else:
                raise ValueError(f"Unknown thresholding method: {method}")
        
        mask = channel > threshold
        return mask.astype(np.uint8)
    
    def compute_colocalization(
        self,
        channel1: np.ndarray,
        channel2: np.ndarray,
        threshold1: Optional[float] = None,
        threshold2: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute colocalization metrics between two channels.
        
        Args:
            channel1: First channel (H, W)
            channel2: Second channel (H, W)
            threshold1: Threshold for channel 1
            threshold2: Threshold for channel 2
        
        Returns:
            Dict with colocalization metrics
        """
        # Get binary masks
        mask1 = self.extract_marker_positive_regions(channel1, threshold1)
        mask2 = self.extract_marker_positive_regions(channel2, threshold2)
        
        # Compute overlap
        overlap = np.logical_and(mask1, mask2)
        
        # Metrics
        total_pixels = mask1.size
        positive1 = mask1.sum()
        positive2 = mask2.sum()
        colocalized = overlap.sum()
        
        # Pearson correlation
        pearson = np.corrcoef(channel1.flatten(), channel2.flatten())[0, 1]
        
        # Manders coefficients
        m1 = colocalized / (positive1 + 1e-8)  # Fraction of ch1 colocalized with ch2
        m2 = colocalized / (positive2 + 1e-8)  # Fraction of ch2 colocalized with ch1
        
        return {
            'pearson_correlation': float(pearson),
            'manders_m1': float(m1),
            'manders_m2': float(m2),
            'colocalized_pixels': int(colocalized),
            'channel1_positive': int(positive1),
            'channel2_positive': int(positive2),
            'overlap_coefficient': float(colocalized / min(positive1, positive2 + 1e-8))
        }


class CODEXProcessor(MultiplexedImageProcessor):
    """
    Specialized processor for CODEX data.
    
    CODEX typically has 20-60 protein markers.
    """
    
    def __init__(
        self,
        marker_panel: Optional[List[str]] = None,
        background_subtraction: bool = True
    ):
        """
        Initialize CODEX processor.
        
        Args:
            marker_panel: List of marker names
            background_subtraction: Whether to subtract background
        """
        super().__init__(channel_names=marker_panel)
        self.background_subtraction = background_subtraction
    
    def subtract_background(
        self,
        image: np.ndarray,
        radius: int = 50
    ) -> np.ndarray:
        """
        Subtract background using rolling ball algorithm.
        
        Args:
            image: Input image (H, W)
            radius: Rolling ball radius
        
        Returns:
            Background-subtracted image
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required for background subtraction")
        
        from skimage.morphology import disk, white_tophat
        
        # White top-hat transform
        selem = disk(radius)
        background = white_tophat(image, selem)
        
        return image - background


class VectraProcessor(MultiplexedImageProcessor):
    """
    Specialized processor for Vectra data.
    
    Vectra typically has 4-8 fluorescent markers.
    """
    
    def __init__(
        self,
        marker_panel: Optional[List[str]] = None,
        spectral_unmixing: bool = True
    ):
        """
        Initialize Vectra processor.
        
        Args:
            marker_panel: List of marker names
            spectral_unmixing: Whether to perform spectral unmixing
        """
        super().__init__(channel_names=marker_panel)
        self.spectral_unmixing = spectral_unmixing
    
    def unmix_spectra(
        self,
        image: np.ndarray,
        unmixing_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Perform spectral unmixing.
        
        Args:
            image: Multi-channel image (H, W, C)
            unmixing_matrix: Unmixing matrix (C, C)
        
        Returns:
            Unmixed image (H, W, C)
        """
        h, w, c = image.shape
        
        # Reshape for matrix multiplication
        pixels = image.reshape(-1, c)
        
        # Apply unmixing
        unmixed = pixels @ unmixing_matrix.T
        
        # Reshape back
        unmixed = unmixed.reshape(h, w, c)
        
        # Clip negative values
        unmixed = np.maximum(unmixed, 0)
        
        return unmixed


def process_codex_image(
    image: np.ndarray,
    marker_panel: List[str],
    normalize: bool = True,
    background_subtract: bool = True
) -> Dict[str, np.ndarray]:
    """
    Convenience function for CODEX processing.
    
    Args:
        image: Multi-channel CODEX image
        marker_panel: List of marker names
        normalize: Whether to normalize channels
        background_subtract: Whether to subtract background
    
    Returns:
        Dict mapping marker names to processed images
    """
    processor = CODEXProcessor(
        marker_panel=marker_panel,
        background_subtraction=background_subtract
    )
    
    return processor.process_multiplexed_image(image, marker_panel)


def process_vectra_image(
    image: np.ndarray,
    marker_panel: List[str],
    unmixing_matrix: Optional[np.ndarray] = None,
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """
    Convenience function for Vectra processing.
    
    Args:
        image: Multi-channel Vectra image
        marker_panel: List of marker names
        unmixing_matrix: Optional spectral unmixing matrix
        normalize: Whether to normalize channels
    
    Returns:
        Dict mapping marker names to processed images
    """
    processor = VectraProcessor(
        marker_panel=marker_panel,
        spectral_unmixing=unmixing_matrix is not None
    )
    
    if unmixing_matrix is not None:
        image = processor.unmix_spectra(image, unmixing_matrix)
    
    return processor.process_multiplexed_image(image, marker_panel)
