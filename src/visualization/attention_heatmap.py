"""
Attention heatmap visualization for whole-slide images.

This module provides the AttentionHeatmapGenerator class for creating
attention heatmap visualizations from saved attention weights. Heatmaps
can be overlaid on slide thumbnails for interpretability analysis.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class AttentionHeatmapGenerator:
    """Generate attention heatmaps for whole-slide images.
    
    This class loads attention weights from HDF5 files and generates
    heatmap visualizations that can be overlaid on slide thumbnails.
    Supports batch processing and configurable colormaps.
    
    Args:
        attention_dir: Directory containing attention weight HDF5 files
        output_dir: Directory to save heatmap images
        colormap: Matplotlib colormap name (default: 'jet')
        thumbnail_size: Size for slide thumbnail (default: (1000, 1000))
        
    Example:
        >>> generator = AttentionHeatmapGenerator(
        ...     attention_dir="outputs/attention_weights",
        ...     output_dir="outputs/heatmaps",
        ...     colormap="jet"
        ... )
        >>> generator.generate_heatmap("slide_001")
        >>> generator.generate_batch(["slide_001", "slide_002"])
    """
    
    def __init__(
        self,
        attention_dir: Path,
        output_dir: Path,
        colormap: str = "jet",
        thumbnail_size: Tuple[int, int] = (1000, 1000),
    ):
        """Initialize the attention heatmap generator.
        
        Args:
            attention_dir: Directory containing attention weight HDF5 files
            output_dir: Directory to save heatmap images
            colormap: Matplotlib colormap name (default: 'jet')
            thumbnail_size: Size for slide thumbnail (default: (1000, 1000))
        """
        self.attention_dir = Path(attention_dir)
        self.output_dir = Path(output_dir)
        self.colormap = plt.get_cmap(colormap)
        self.thumbnail_size = thumbnail_size
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initialized AttentionHeatmapGenerator: "
            f"attention_dir={self.attention_dir}, "
            f"output_dir={self.output_dir}, "
            f"colormap={colormap}"
        )
    
    def load_attention_weights(
        self, slide_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load attention weights and coordinates for a slide.
        
        This method loads previously saved attention weights and patch
        coordinates from an HDF5 file. Returns None if the file is not found.
        
        Args:
            slide_id: Slide identifier
            
        Returns:
            Tuple of (attention_weights, coordinates) or None if not found.
            attention_weights has shape [num_patches] and coordinates has
            shape [num_patches, 2].
            
        Example:
            >>> weights, coords = generator.load_attention_weights("slide_001")
            >>> if weights is not None:
            ...     print(f"Loaded {len(weights)} attention weights")
        """
        attention_path = self.attention_dir / f"{slide_id}.h5"
        
        if not attention_path.exists():
            logger.warning(f"Attention weights not found: {attention_path}")
            return None
        
        try:
            with h5py.File(attention_path, "r") as f:
                attention_weights = f["attention_weights"][:]
                coordinates = f["coordinates"][:]
            
            logger.info(
                f"Loaded attention weights for {slide_id}: "
                f"{len(attention_weights)} patches"
            )
            return attention_weights, coordinates
        
        except Exception as e:
            logger.error(f"Error loading attention weights for {slide_id}: {e}")
            return None
    
    def create_heatmap_array(
        self,
        attention_weights: np.ndarray,
        coordinates: np.ndarray,
        canvas_size: Optional[Tuple[int, int]] = None,
        patch_size: int = 256,
    ) -> np.ndarray:
        """Create heatmap array from attention weights and coordinates.
        
        This method normalizes attention weights to [0, 1] range and maps
        them to a 2D canvas based on patch coordinates. Overlapping regions
        are averaged.
        
        Args:
            attention_weights: [num_patches] attention weights
            coordinates: [num_patches, 2] patch coordinates in (x, y) format
            canvas_size: (height, width) of output canvas. If None, uses
                self.thumbnail_size
            patch_size: Size of each patch in pixels (default: 256)
            
        Returns:
            Heatmap array [height, width] with normalized attention values
            
        Example:
            >>> weights = np.array([0.1, 0.3, 0.6])
            >>> coords = np.array([[0, 0], [256, 0], [0, 256]])
            >>> heatmap = generator.create_heatmap_array(weights, coords)
        """
        if canvas_size is None:
            canvas_size = self.thumbnail_size
        
        # Normalize attention weights to [0, 1]
        attention_norm = (attention_weights - attention_weights.min()) / \
                        (attention_weights.max() - attention_weights.min() + 1e-8)
        
        # Create canvas
        heatmap = np.zeros(canvas_size, dtype=np.float32)
        counts = np.zeros(canvas_size, dtype=np.int32)
        
        # Find coordinate ranges for scaling
        x_coords = coordinates[:, 0]
        y_coords = coordinates[:, 1]
        x_max = x_coords.max() + patch_size
        y_max = y_coords.max() + patch_size
        
        # Place attention values at patch coordinates
        for i, (x, y) in enumerate(coordinates):
            # Convert coordinates to canvas space
            y_start = int(y * canvas_size[0] / y_max)
            x_start = int(x * canvas_size[1] / x_max)
            
            # Determine patch size in canvas space
            patch_h = max(1, patch_size * canvas_size[0] // y_max)
            patch_w = max(1, patch_size * canvas_size[1] // x_max)
            
            y_end = min(canvas_size[0], y_start + patch_h)
            x_end = min(canvas_size[1], x_start + patch_w)
            
            # Add attention value
            heatmap[y_start:y_end, x_start:x_end] += attention_norm[i]
            counts[y_start:y_end, x_start:x_end] += 1
        
        # Average overlapping regions
        # Use np.where to avoid division by zero
        with np.errstate(invalid='ignore'):
            heatmap = np.where(counts > 0, heatmap / counts, 0.0)
        
        return heatmap
    
    def generate_heatmap(
        self,
        slide_id: str,
        thumbnail_path: Optional[Path] = None,
        alpha: float = 0.5,
    ) -> Optional[Path]:
        """Generate attention heatmap for a slide.
        
        This method loads attention weights, creates a heatmap array, and
        generates a matplotlib figure with optional thumbnail overlay. The
        figure is saved to the output directory.
        
        Args:
            slide_id: Slide identifier
            thumbnail_path: Optional path to slide thumbnail image
            alpha: Transparency for heatmap overlay (0=transparent, 1=opaque)
            
        Returns:
            Path to generated heatmap or None if failed
            
        Example:
            >>> path = generator.generate_heatmap(
            ...     "slide_001",
            ...     thumbnail_path=Path("thumbnails/slide_001.png"),
            ...     alpha=0.5
            ... )
            >>> print(f"Heatmap saved to {path}")
        """
        # Load attention weights
        result = self.load_attention_weights(slide_id)
        if result is None:
            return None
        
        attention_weights, coordinates = result
        
        # Create heatmap array
        heatmap = self.create_heatmap_array(
            attention_weights,
            coordinates,
            self.thumbnail_size,
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Load and display thumbnail if available
        if thumbnail_path and thumbnail_path.exists():
            thumbnail = Image.open(thumbnail_path).resize(self.thumbnail_size)
            ax.imshow(thumbnail, aspect='auto')
        
        # Overlay heatmap
        im = ax.imshow(heatmap, cmap=self.colormap, alpha=alpha, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention Weight", rotation=270, labelpad=20)
        
        # Set title and labels
        ax.set_title(f"Attention Heatmap: {slide_id}", fontsize=16)
        ax.axis('off')
        
        # Save figure
        output_path = self.output_dir / f"{slide_id}_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Generated heatmap: {output_path}")
        return output_path
    
    def generate_batch(
        self,
        slide_ids: list[str],
        thumbnail_dir: Optional[Path] = None,
    ) -> list[Path]:
        """Generate heatmaps for multiple slides.
        
        This method iterates over a list of slide IDs and generates a
        heatmap for each one. Optionally loads thumbnails from a directory.
        
        Args:
            slide_ids: List of slide identifiers
            thumbnail_dir: Optional directory containing slide thumbnails
            
        Returns:
            List of paths to generated heatmaps
            
        Example:
            >>> paths = generator.generate_batch(
            ...     ["slide_001", "slide_002", "slide_003"],
            ...     thumbnail_dir=Path("thumbnails")
            ... )
            >>> print(f"Generated {len(paths)} heatmaps")
        """
        heatmap_paths = []
        
        for slide_id in slide_ids:
            thumbnail_path = None
            if thumbnail_dir:
                thumbnail_path = Path(thumbnail_dir) / f"{slide_id}.png"
            
            heatmap_path = self.generate_heatmap(slide_id, thumbnail_path)
            if heatmap_path:
                heatmap_paths.append(heatmap_path)
        
        logger.info(f"Generated {len(heatmap_paths)} heatmaps")
        return heatmap_paths
