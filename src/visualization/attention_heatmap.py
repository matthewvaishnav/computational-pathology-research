"""
Attention heatmap visualization for whole-slide images.

This module provides the AttentionHeatmapGenerator class for creating
attention heatmap visualizations from saved attention weights. Heatmaps
can be overlaid on slide thumbnails for interpretability analysis.

Supports multi-disease-state attention visualization for clinical explainability.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
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

    def load_attention_weights(self, slide_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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
                f"Loaded attention weights for {slide_id}: " f"{len(attention_weights)} patches"
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
        attention_norm = (attention_weights - attention_weights.min()) / (
            attention_weights.max() - attention_weights.min() + 1e-8
        )

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
        with np.errstate(invalid="ignore"):
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
            ax.imshow(thumbnail, aspect="auto")

        # Overlay heatmap
        im = ax.imshow(heatmap, cmap=self.colormap, alpha=alpha, aspect="auto")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention Weight", rotation=270, labelpad=20)

        # Set title and labels
        ax.set_title(f"Attention Heatmap: {slide_id}", fontsize=16)
        ax.axis("off")

        # Save figure
        output_path = self.output_dir / f"{slide_id}_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
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

    def generate_heatmap_with_zoom(
        self,
        slide_id: str,
        thumbnail_path: Optional[Path] = None,
        alpha: float = 0.5,
        zoom_regions: Optional[List[Tuple[int, int, int, int]]] = None,
        top_k_patches: int = 5,
    ) -> Optional[Path]:
        """Generate attention heatmap with zoom functionality for high-attention regions.

        This method creates a multi-panel visualization showing the full heatmap
        alongside zoomed views of high-attention regions. Zoom regions can be
        specified manually or automatically detected from top-k attention patches.

        Args:
            slide_id: Slide identifier
            thumbnail_path: Optional path to slide thumbnail image
            alpha: Transparency for heatmap overlay (0=transparent, 1=opaque)
            zoom_regions: Optional list of (x, y, width, height) tuples for zoom regions
            top_k_patches: Number of top attention patches to zoom into (default: 5)

        Returns:
            Path to generated heatmap with zoom panels or None if failed

        Example:
            >>> # Auto-detect top-5 high-attention regions
            >>> path = generator.generate_heatmap_with_zoom(
            ...     "slide_001",
            ...     thumbnail_path=Path("thumbnails/slide_001.png"),
            ...     top_k_patches=5
            ... )
            >>> # Manual zoom regions
            >>> path = generator.generate_heatmap_with_zoom(
            ...     "slide_001",
            ...     zoom_regions=[(100, 100, 200, 200), (500, 300, 200, 200)]
            ... )
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

        # Identify high-attention regions if not provided
        if zoom_regions is None:
            zoom_regions = self._identify_high_attention_regions(
                attention_weights, coordinates, top_k=top_k_patches
            )

        # Create multi-panel figure
        num_zoom = len(zoom_regions)

        # Calculate grid layout: main heatmap takes 2 columns in first row
        # Zoom panels fill remaining space
        ncols = max(3, num_zoom)
        nrows = 2

        fig = plt.figure(figsize=(6 * ncols, 6 * nrows))
        gs = fig.add_gridspec(nrows, ncols)

        # Main heatmap panel (spans first 2 columns of first row)
        ax_main = fig.add_subplot(gs[0, :2])

        # Load and display thumbnail if available
        if thumbnail_path and thumbnail_path.exists():
            thumbnail = Image.open(thumbnail_path).resize(self.thumbnail_size)
            ax_main.imshow(thumbnail, aspect="auto")

        # Overlay heatmap
        im = ax_main.imshow(heatmap, cmap=self.colormap, alpha=alpha, aspect="auto")

        # Draw rectangles around zoom regions
        for i, (x, y, w, h) in enumerate(zoom_regions):
            rect = Rectangle(
                (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none", linestyle="--"
            )
            ax_main.add_patch(rect)
            ax_main.text(x, y - 10, f"Region {i+1}", color="red", fontsize=10, weight="bold")

        ax_main.set_title(f"Attention Heatmap: {slide_id}", fontsize=14, weight="bold")
        ax_main.axis("off")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
        cbar.set_label("Attention Weight", rotation=270, labelpad=20)

        # Zoom panels - distribute across remaining space
        for i, (x, y, w, h) in enumerate(zoom_regions):
            # Calculate position in grid
            if i < ncols - 2:
                # First row, after main heatmap
                ax_zoom = fig.add_subplot(gs[0, 2 + i])
            else:
                # Second row
                col_idx = i - (ncols - 2)
                ax_zoom = fig.add_subplot(gs[1, col_idx])

            # Extract zoom region from heatmap
            zoom_heatmap = heatmap[y: y + h, x: x + w]

            # Display zoomed region
            if thumbnail_path and thumbnail_path.exists():
                thumbnail = Image.open(thumbnail_path).resize(self.thumbnail_size)
                thumbnail_array = np.array(thumbnail)
                zoom_thumbnail = thumbnail_array[y: y + h, x: x + w]
                ax_zoom.imshow(zoom_thumbnail, aspect="auto")

            ax_zoom.imshow(zoom_heatmap, cmap=self.colormap, alpha=alpha, aspect="auto")
            ax_zoom.set_title(f"High-Attention Region {i+1}", fontsize=12)
            ax_zoom.axis("off")

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"{slide_id}_heatmap_zoom.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Generated heatmap with zoom: {output_path}")
        return output_path

    def _identify_high_attention_regions(
        self,
        attention_weights: np.ndarray,
        coordinates: np.ndarray,
        top_k: int = 5,
        region_size: int = 200,
    ) -> List[Tuple[int, int, int, int]]:
        """Identify high-attention regions from attention weights.

        Args:
            attention_weights: [num_patches] attention weights
            coordinates: [num_patches, 2] patch coordinates
            top_k: Number of top attention patches to identify
            region_size: Size of zoom region around each patch

        Returns:
            List of (x, y, width, height) tuples for zoom regions
        """
        # Get top-k attention patches
        top_k_indices = np.argsort(attention_weights)[-top_k:][::-1]

        # Convert to canvas coordinates
        x_coords = coordinates[:, 0]
        y_coords = coordinates[:, 1]
        x_max = x_coords.max() + 256
        y_max = y_coords.max() + 256

        zoom_regions = []
        for idx in top_k_indices:
            x, y = coordinates[idx]

            # Convert to canvas space
            x_canvas = int(x * self.thumbnail_size[1] / x_max)
            y_canvas = int(y * self.thumbnail_size[0] / y_max)

            # Center region around patch
            x_start = max(0, x_canvas - region_size // 2)
            y_start = max(0, y_canvas - region_size // 2)

            # Ensure region fits in canvas
            x_start = min(x_start, self.thumbnail_size[1] - region_size)
            y_start = min(y_start, self.thumbnail_size[0] - region_size)

            zoom_regions.append((x_start, y_start, region_size, region_size))

        return zoom_regions

    def generate_multi_disease_heatmaps(
        self,
        slide_id: str,
        disease_attention_weights: Dict[str, np.ndarray],
        coordinates: np.ndarray,
        thumbnail_path: Optional[Path] = None,
        alpha: float = 0.5,
        min_probability: float = 0.1,
    ) -> Optional[Path]:
        """Generate separate attention heatmaps for multiple disease states.

        This method creates a multi-panel visualization showing attention heatmaps
        for each significant disease state. Each disease state has its own attention
        distribution that sums to 1.0 across all patches.

        Args:
            slide_id: Slide identifier
            disease_attention_weights: Dictionary mapping disease IDs to attention weights
                                      Each array has shape [num_patches] and sums to 1.0
            coordinates: [num_patches, 2] patch coordinates
            thumbnail_path: Optional path to slide thumbnail image
            alpha: Transparency for heatmap overlay (0=transparent, 1=opaque)
            min_probability: Minimum disease probability to include in visualization

        Returns:
            Path to generated multi-disease heatmap or None if failed

        Example:
            >>> disease_weights = {
            ...     'grade_1': np.array([0.2, 0.3, 0.5]),
            ...     'grade_2': np.array([0.4, 0.4, 0.2]),
            ...     'grade_3': np.array([0.1, 0.6, 0.3])
            ... }
            >>> coords = np.array([[0, 0], [256, 0], [512, 0]])
            >>> path = generator.generate_multi_disease_heatmaps(
            ...     "slide_001",
            ...     disease_weights,
            ...     coords,
            ...     thumbnail_path=Path("thumbnails/slide_001.png")
            ... )
        """
        # Validate attention weights sum to 1.0 for each disease
        for disease_id, weights in disease_attention_weights.items():
            weight_sum = np.sum(weights)
            if not np.isclose(weight_sum, 1.0, atol=1e-6):
                logger.warning(
                    f"Attention weights for disease '{disease_id}' sum to {weight_sum:.6f}, "
                    f"expected 1.0. Normalizing weights."
                )
                disease_attention_weights[disease_id] = weights / weight_sum

        # Filter diseases by minimum probability
        significant_diseases = {
            disease_id: weights
            for disease_id, weights in disease_attention_weights.items()
            if np.max(weights) >= min_probability
        }

        if not significant_diseases:
            logger.warning(f"No significant diseases found for {slide_id}")
            return None

        num_diseases = len(significant_diseases)

        # Create multi-panel figure
        ncols = min(3, num_diseases)
        nrows = (num_diseases + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))

        # Ensure axes is always a list
        if num_diseases == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if num_diseases > 1 else [axes]

        # Generate heatmap for each disease
        for idx, (disease_id, attention_weights) in enumerate(significant_diseases.items()):
            ax = axes[idx]

            # Create heatmap array
            heatmap = self.create_heatmap_array(
                attention_weights,
                coordinates,
                self.thumbnail_size,
            )

            # Load and display thumbnail if available
            if thumbnail_path and thumbnail_path.exists():
                thumbnail = Image.open(thumbnail_path).resize(self.thumbnail_size)
                ax.imshow(thumbnail, aspect="auto")

            # Overlay heatmap
            im = ax.imshow(heatmap, cmap=self.colormap, alpha=alpha, aspect="auto")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Attention Weight", rotation=270, labelpad=20)

            # Set title with disease ID and max attention
            max_attention = np.max(attention_weights)
            ax.set_title(
                f"Disease: {disease_id}\nMax Attention: {max_attention:.3f}",
                fontsize=12,
                weight="bold",
            )
            ax.axis("off")

        # Hide unused subplots
        for idx in range(num_diseases, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Multi-Disease Attention Heatmaps: {slide_id}", fontsize=16, weight="bold", y=0.98
        )
        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"{slide_id}_multi_disease_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Generated multi-disease heatmap: {output_path}")
        return output_path

    def generate_feature_importance_explanation(
        self,
        slide_id: str,
        attention_weights: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> Dict[str, any]:
        """Generate feature importance explanations for learned features.

        This method analyzes attention weights to identify which patches and
        features contribute most to predictions, providing interpretable
        explanations for clinical users.

        Args:
            slide_id: Slide identifier
            attention_weights: [num_patches] attention weights
            feature_names: Optional list of feature names for interpretation
            top_k: Number of top features to include in explanation

        Returns:
            Dictionary containing:
                - 'slide_id': Slide identifier
                - 'top_patches': Indices of top-k attention patches
                - 'top_attention_values': Attention values for top patches
                - 'attention_statistics': Statistics (mean, std, max, min)
                - 'feature_importance': Feature importance scores if feature_names provided

        Example:
            >>> weights = np.array([0.1, 0.3, 0.6, 0.2, 0.4])
            >>> explanation = generator.generate_feature_importance_explanation(
            ...     "slide_001",
            ...     weights,
            ...     feature_names=['texture', 'color', 'shape'],
            ...     top_k=3
            ... )
            >>> print(explanation['top_patches'])  # [2, 4, 1]
        """
        # Validate attention weights sum to 1.0
        weight_sum = np.sum(attention_weights)
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            logger.warning(
                f"Attention weights sum to {weight_sum:.6f}, expected 1.0. "
                f"This may indicate unnormalized weights."
            )

        # Get top-k patches
        top_k_indices = np.argsort(attention_weights)[-top_k:][::-1]
        top_k_values = attention_weights[top_k_indices]

        # Compute statistics
        statistics = {
            "mean": float(np.mean(attention_weights)),
            "std": float(np.std(attention_weights)),
            "max": float(np.max(attention_weights)),
            "min": float(np.min(attention_weights)),
            "median": float(np.median(attention_weights)),
            "sum": float(weight_sum),
        }

        explanation = {
            "slide_id": slide_id,
            "num_patches": len(attention_weights),
            "top_patches": top_k_indices.tolist(),
            "top_attention_values": top_k_values.tolist(),
            "attention_statistics": statistics,
        }

        # Add feature importance if feature names provided
        if feature_names is not None:
            # Compute feature importance as weighted average
            # This is a simplified version - in practice, would use actual feature vectors
            feature_importance = {
                name: float(np.mean(attention_weights)) for name in feature_names[:top_k]
            }
            explanation["feature_importance"] = feature_importance

        logger.info(f"Generated feature importance explanation for {slide_id}")
        return explanation
