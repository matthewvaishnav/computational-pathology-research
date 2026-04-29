"""Progressive visualization for real-time WSI streaming."""

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

# Import plotly for interactive visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)

if not PLOTLY_AVAILABLE:
    logger.warning("Plotly not available. Interactive features will be disabled.")


@dataclass
class VisualizationUpdate:
    """Update for progressive visualization."""

    timestamp: float
    patches_processed: int
    confidence: float
    attention_weights: Optional[np.ndarray] = None
    coordinates: Optional[np.ndarray] = None
    heatmap_data: Optional[np.ndarray] = None


@dataclass
class InteractiveConfig:
    """Configuration for interactive visualization features."""

    enable_zoom_pan: bool = True
    enable_overlay: bool = True
    enable_parameter_controls: bool = True
    update_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    initial_zoom_level: float = 1.0
    max_zoom_level: float = 10.0
    min_zoom_level: float = 0.5


class ProgressiveVisualizer:
    """Real-time visualization for streaming WSI processing.

    Features:
    - Real-time attention heatmap updates
    - Confidence progression plotting
    - Processing statistics dashboard
    - Export to PNG, PDF, SVG formats
    """

    def __init__(
        self,
        output_dir: str,
        slide_dimensions: Tuple[int, int],
        tile_size: int = 1024,
        update_interval: float = 1.0,
        interactive_config: Optional[InteractiveConfig] = None,
    ):
        """Initialize progressive visualizer.

        Args:
            output_dir: Directory for saving visualizations
            slide_dimensions: (width, height) of WSI in pixels
            tile_size: Size of tiles being processed
            update_interval: Minimum seconds between visualization updates
            interactive_config: Configuration for interactive features
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.slide_dimensions = slide_dimensions
        self.tile_size = tile_size
        self.update_interval = update_interval

        # Interactive configuration
        self.interactive_config = interactive_config or InteractiveConfig()

        # Calculate heatmap dimensions
        self.heatmap_width = (slide_dimensions[0] + tile_size - 1) // tile_size
        self.heatmap_height = (slide_dimensions[1] + tile_size - 1) // tile_size

        # Initialize heatmap data
        self.attention_heatmap = np.zeros(
            (self.heatmap_height, self.heatmap_width), dtype=np.float32
        )
        self.coverage_mask = np.zeros((self.heatmap_height, self.heatmap_width), dtype=bool)

        # Tracking data
        self.confidence_history: List[Tuple[float, float]] = []  # (timestamp, confidence)
        self.throughput_history: List[Tuple[float, float]] = []  # (timestamp, patches/sec)
        self.last_update_time = 0.0

        # Thread-safe update queue
        self.update_queue: Queue = Queue()
        self.visualization_thread: Optional[threading.Thread] = None
        self.running = False

        # Colormap for attention heatmaps
        self.colormap = self._create_attention_colormap()

        # Interactive visualization state
        self.current_zoom_level = self.interactive_config.initial_zoom_level
        self.current_pan_offset = (0, 0)
        self.overlay_opacity = 0.6
        self.parameter_values: Dict[str, Any] = {}

        logger.info(
            f"Initialized ProgressiveVisualizer: {self.heatmap_width}x{self.heatmap_height} heatmap"
        )

    def _create_attention_colormap(self) -> LinearSegmentedColormap:
        """Create custom colormap for attention visualization."""
        # Blue (low) -> Yellow (medium) -> Red (high)
        colors = ["#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000"]
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list("attention", colors, N=n_bins)
        return cmap

    def start_async_updates(self):
        """Start background thread for async visualization updates."""
        if self.running:
            logger.warning("Visualization thread already running")
            return

        self.running = True
        self.visualization_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.visualization_thread.start()
        logger.info("Started async visualization updates")

    def stop_async_updates(self):
        """Stop background visualization thread."""
        self.running = False
        if self.visualization_thread:
            self.visualization_thread.join(timeout=5.0)
        logger.info("Stopped async visualization updates")

    def _update_loop(self):
        """Background loop for processing visualization updates."""
        while self.running:
            try:
                # Get update from queue (blocking with timeout)
                update = self.update_queue.get(timeout=0.1)

                # Check if enough time has passed since last update
                current_time = time.time()
                if current_time - self.last_update_time >= self.update_interval:
                    self._process_update(update)
                    self.last_update_time = current_time

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in visualization update loop: {e}")

    def update_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        coordinates: np.ndarray,
        confidence: float,
        patches_processed: int,
    ):
        """Update attention heatmap with new batch data.

        Args:
            attention_weights: Attention weights for batch [batch_size]
            coordinates: Tile coordinates [batch_size, 2] (x, y)
            confidence: Current prediction confidence
            patches_processed: Total patches processed so far
        """
        # Validate inputs
        if attention_weights.shape[0] != coordinates.shape[0]:
            raise ValueError("Attention weights and coordinates must have same batch size")

        # Update heatmap data
        for weight, (x, y) in zip(attention_weights, coordinates):
            if 0 <= x < self.heatmap_width and 0 <= y < self.heatmap_height:
                # Accumulate attention weights (will normalize later)
                self.attention_heatmap[y, x] += weight
                self.coverage_mask[y, x] = True

        # Track confidence
        timestamp = time.time()
        self.confidence_history.append((timestamp, confidence))

        # Queue update for async processing
        update = VisualizationUpdate(
            timestamp=timestamp,
            patches_processed=patches_processed,
            confidence=confidence,
            attention_weights=attention_weights.copy(),
            coordinates=coordinates.copy(),
        )

        if self.running:
            self.update_queue.put(update)
        else:
            # Synchronous update if async not enabled
            if timestamp - self.last_update_time >= self.update_interval:
                self._process_update(update)
                self.last_update_time = timestamp

    def _process_update(self, update: VisualizationUpdate):
        """Process a visualization update (called by background thread)."""
        try:
            # Generate and save heatmap
            self._save_attention_heatmap(update.patches_processed)

            # Update confidence plot
            self._save_confidence_plot()

            logger.debug(
                f"Updated visualizations: {update.patches_processed} patches, "
                f"confidence={update.confidence:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed to process visualization update: {e}")

    def _save_attention_heatmap(self, patches_processed: int):
        """Save current attention heatmap to file."""
        # Normalize heatmap by coverage
        normalized_heatmap = np.zeros_like(self.attention_heatmap)
        normalized_heatmap[self.coverage_mask] = self.attention_heatmap[self.coverage_mask]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10), dpi=100)

        # Plot heatmap
        im = ax.imshow(
            normalized_heatmap, cmap=self.colormap, interpolation="bilinear", aspect="auto"
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention Weight", rotation=270, labelpad=20)

        # Labels and title
        ax.set_xlabel("Tile X")
        ax.set_ylabel("Tile Y")
        ax.set_title(f"Real-Time Attention Heatmap\n{patches_processed} patches processed")

        # Save
        output_path = self.output_dir / "attention_heatmap_realtime.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _save_confidence_plot(self):
        """Save confidence progression plot."""
        if len(self.confidence_history) < 2:
            return

        # Extract data
        timestamps = np.array([t for t, _ in self.confidence_history])
        confidences = np.array([c for _, c in self.confidence_history])

        # Normalize timestamps to start at 0
        timestamps = timestamps - timestamps[0]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        # Plot confidence over time
        ax.plot(timestamps, confidences, "b-", linewidth=2, label="Confidence")
        ax.axhline(y=0.95, color="r", linestyle="--", linewidth=1, label="Target (0.95)")

        # Labels and title
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Confidence")
        ax.set_title("Real-Time Confidence Progression")
        ax.set_ylim([0.0, 1.0])
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Save
        output_path = self.output_dir / "confidence_progression.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    def save_final_visualizations(self, export_formats: List[str] = ["png"]):
        """Save final high-quality visualizations.

        Args:
            export_formats: List of formats to export ('png', 'pdf', 'svg')
        """
        logger.info("Generating final visualizations...")

        # Normalize final heatmap
        normalized_heatmap = np.zeros_like(self.attention_heatmap)
        if self.coverage_mask.any():
            normalized_heatmap[self.coverage_mask] = self.attention_heatmap[self.coverage_mask]
            # Normalize to [0, 1]
            max_val = normalized_heatmap.max()
            if max_val > 0:
                normalized_heatmap = normalized_heatmap / max_val

        # Create high-quality figure
        fig, ax = plt.subplots(figsize=(16, 12), dpi=300)

        # Plot heatmap
        im = ax.imshow(
            normalized_heatmap, cmap=self.colormap, interpolation="bilinear", aspect="auto"
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Attention Weight", rotation=270, labelpad=25, fontsize=12)

        # Labels and title
        ax.set_xlabel("Tile X", fontsize=12)
        ax.set_ylabel("Tile Y", fontsize=12)
        ax.set_title("Final Attention Heatmap", fontsize=14, fontweight="bold")

        # Save in requested formats
        for fmt in export_formats:
            output_path = self.output_dir / f"attention_heatmap_final.{fmt}"
            plt.savefig(output_path, bbox_inches="tight", dpi=300, format=fmt)
            logger.info(f"Saved final heatmap: {output_path}")

        plt.close(fig)

        # Save final confidence plot
        self._save_final_confidence_plot(export_formats)

        # Save processing statistics dashboard
        self._save_statistics_dashboard(export_formats)

    def _save_final_confidence_plot(self, export_formats: List[str]):
        """Save final confidence progression plot."""
        if len(self.confidence_history) < 2:
            return

        # Extract data
        timestamps = np.array([t for t, _ in self.confidence_history])
        confidences = np.array([c for _, c in self.confidence_history])
        timestamps = timestamps - timestamps[0]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

        # Plot confidence
        ax.plot(timestamps, confidences, "b-", linewidth=2.5, label="Confidence")
        ax.axhline(y=0.95, color="r", linestyle="--", linewidth=1.5, label="Target (0.95)")

        # Fill area under curve
        ax.fill_between(timestamps, 0, confidences, alpha=0.2, color="blue")

        # Labels and title
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Confidence", fontsize=12)
        ax.set_title("Confidence Progression Over Time", fontsize=14, fontweight="bold")
        ax.set_ylim([0.0, 1.0])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Save in requested formats
        for fmt in export_formats:
            output_path = self.output_dir / f"confidence_progression_final.{fmt}"
            plt.savefig(output_path, bbox_inches="tight", dpi=300, format=fmt)
            logger.info(f"Saved final confidence plot: {output_path}")

        plt.close(fig)

    def _save_statistics_dashboard(self, export_formats: List[str]):
        """Save processing statistics dashboard."""
        if not self.confidence_history:
            return

        # Calculate statistics
        stats = self.get_statistics()

        timestamps = np.array([t for t, _ in self.confidence_history])
        confidences = np.array([c for _, c in self.confidence_history])
        timestamps = timestamps - timestamps[0]

        total_time = timestamps[-1] if len(timestamps) > 0 else 0.0
        final_confidence = confidences[-1] if len(confidences) > 0 else 0.0
        avg_confidence = confidences.mean() if len(confidences) > 0 else 0.0

        # Calculate throughput if we have throughput history
        avg_throughput = 0.0
        if self.throughput_history:
            throughputs = np.array([t for _, t in self.throughput_history])
            avg_throughput = throughputs.mean()

        # Create dashboard figure with subplots
        fig = plt.figure(figsize=(16, 10), dpi=300)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Confidence over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(timestamps, confidences, "b-", linewidth=2)
        ax1.axhline(y=0.95, color="r", linestyle="--", linewidth=1)
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Confidence")
        ax1.set_title("Confidence Progression")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.0, 1.0])

        # 2. Coverage heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        coverage_viz = self.coverage_mask.astype(float)
        ax2.imshow(coverage_viz, cmap="Greys", aspect="auto")
        ax2.set_xlabel("Tile X")
        ax2.set_ylabel("Tile Y")
        ax2.set_title(f'Spatial Coverage ({stats["coverage_percent"]:.1f}%)')

        # 3. Statistics text
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis("off")

        stats_text = f"""
        Processing Statistics
        ═══════════════════════════
        
        Total Time: {total_time:.2f}s
        Final Confidence: {final_confidence:.4f}
        Average Confidence: {avg_confidence:.4f}
        
        Heatmap Size: {stats['heatmap_dimensions'][0]}×{stats['heatmap_dimensions'][1]}
        Coverage: {stats['coverage_percent']:.2f}%
        Total Updates: {stats['total_updates']}
        
        Avg Throughput: {avg_throughput:.1f} patches/sec
        """

        ax3.text(
            0.1,
            0.5,
            stats_text,
            fontsize=11,
            family="monospace",
            verticalalignment="center",
            transform=ax3.transAxes,
        )

        # 4. Attention distribution histogram
        ax4 = fig.add_subplot(gs[2, 0])
        attention_values = self.attention_heatmap[self.coverage_mask]
        if len(attention_values) > 0:
            ax4.hist(attention_values, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
            ax4.set_xlabel("Attention Weight")
            ax4.set_ylabel("Frequency")
            ax4.set_title("Attention Weight Distribution")
            ax4.grid(True, alpha=0.3, axis="y")

        # 5. Confidence statistics
        ax5 = fig.add_subplot(gs[2, 1])
        confidence_stats = {
            "Min": confidences.min(),
            "Q1": np.percentile(confidences, 25),
            "Median": np.median(confidences),
            "Q3": np.percentile(confidences, 75),
            "Max": confidences.max(),
        }

        ax5.boxplot([confidences], vert=True, widths=0.5)
        ax5.set_ylabel("Confidence")
        ax5.set_title("Confidence Distribution")
        ax5.set_xticklabels(["All Updates"])
        ax5.grid(True, alpha=0.3, axis="y")
        ax5.set_ylim([0.0, 1.0])

        # Overall title
        fig.suptitle(
            "Real-Time WSI Streaming - Processing Dashboard", fontsize=16, fontweight="bold", y=0.98
        )

        # Save in requested formats
        for fmt in export_formats:
            output_path = self.output_dir / f"processing_dashboard.{fmt}"
            plt.savefig(output_path, bbox_inches="tight", dpi=300, format=fmt)
            logger.info(f"Saved statistics dashboard: {output_path}")

        plt.close(fig)

    def get_statistics(self) -> Dict[str, Any]:
        """Get current visualization statistics."""
        coverage_percent = (self.coverage_mask.sum() / self.coverage_mask.size) * 100

        stats = {
            "heatmap_dimensions": (self.heatmap_width, self.heatmap_height),
            "coverage_percent": coverage_percent,
            "total_updates": len(self.confidence_history),
            "current_confidence": (
                self.confidence_history[-1][1] if self.confidence_history else 0.0
            ),
            "output_directory": str(self.output_dir),
        }

        return stats

    # ========== Interactive Visualization Features ==========

    def create_interactive_heatmap(
        self, slide_thumbnail: Optional[np.ndarray] = None
    ) -> Optional[go.Figure]:
        """Create interactive attention heatmap with zoom and pan capabilities.

        Args:
            slide_thumbnail: Optional slide thumbnail image for overlay [H, W, 3]

        Returns:
            Plotly Figure object with interactive controls, or None if Plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create interactive heatmap.")
            return None

        # Normalize heatmap
        normalized_heatmap = np.zeros_like(self.attention_heatmap)
        if self.coverage_mask.any():
            normalized_heatmap[self.coverage_mask] = self.attention_heatmap[self.coverage_mask]
            max_val = normalized_heatmap.max()
            if max_val > 0:
                normalized_heatmap = normalized_heatmap / max_val

        # Create figure
        fig = go.Figure()

        # Add slide thumbnail as background if provided
        if slide_thumbnail is not None and self.interactive_config.enable_overlay:
            fig.add_trace(
                go.Image(
                    z=slide_thumbnail, name="Slide Thumbnail", opacity=1.0 - self.overlay_opacity
                )
            )

        # Add attention heatmap
        fig.add_trace(
            go.Heatmap(
                z=normalized_heatmap,
                colorscale="Jet",
                name="Attention Weights",
                opacity=self.overlay_opacity,
                colorbar=dict(
                    title="Attention Weight", x=1.02, tickmode="linear", tick0=0, dtick=0.2
                ),
                hovertemplate="X: %{x}<br>Y: %{y}<br>Attention: %{z:.3f}<extra></extra>",
            )
        )

        # Configure layout with zoom and pan
        fig.update_layout(
            title="Interactive Attention Heatmap",
            xaxis=dict(title="Tile X", scaleanchor="y", scaleratio=1, constrain="domain"),
            yaxis=dict(title="Tile Y", constrain="domain"),
            width=1200,
            height=1000,
            dragmode="pan" if self.interactive_config.enable_zoom_pan else "zoom",
            hovermode="closest",
        )

        # Add zoom and pan controls
        if self.interactive_config.enable_zoom_pan:
            fig.update_xaxes(
                range=[
                    self.current_pan_offset[0],
                    self.current_pan_offset[0] + self.heatmap_width / self.current_zoom_level,
                ]
            )
            fig.update_yaxes(
                range=[
                    self.current_pan_offset[1],
                    self.current_pan_offset[1] + self.heatmap_height / self.current_zoom_level,
                ]
            )

        return fig

    def create_interactive_confidence_plot(self) -> Optional[go.Figure]:
        """Create interactive confidence progression plot.

        Returns:
            Plotly Figure object with interactive controls, or None if Plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create interactive confidence plot.")
            return None

        if len(self.confidence_history) < 2:
            return None

        # Extract data
        timestamps = np.array([t for t, _ in self.confidence_history])
        confidences = np.array([c for _, c in self.confidence_history])
        timestamps = timestamps - timestamps[0]

        # Create figure
        fig = go.Figure()

        # Add confidence trace
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidences,
                mode="lines+markers",
                name="Confidence",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
                hovertemplate="Time: %{x:.2f}s<br>Confidence: %{y:.4f}<extra></extra>",
            )
        )

        # Add target threshold line
        fig.add_hline(
            y=0.95,
            line_dash="dash",
            line_color="red",
            annotation_text="Target (0.95)",
            annotation_position="right",
        )

        # Configure layout
        fig.update_layout(
            title="Real-Time Confidence Progression",
            xaxis_title="Time (seconds)",
            yaxis_title="Confidence",
            yaxis_range=[0.0, 1.0],
            width=1000,
            height=600,
            hovermode="x unified",
            showlegend=True,
        )

        return fig

    def create_interactive_dashboard(
        self, slide_thumbnail: Optional[np.ndarray] = None
    ) -> Optional[go.Figure]:
        """Create comprehensive interactive dashboard with multiple visualizations.

        Args:
            slide_thumbnail: Optional slide thumbnail image for overlay

        Returns:
            Plotly Figure object with subplots, or None if Plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create interactive dashboard.")
            return None

        if not self.confidence_history:
            return None

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Attention Heatmap",
                "Confidence Progression",
                "Spatial Coverage",
                "Attention Distribution",
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "histogram"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
        )

        # 1. Attention heatmap
        normalized_heatmap = np.zeros_like(self.attention_heatmap)
        if self.coverage_mask.any():
            normalized_heatmap[self.coverage_mask] = self.attention_heatmap[self.coverage_mask]
            max_val = normalized_heatmap.max()
            if max_val > 0:
                normalized_heatmap = normalized_heatmap / max_val

        fig.add_trace(
            go.Heatmap(
                z=normalized_heatmap,
                colorscale="Jet",
                showscale=True,
                colorbar=dict(x=0.46, len=0.4),
            ),
            row=1,
            col=1,
        )

        # 2. Confidence progression
        timestamps = np.array([t for t, _ in self.confidence_history])
        confidences = np.array([c for _, c in self.confidence_history])
        timestamps = timestamps - timestamps[0]

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidences,
                mode="lines+markers",
                name="Confidence",
                line=dict(color="blue", width=2),
                marker=dict(size=4),
            ),
            row=1,
            col=2,
        )

        # Add target line
        fig.add_hline(y=0.95, line_dash="dash", line_color="red", row=1, col=2)

        # 3. Spatial coverage
        coverage_viz = self.coverage_mask.astype(float)
        fig.add_trace(go.Heatmap(z=coverage_viz, colorscale="Greys", showscale=False), row=2, col=1)

        # 4. Attention distribution
        attention_values = self.attention_heatmap[self.coverage_mask]
        if len(attention_values) > 0:
            fig.add_trace(
                go.Histogram(
                    x=attention_values, nbinsx=50, name="Attention", marker=dict(color="steelblue")
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title_text="Real-Time WSI Streaming Dashboard", showlegend=False, height=900, width=1400
        )

        # Update axes
        fig.update_xaxes(title_text="Tile X", row=1, col=1)
        fig.update_yaxes(title_text="Tile Y", row=1, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
        fig.update_yaxes(title_text="Confidence", range=[0, 1], row=1, col=2)
        fig.update_xaxes(title_text="Tile X", row=2, col=1)
        fig.update_yaxes(title_text="Tile Y", row=2, col=1)
        fig.update_xaxes(title_text="Attention Weight", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)

        return fig

    def save_interactive_html(
        self,
        filename: str = "interactive_dashboard.html",
        slide_thumbnail: Optional[np.ndarray] = None,
    ) -> Optional[Path]:
        """Save interactive dashboard as standalone HTML file.

        Args:
            filename: Output filename
            slide_thumbnail: Optional slide thumbnail for overlay

        Returns:
            Path to saved HTML file, or None if Plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot save interactive HTML.")
            return None

        fig = self.create_interactive_dashboard(slide_thumbnail)
        if fig is None:
            return None

        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        logger.info(f"Saved interactive dashboard: {output_path}")

        return output_path

    def set_zoom_level(self, zoom_level: float) -> None:
        """Set zoom level for interactive visualizations.

        Args:
            zoom_level: Zoom level (1.0 = normal, >1.0 = zoomed in)
        """
        if not self.interactive_config.enable_zoom_pan:
            logger.warning("Zoom/pan not enabled in configuration")
            return

        # Clamp to valid range
        zoom_level = max(
            self.interactive_config.min_zoom_level,
            min(zoom_level, self.interactive_config.max_zoom_level),
        )

        self.current_zoom_level = zoom_level
        logger.debug(f"Zoom level set to {zoom_level:.2f}x")

    def set_pan_offset(self, x_offset: int, y_offset: int) -> None:
        """Set pan offset for interactive visualizations.

        Args:
            x_offset: Horizontal offset in tile coordinates
            y_offset: Vertical offset in tile coordinates
        """
        if not self.interactive_config.enable_zoom_pan:
            logger.warning("Zoom/pan not enabled in configuration")
            return

        # Clamp to valid range
        x_offset = max(0, min(x_offset, self.heatmap_width))
        y_offset = max(0, min(y_offset, self.heatmap_height))

        self.current_pan_offset = (x_offset, y_offset)
        logger.debug(f"Pan offset set to ({x_offset}, {y_offset})")

    def set_overlay_opacity(self, opacity: float) -> None:
        """Set opacity for attention weight overlay.

        Args:
            opacity: Opacity value between 0.0 (transparent) and 1.0 (opaque)
        """
        if not self.interactive_config.enable_overlay:
            logger.warning("Overlay not enabled in configuration")
            return

        # Clamp to valid range
        opacity = max(0.0, min(1.0, opacity))
        self.overlay_opacity = opacity
        logger.debug(f"Overlay opacity set to {opacity:.2f}")

    def update_parameter(self, param_name: str, param_value: Any) -> None:
        """Update processing parameter and trigger callback if configured.

        Args:
            param_name: Name of parameter to update
            param_value: New parameter value
        """
        if not self.interactive_config.enable_parameter_controls:
            logger.warning("Parameter controls not enabled in configuration")
            return

        # Store parameter value
        old_value = self.parameter_values.get(param_name)
        self.parameter_values[param_name] = param_value

        logger.info(f"Parameter '{param_name}' updated: {old_value} -> {param_value}")

        # Trigger callback if configured
        if self.interactive_config.update_callback is not None:
            try:
                self.interactive_config.update_callback(
                    {
                        "parameter": param_name,
                        "old_value": old_value,
                        "new_value": param_value,
                        "all_parameters": self.parameter_values.copy(),
                    }
                )
            except Exception as e:
                logger.error(f"Error in parameter update callback: {e}")

    def get_parameter(self, param_name: str, default: Any = None) -> Any:
        """Get current value of a parameter.

        Args:
            param_name: Name of parameter
            default: Default value if parameter not set

        Returns:
            Current parameter value or default
        """
        return self.parameter_values.get(param_name, default)

    def export_visualization_state(self) -> Dict[str, Any]:
        """Export current visualization state as JSON-serializable dict.

        Returns:
            Dictionary containing visualization state
        """
        state = {
            "slide_dimensions": self.slide_dimensions,
            "heatmap_dimensions": (self.heatmap_width, self.heatmap_height),
            "tile_size": self.tile_size,
            "zoom_level": self.current_zoom_level,
            "pan_offset": self.current_pan_offset,
            "overlay_opacity": self.overlay_opacity,
            "parameters": self.parameter_values.copy(),
            "statistics": self.get_statistics(),
            "confidence_history": [
                {"timestamp": float(t), "confidence": float(c)} for t, c in self.confidence_history
            ],
        }

        return state

    def save_visualization_state(self, filename: str = "visualization_state.json") -> Path:
        """Save visualization state to JSON file.

        Args:
            filename: Output filename

        Returns:
            Path to saved JSON file
        """
        state = self.export_visualization_state()
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved visualization state: {output_path}")
        return output_path

    def __enter__(self):
        """Context manager entry."""
        self.start_async_updates()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_async_updates()
        self.save_final_visualizations()
