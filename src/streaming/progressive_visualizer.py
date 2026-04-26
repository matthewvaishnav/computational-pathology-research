"""Progressive visualization for real-time WSI streaming."""

import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading
from queue import Queue, Empty

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

logger = logging.getLogger(__name__)


@dataclass
class VisualizationUpdate:
    """Update for progressive visualization."""
    timestamp: float
    patches_processed: int
    confidence: float
    attention_weights: Optional[np.ndarray] = None
    coordinates: Optional[np.ndarray] = None
    heatmap_data: Optional[np.ndarray] = None


class ProgressiveVisualizer:
    """Real-time visualization for streaming WSI processing.
    
    Features:
    - Real-time attention heatmap updates
    - Confidence progression plotting
    - Processing statistics dashboard
    - Export to PNG, PDF, SVG formats
    """
    
    def __init__(self, output_dir: str, slide_dimensions: Tuple[int, int],
                 tile_size: int = 1024, update_interval: float = 1.0):
        """Initialize progressive visualizer.
        
        Args:
            output_dir: Directory for saving visualizations
            slide_dimensions: (width, height) of WSI in pixels
            tile_size: Size of tiles being processed
            update_interval: Minimum seconds between visualization updates
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.slide_dimensions = slide_dimensions
        self.tile_size = tile_size
        self.update_interval = update_interval
        
        # Calculate heatmap dimensions
        self.heatmap_width = (slide_dimensions[0] + tile_size - 1) // tile_size
        self.heatmap_height = (slide_dimensions[1] + tile_size - 1) // tile_size
        
        # Initialize heatmap data
        self.attention_heatmap = np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.float32)
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
        
        logger.info(f"Initialized ProgressiveVisualizer: {self.heatmap_width}x{self.heatmap_height} heatmap")
    
    def _create_attention_colormap(self) -> LinearSegmentedColormap:
        """Create custom colormap for attention visualization."""
        # Blue (low) -> Yellow (medium) -> Red (high)
        colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('attention', colors, N=n_bins)
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
    
    def update_attention_heatmap(self, attention_weights: np.ndarray, 
                                coordinates: np.ndarray, confidence: float,
                                patches_processed: int):
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
            coordinates=coordinates.copy()
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
            
            logger.debug(f"Updated visualizations: {update.patches_processed} patches, "
                        f"confidence={update.confidence:.3f}")
            
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
        im = ax.imshow(normalized_heatmap, cmap=self.colormap, 
                      interpolation='bilinear', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        # Labels and title
        ax.set_xlabel('Tile X')
        ax.set_ylabel('Tile Y')
        ax.set_title(f'Real-Time Attention Heatmap\n{patches_processed} patches processed')
        
        # Save
        output_path = self.output_dir / 'attention_heatmap_realtime.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
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
        ax.plot(timestamps, confidences, 'b-', linewidth=2, label='Confidence')
        ax.axhline(y=0.95, color='r', linestyle='--', linewidth=1, label='Target (0.95)')
        
        # Labels and title
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Confidence')
        ax.set_title('Real-Time Confidence Progression')
        ax.set_ylim([0.0, 1.0])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Save
        output_path = self.output_dir / 'confidence_progression.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    def save_final_visualizations(self, export_formats: List[str] = ['png']):
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
        im = ax.imshow(normalized_heatmap, cmap=self.colormap, 
                      interpolation='bilinear', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Attention Weight', rotation=270, labelpad=25, fontsize=12)
        
        # Labels and title
        ax.set_xlabel('Tile X', fontsize=12)
        ax.set_ylabel('Tile Y', fontsize=12)
        ax.set_title('Final Attention Heatmap', fontsize=14, fontweight='bold')
        
        # Save in requested formats
        for fmt in export_formats:
            output_path = self.output_dir / f'attention_heatmap_final.{fmt}'
            plt.savefig(output_path, bbox_inches='tight', dpi=300, format=fmt)
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
        ax.plot(timestamps, confidences, 'b-', linewidth=2.5, label='Confidence')
        ax.axhline(y=0.95, color='r', linestyle='--', linewidth=1.5, label='Target (0.95)')
        
        # Fill area under curve
        ax.fill_between(timestamps, 0, confidences, alpha=0.2, color='blue')
        
        # Labels and title
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_title('Confidence Progression Over Time', fontsize=14, fontweight='bold')
        ax.set_ylim([0.0, 1.0])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Save in requested formats
        for fmt in export_formats:
            output_path = self.output_dir / f'confidence_progression_final.{fmt}'
            plt.savefig(output_path, bbox_inches='tight', dpi=300, format=fmt)
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
        ax1.plot(timestamps, confidences, 'b-', linewidth=2)
        ax1.axhline(y=0.95, color='r', linestyle='--', linewidth=1)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Confidence Progression')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.0, 1.0])
        
        # 2. Coverage heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        coverage_viz = self.coverage_mask.astype(float)
        ax2.imshow(coverage_viz, cmap='Greys', aspect='auto')
        ax2.set_xlabel('Tile X')
        ax2.set_ylabel('Tile Y')
        ax2.set_title(f'Spatial Coverage ({stats["coverage_percent"]:.1f}%)')
        
        # 3. Statistics text
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
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
        
        ax3.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax3.transAxes)
        
        # 4. Attention distribution histogram
        ax4 = fig.add_subplot(gs[2, 0])
        attention_values = self.attention_heatmap[self.coverage_mask]
        if len(attention_values) > 0:
            ax4.hist(attention_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Attention Weight')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Attention Weight Distribution')
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Confidence statistics
        ax5 = fig.add_subplot(gs[2, 1])
        confidence_stats = {
            'Min': confidences.min(),
            'Q1': np.percentile(confidences, 25),
            'Median': np.median(confidences),
            'Q3': np.percentile(confidences, 75),
            'Max': confidences.max()
        }
        
        ax5.boxplot([confidences], vert=True, widths=0.5)
        ax5.set_ylabel('Confidence')
        ax5.set_title('Confidence Distribution')
        ax5.set_xticklabels(['All Updates'])
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim([0.0, 1.0])
        
        # Overall title
        fig.suptitle('Real-Time WSI Streaming - Processing Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save in requested formats
        for fmt in export_formats:
            output_path = self.output_dir / f'processing_dashboard.{fmt}'
            plt.savefig(output_path, bbox_inches='tight', dpi=300, format=fmt)
            logger.info(f"Saved statistics dashboard: {output_path}")
        
        plt.close(fig)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current visualization statistics."""
        coverage_percent = (self.coverage_mask.sum() / self.coverage_mask.size) * 100
        
        stats = {
            'heatmap_dimensions': (self.heatmap_width, self.heatmap_height),
            'coverage_percent': coverage_percent,
            'total_updates': len(self.confidence_history),
            'current_confidence': self.confidence_history[-1][1] if self.confidence_history else 0.0,
            'output_directory': str(self.output_dir)
        }
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        self.start_async_updates()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_async_updates()
        self.save_final_visualizations()
