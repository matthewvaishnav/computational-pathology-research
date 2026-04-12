"""
Visualization utilities for attention-based MIL models.

This module provides tools for visualizing attention weights as heatmaps
overlaid on whole-slide images, and timeline visualizations for longitudinal
patient tracking.
"""

from src.visualization.attention_heatmap import AttentionHeatmapGenerator
from src.visualization.timeline import TimelineVisualizer

__all__ = ["AttentionHeatmapGenerator", "TimelineVisualizer"]
