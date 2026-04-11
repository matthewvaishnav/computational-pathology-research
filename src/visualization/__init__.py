"""
Visualization utilities for attention-based MIL models.

This module provides tools for visualizing attention weights as heatmaps
overlaid on whole-slide images.
"""

from src.visualization.attention_heatmap import AttentionHeatmapGenerator

__all__ = ["AttentionHeatmapGenerator"]
