"""Real-time WSI streaming components for HistoCore."""

from .wsi_stream_reader import WSIStreamReader, StreamingMetadata, StreamingProgress, TileBatch
from .gpu_pipeline import GPUPipeline, ThroughputMetrics
from .attention_aggregator import (
    StreamingAttentionAggregator, 
    ConfidenceUpdate, 
    PredictionResult,
    AttentionMIL,
    ConfidenceCalibrator
)
from .progressive_visualizer import ProgressiveVisualizer, VisualizationUpdate
from .web_dashboard import (
    app as dashboard_app,
    ProcessingStatus,
    HeatmapData,
    ConfidenceData,
    ProcessingParameters,
    update_dashboard_status,
    update_dashboard_error,
    update_dashboard_complete
)

__all__ = [
    'WSIStreamReader',
    'StreamingMetadata', 
    'StreamingProgress',
    'TileBatch',
    'GPUPipeline',
    'ThroughputMetrics',
    'StreamingAttentionAggregator',
    'ConfidenceUpdate',
    'PredictionResult',
    'AttentionMIL',
    'ConfidenceCalibrator',
    'ProgressiveVisualizer',
    'VisualizationUpdate',
    'dashboard_app',
    'ProcessingStatus',
    'HeatmapData',
    'ConfidenceData',
    'ProcessingParameters',
    'update_dashboard_status',
    'update_dashboard_error',
    'update_dashboard_complete'
]