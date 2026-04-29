"""Real-time WSI streaming components for HistoCore."""

from .attention_aggregator import (
    ConfidenceCalibrator,
    ConfidenceUpdate,
    PredictionResult,
    StreamingAttentionAggregator,
)
from .checkpoint_loader import CheckpointLoader, load_checkpoint_for_streaming
from .gpu_pipeline import GPUPipeline, ThroughputMetrics
from .mock_models import (
    AttentionMIL,
    MockAttentionMIL,
    MockCNNEncoder,
    create_mock_attention_model,
    create_mock_cnn_encoder,
    create_mock_models,
)
from .progressive_visualizer import ProgressiveVisualizer, VisualizationUpdate
from .realtime_processor import (
    RealTimeWSIProcessor,
    StreamingConfig,
    StreamingResult,
    process_wsi_realtime,
)
from .web_dashboard import (
    ConfidenceData,
    HeatmapData,
    ProcessingParameters,
    ProcessingStatus,
)
from .web_dashboard import app as dashboard_app
from .web_dashboard import (
    update_dashboard_complete,
    update_dashboard_error,
    update_dashboard_status,
)
from .wsi_stream_reader import StreamingMetadata, StreamingProgress, TileBatch, WSIStreamReader

__all__ = [
    "WSIStreamReader",
    "StreamingMetadata",
    "StreamingProgress",
    "TileBatch",
    "GPUPipeline",
    "ThroughputMetrics",
    "StreamingAttentionAggregator",
    "ConfidenceUpdate",
    "PredictionResult",
    "AttentionMIL",
    "ConfidenceCalibrator",
    "ProgressiveVisualizer",
    "VisualizationUpdate",
    "dashboard_app",
    "ProcessingStatus",
    "HeatmapData",
    "ConfidenceData",
    "ProcessingParameters",
    "update_dashboard_status",
    "update_dashboard_error",
    "update_dashboard_complete",
    "RealTimeWSIProcessor",
    "StreamingConfig",
    "StreamingResult",
    "process_wsi_realtime",
    "CheckpointLoader",
    "load_checkpoint_for_streaming",
    "MockCNNEncoder",
    "MockAttentionMIL",
    "create_mock_cnn_encoder",
    "create_mock_attention_model",
    "create_mock_models",
]
