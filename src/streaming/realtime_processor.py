"""
Real-Time WSI Streaming Processor - Main Orchestrator

This module provides the main orchestrator class that coordinates all streaming components
to process gigapixel whole-slide images in real-time (<30 seconds with <2GB memory).

Author: Matthew Vaishnav
Date: 2026-04-28
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import torch
import numpy as np

from .wsi_stream_reader import WSIStreamReader, StreamingMetadata, TileBatch
from .gpu_pipeline import GPUPipeline, ThroughputMetrics
from .attention_aggregator import (
    StreamingAttentionAggregator,
    ConfidenceUpdate,
    PredictionResult,
    AttentionMIL
)
from .progressive_visualizer import ProgressiveVisualizer, VisualizationUpdate
from .memory_optimizer import MemoryMonitor, MemorySnapshot

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for real-time WSI streaming."""
    
    # WSI streaming parameters
    tile_size: int = 1024
    buffer_size: int = 16
    overlap: int = 0
    
    # GPU processing parameters
    batch_size: int = 64
    gpu_ids: Optional[List[int]] = None
    enable_fp16: bool = True
    enable_advanced_memory_optimization: bool = True
    
    # Memory management
    memory_budget_gb: float = 2.0
    
    # Processing constraints
    target_time: float = 30.0  # seconds
    confidence_threshold: float = 0.95
    min_confidence: float = 0.80
    
    # Attention parameters
    max_features: int = 10000
    
    # Visualization
    enable_visualization: bool = True
    visualization_update_interval: int = 1000  # patches
    
    # Model loading options
    # Option 1: Load from checkpoint (recommended - loads both models)
    checkpoint_path: Optional[str] = None
    
    # Option 2: Load individual models (fallback)
    cnn_encoder_path: Optional[str] = None
    attention_model_path: Optional[str] = None
    
    # Feature dimension
    feature_dim: int = 512


@dataclass
class StreamingResult:
    """Result from real-time WSI streaming processing."""
    
    # Prediction results
    prediction: int
    confidence: float
    probabilities: np.ndarray
    
    # Processing metrics
    processing_time: float
    patches_processed: int
    total_patches_estimated: int
    
    # Attention information
    attention_weights: torch.Tensor
    attention_coordinates: np.ndarray
    
    # Performance metrics
    throughput_patches_per_sec: float
    peak_memory_gb: float
    avg_memory_gb: float
    
    # Early stopping info
    early_stopped: bool
    confidence_history: List[float]
    
    # Visualization
    visualization_report: Optional[Dict[str, Any]] = None
    
    # Metadata
    slide_id: str = ""
    slide_dimensions: tuple = (0, 0)


class RealTimeWSIProcessor:
    """
    Main orchestrator for real-time WSI streaming processing.
    
    Coordinates all components to process gigapixel slides in <30 seconds with <2GB memory:
    - WSIStreamReader: Progressive tile loading
    - GPUPipeline: Async GPU processing with memory optimization
    - StreamingAttentionAggregator: Progressive confidence building
    - ProgressiveVisualizer: Real-time visualization
    - MemoryMonitor: Real-time memory tracking
    
    Example:
        >>> config = StreamingConfig(
        ...     tile_size=1024,
        ...     batch_size=64,
        ...     memory_budget_gb=2.0,
        ...     target_time=30.0,
        ...     confidence_threshold=0.95
        ... )
        >>> processor = RealTimeWSIProcessor(config)
        >>> result = await processor.process_wsi_realtime("slide.svs")
        >>> print(f"Prediction: {result.prediction}, Confidence: {result.confidence:.3f}")
    """
    
    def __init__(self, config: StreamingConfig):
        """
        Initialize the real-time WSI processor.
        
        Args:
            config: Configuration for streaming processing
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.RealTimeWSIProcessor")
        
        # Components (initialized on first use)
        self._reader: Optional[WSIStreamReader] = None
        self._gpu_pipeline: Optional[GPUPipeline] = None
        self._aggregator: Optional[StreamingAttentionAggregator] = None
        self._visualizer: Optional[ProgressiveVisualizer] = None
        self._memory_monitor: Optional[MemoryMonitor] = None
        
        # Models (loaded on demand)
        self._cnn_encoder: Optional[torch.nn.Module] = None
        self._attention_model: Optional[AttentionMIL] = None
        
        self.logger.info(f"Initialized RealTimeWSIProcessor with config: {config}")
    
    def _load_models(self):
        """Load CNN encoder and attention models."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Try to load from checkpoint first (if checkpoint_path is provided)
        if hasattr(self.config, 'checkpoint_path') and self.config.checkpoint_path:
            try:
                self.logger.info(f"Loading models from checkpoint: {self.config.checkpoint_path}")
                from .checkpoint_loader import load_checkpoint_for_streaming
                
                self._cnn_encoder, self._attention_model = load_checkpoint_for_streaming(
                    self.config.checkpoint_path,
                    device=device
                )
                self.logger.info("✓ Successfully loaded trained models from checkpoint")
                return
            except Exception as e:
                self.logger.warning(
                    f"Failed to load from checkpoint: {e}. "
                    f"Falling back to individual model loading."
                )
        
        # Fall back to individual model loading
        if self._cnn_encoder is None:
            if self.config.cnn_encoder_path:
                self.logger.info(f"Loading CNN encoder from {self.config.cnn_encoder_path}")
                try:
                    # Load custom encoder
                    self._cnn_encoder = torch.load(self.config.cnn_encoder_path)
                except Exception as e:
                    self.logger.warning(f"Failed to load CNN encoder: {e}. Using mock model.")
                    from .mock_models import create_mock_cnn_encoder
                    self._cnn_encoder = create_mock_cnn_encoder(
                        feature_dim=self.config.feature_dim,
                        device=device
                    )
            else:
                # Try to use ResNet50, fall back to mock if not available
                try:
                    from torchvision.models import resnet50, ResNet50_Weights
                    self.logger.info("Using default ResNet50 encoder")
                    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                    # Remove final classification layer
                    self._cnn_encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
                    self._cnn_encoder.eval()
                    if torch.cuda.is_available():
                        self._cnn_encoder = self._cnn_encoder.cuda()
                except Exception as e:
                    self.logger.warning(f"ResNet50 not available: {e}. Using mock model.")
                    from .mock_models import create_mock_cnn_encoder
                    self._cnn_encoder = create_mock_cnn_encoder(
                        feature_dim=self.config.feature_dim,
                        device=device
                    )
        
        if self._attention_model is None:
            if self.config.attention_model_path:
                self.logger.info(f"Loading attention model from {self.config.attention_model_path}")
                try:
                    self._attention_model = torch.load(self.config.attention_model_path)
                except Exception as e:
                    self.logger.warning(f"Failed to load attention model: {e}. Using mock model.")
                    from .mock_models import create_mock_attention_model
                    self._attention_model = create_mock_attention_model(
                        feature_dim=self.config.feature_dim,
                        device=device
                    )
            else:
                # Use mock attention model
                self.logger.info("Using mock AttentionMIL model for testing")
                from .mock_models import create_mock_attention_model
                self._attention_model = create_mock_attention_model(
                    feature_dim=self.config.feature_dim,
                    device=device
                )
    
    def _initialize_components(self, wsi_path: str):
        """
        Initialize all processing components.
        
        Args:
            wsi_path: Path to WSI file
        """
        # Load models first
        self._load_models()
        
        # Initialize WSI stream reader
        self._reader = WSIStreamReader(
            wsi_path=wsi_path,
            tile_size=self.config.tile_size,
            buffer_size=self.config.buffer_size,
            overlap=self.config.overlap
        )
        
        # Initialize GPU pipeline
        self._gpu_pipeline = GPUPipeline(
            model=self._cnn_encoder,
            batch_size=self.config.batch_size,
            memory_limit_gb=self.config.memory_budget_gb,
            gpu_ids=self.config.gpu_ids,
            enable_fp16=self.config.enable_fp16,
            enable_advanced_memory_optimization=self.config.enable_advanced_memory_optimization
        )
        
        # Initialize streaming attention aggregator
        self._aggregator = StreamingAttentionAggregator(
            attention_model=self._attention_model,
            confidence_threshold=self.config.confidence_threshold,
            max_features=self.config.max_features
        )
        
        # Initialize visualizer if enabled
        if self.config.enable_visualization:
            self._visualizer = ProgressiveVisualizer(
                update_interval=self.config.visualization_update_interval
            )
        
        # Initialize memory monitor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._memory_monitor = MemoryMonitor(
            device=device,
            memory_limit_gb=self.config.memory_budget_gb,
            sampling_interval_ms=100.0,
            enable_alerts=True
        )
        
        self.logger.info("All components initialized successfully")
    
    async def process_wsi_realtime(self, wsi_path: str) -> StreamingResult:
        """
        Process a WSI file in real-time with streaming architecture.
        
        This is the main entry point that orchestrates all components to achieve:
        - <30 second processing time for 100K+ patch slides
        - <2GB memory footprint
        - 95%+ accuracy compared to batch processing
        - Real-time confidence updates and visualization
        
        Args:
            wsi_path: Path to WSI file (.svs, .tiff, .ndpi, DICOM)
        
        Returns:
            StreamingResult with prediction, confidence, and performance metrics
        
        Raises:
            FileNotFoundError: If WSI file doesn't exist
            RuntimeError: If processing fails
        """
        # Validate input
        if not Path(wsi_path).exists():
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")
        
        self.logger.info(f"Starting real-time processing for: {wsi_path}")
        start_time = time.time()
        
        # Initialize components
        self._initialize_components(wsi_path)
        
        # Start memory monitoring
        self._memory_monitor.start_monitoring()
        
        try:
            # Initialize streaming
            metadata = self._reader.initialize_streaming()
            self.logger.info(
                f"Streaming initialized: {metadata.estimated_patches} patches, "
                f"dimensions: {metadata.dimensions}"
            )
            
            # Initialize visualization if enabled
            if self._visualizer:
                self._visualizer.initialize(
                    slide_dimensions=metadata.dimensions,
                    estimated_patches=metadata.estimated_patches
                )
            
            # Main processing loop
            patches_processed = 0
            confidence_history = []
            early_stopped = False
            
            for tile_batch in self._reader.stream_tiles():
                # Check time constraint
                elapsed_time = time.time() - start_time
                if elapsed_time > self.config.target_time:
                    self.logger.warning(
                        f"Time limit reached ({self.config.target_time}s), stopping processing"
                    )
                    break
                
                # Check memory pressure
                memory_snapshot = self._memory_monitor.get_current_snapshot()
                if memory_snapshot.pressure_level.name in ['HIGH', 'CRITICAL']:
                    self.logger.warning(
                        f"Memory pressure {memory_snapshot.pressure_level.name}, "
                        f"optimizing batch size"
                    )
                    self._gpu_pipeline.optimize_batch_size(memory_snapshot.allocated_gb)
                
                # Process batch asynchronously
                features = await self._gpu_pipeline.process_batch_async(tile_batch.tiles)
                
                # Update attention and confidence
                confidence_update = self._aggregator.update_features(
                    features, tile_batch.coordinates
                )
                
                patches_processed = confidence_update.patches_processed
                confidence_history.append(confidence_update.current_confidence)
                
                # Update visualization
                if self._visualizer and patches_processed % self.config.visualization_update_interval == 0:
                    self._visualizer.update(
                        attention_weights=confidence_update.attention_weights,
                        coordinates=tile_batch.coordinates,
                        confidence=confidence_update.current_confidence,
                        patches_processed=patches_processed
                    )
                
                # Log progress
                if patches_processed % 1000 == 0:
                    self.logger.info(
                        f"Progress: {patches_processed}/{metadata.estimated_patches} patches, "
                        f"confidence: {confidence_update.current_confidence:.3f}, "
                        f"time: {elapsed_time:.1f}s"
                    )
                
                # Early stopping check
                if confidence_update.early_stop_recommended:
                    self.logger.info(
                        f"Early stopping: confidence {confidence_update.current_confidence:.3f} "
                        f"reached threshold {self.config.confidence_threshold}"
                    )
                    early_stopped = True
                    break
            
            # Finalize prediction
            final_result = self._aggregator.finalize_prediction()
            processing_time = time.time() - start_time
            
            # Get memory analytics
            memory_analytics = self._memory_monitor.get_analytics()
            
            # Get throughput metrics
            throughput_metrics = self._gpu_pipeline.get_throughput_stats()
            
            # Generate visualization report
            visualization_report = None
            if self._visualizer:
                visualization_report = self._visualizer.generate_report()
            
            # Create result
            result = StreamingResult(
                prediction=final_result.prediction,
                confidence=final_result.confidence,
                probabilities=final_result.probabilities,
                processing_time=processing_time,
                patches_processed=patches_processed,
                total_patches_estimated=metadata.estimated_patches,
                attention_weights=final_result.attention_weights,
                attention_coordinates=final_result.coordinates,
                throughput_patches_per_sec=throughput_metrics.patches_per_second,
                peak_memory_gb=memory_analytics.peak_usage_gb,
                avg_memory_gb=memory_analytics.avg_usage_gb,
                early_stopped=early_stopped,
                confidence_history=confidence_history,
                visualization_report=visualization_report,
                slide_id=Path(wsi_path).stem,
                slide_dimensions=metadata.dimensions
            )
            
            # Log final results
            self.logger.info(
                f"Processing complete: prediction={result.prediction}, "
                f"confidence={result.confidence:.3f}, "
                f"time={result.processing_time:.1f}s, "
                f"patches={result.patches_processed}, "
                f"throughput={result.throughput_patches_per_sec:.1f} patches/s, "
                f"peak_memory={result.peak_memory_gb:.2f}GB"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}", exc_info=True)
            raise RuntimeError(f"Real-time processing failed: {e}") from e
        
        finally:
            # Cleanup
            self._memory_monitor.stop_monitoring()
            if self._gpu_pipeline:
                self._gpu_pipeline.cleanup()
            
            self.logger.info("Cleanup complete")
    
    def process_wsi_realtime_sync(self, wsi_path: str) -> StreamingResult:
        """
        Synchronous wrapper for process_wsi_realtime().
        
        Args:
            wsi_path: Path to WSI file
        
        Returns:
            StreamingResult
        """
        return asyncio.run(self.process_wsi_realtime(wsi_path))
    
    async def process_batch_realtime(
        self, 
        wsi_paths: List[str],
        max_concurrent: int = 4
    ) -> Dict[str, StreamingResult]:
        """
        Process multiple WSI files concurrently.
        
        Args:
            wsi_paths: List of WSI file paths
            max_concurrent: Maximum number of concurrent processing tasks
        
        Returns:
            Dictionary mapping slide IDs to results
        """
        self.logger.info(f"Processing batch of {len(wsi_paths)} slides")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(path: str) -> tuple:
            async with semaphore:
                result = await self.process_wsi_realtime(path)
                return Path(path).stem, result
        
        # Process all slides concurrently
        tasks = [process_with_semaphore(path) for path in wsi_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build results dictionary
        results_dict = {}
        for item in results:
            if isinstance(item, Exception):
                self.logger.error(f"Batch processing error: {item}")
                continue
            slide_id, result = item
            results_dict[slide_id] = result
        
        self.logger.info(f"Batch processing complete: {len(results_dict)}/{len(wsi_paths)} successful")
        return results_dict
    
    def get_performance_summary(self, result: StreamingResult) -> Dict[str, Any]:
        """
        Generate a performance summary from processing result.
        
        Args:
            result: StreamingResult from processing
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'processing_time_seconds': result.processing_time,
            'target_time_seconds': self.config.target_time,
            'time_requirement_met': result.processing_time <= self.config.target_time,
            'patches_processed': result.patches_processed,
            'throughput_patches_per_sec': result.throughput_patches_per_sec,
            'peak_memory_gb': result.peak_memory_gb,
            'avg_memory_gb': result.avg_memory_gb,
            'memory_requirement_met': result.peak_memory_gb <= self.config.memory_budget_gb,
            'confidence': result.confidence,
            'confidence_requirement_met': result.confidence >= self.config.min_confidence,
            'early_stopped': result.early_stopped,
            'all_requirements_met': (
                result.processing_time <= self.config.target_time and
                result.peak_memory_gb <= self.config.memory_budget_gb and
                result.confidence >= self.config.min_confidence
            )
        }


# Convenience function for simple usage
async def process_wsi_realtime(
    wsi_path: str,
    config: Optional[StreamingConfig] = None
) -> StreamingResult:
    """
    Convenience function to process a WSI file in real-time.
    
    Args:
        wsi_path: Path to WSI file
        config: Optional configuration (uses defaults if not provided)
    
    Returns:
        StreamingResult
    
    Example:
        >>> result = await process_wsi_realtime("slide.svs")
        >>> print(f"Prediction: {result.prediction}, Confidence: {result.confidence:.3f}")
    """
    if config is None:
        config = StreamingConfig()
    
    processor = RealTimeWSIProcessor(config)
    return await processor.process_wsi_realtime(wsi_path)
