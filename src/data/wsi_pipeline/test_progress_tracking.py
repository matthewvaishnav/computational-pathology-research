"""
Test suite for WSI streaming progress tracking and ETA estimation.

This module tests the comprehensive progress tracking functionality added to
WSIStreamReader, including ETA estimation, confidence tracking, and real-time callbacks.
"""

import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List
import numpy as np

from .wsi_stream_reader import (
    WSIStreamReader, 
    StreamingProgressTracker, 
    StreamingProgress, 
    ProgressCallback,
    StreamingMetadata,
    TileBatch
)
from .tile_buffer_pool import TileBufferConfig
from .exceptions import ProcessingError, ResourceError


class TestStreamingProgressTracker(unittest.TestCase):
    """Test cases for StreamingProgressTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.total_tiles = 100
        self.confidence_threshold = 0.95
        self.target_time = 30.0
        
        self.tracker = StreamingProgressTracker(
            total_tiles=self.total_tiles,
            confidence_threshold=self.confidence_threshold,
            target_processing_time=self.target_time
        )
    
    def test_initialization(self):
        """Test progress tracker initialization."""
        self.assertEqual(self.tracker.total_tiles, self.total_tiles)
        self.assertEqual(self.tracker.confidence_threshold, self.confidence_threshold)
        self.assertEqual(self.tracker.target_processing_time, self.target_time)
        self.assertEqual(self.tracker.tiles_processed, 0)
        self.assertEqual(self.tracker.current_stage, "initializing")
        self.assertIsNone(self.tracker.start_time)
    
    def test_start_processing(self):
        """Test starting progress tracking."""
        self.tracker.start_processing()
        
        self.assertIsNotNone(self.tracker.start_time)
        self.assertEqual(self.tracker.current_stage, "streaming")
        self.assertIn("streaming", self.tracker.stage_start_times)
    
    def test_stage_transitions(self):
        """Test stage transitions and timing."""
        self.tracker.start_processing()
        initial_time = time.time()
        
        # Simulate some processing time
        time.sleep(0.1)
        
        # Transition to processing stage
        self.tracker.start_stage("processing")
        self.assertEqual(self.tracker.current_stage, "processing")
        self.assertIn("processing", self.tracker.stage_start_times)
        
        # Check that streaming stage duration was recorded
        self.assertIn("streaming", self.tracker.stage_durations)
        streaming_duration = self.tracker.stage_durations["streaming"]
        self.assertGreater(streaming_duration, 0.05)  # At least 50ms
    
    def test_tile_processing_tracking(self):
        """Test tile processing tracking."""
        self.tracker.start_processing()
        
        # Record successful tile processing
        self.tracker.record_tile_processed(
            processing_time=0.1,
            tile_size=1024,
            success=True,
            skipped=False
        )
        
        self.assertEqual(self.tracker.tiles_processed, 1)
        self.assertEqual(self.tracker.tiles_failed, 0)
        self.assertEqual(self.tracker.tiles_skipped, 0)
        self.assertEqual(len(self.tracker.processing_times), 1)
        self.assertEqual(len(self.tracker.throughput_history), 1)
        
        # Record failed tile
        self.tracker.record_tile_processed(
            processing_time=0.05,
            tile_size=1024,
            success=False,
            skipped=False
        )
        
        self.assertEqual(self.tracker.tiles_processed, 1)
        self.assertEqual(self.tracker.tiles_failed, 1)
        
        # Record skipped tile
        self.tracker.record_tile_processed(
            processing_time=0.02,
            tile_size=1024,
            success=True,
            skipped=True
        )
        
        self.assertEqual(self.tracker.tiles_processed, 1)
        self.assertEqual(self.tracker.tiles_skipped, 1)
    
    def test_confidence_tracking(self):
        """Test confidence tracking and early stopping."""
        self.tracker.start_processing()
        
        # Update confidence below threshold
        self.tracker.update_confidence(0.8)
        self.assertEqual(self.tracker.current_confidence, 0.8)
        self.assertFalse(self.tracker.early_stop_recommended)
        
        # Update confidence above threshold
        self.tracker.update_confidence(0.96)
        self.assertEqual(self.tracker.current_confidence, 0.96)
        self.assertTrue(self.tracker.early_stop_recommended)
        
        # Check confidence delta calculation
        self.assertAlmostEqual(self.tracker.confidence_delta, 0.16, places=2)
    
    def test_eta_calculation(self):
        """Test ETA calculation accuracy."""
        self.tracker.start_processing()
        
        # Process some tiles with known timing
        for i in range(10):
            self.tracker.record_tile_processed(
                processing_time=0.1,  # 100ms per tile
                tile_size=1024,
                success=True,
                skipped=False
            )
        
        progress = self.tracker.get_current_progress()
        
        # With 10 tiles processed at 100ms each, remaining 90 tiles should take ~9 seconds
        expected_eta = 90 * 0.1  # 9.0 seconds
        self.assertAlmostEqual(progress.estimated_time_remaining, expected_eta, delta=1.0)
        
        # Check progress ratio
        expected_progress = 10 / 100  # 10%
        self.assertAlmostEqual(progress.progress_ratio, expected_progress, places=2)
    
    def test_progress_callbacks(self):
        """Test progress callback functionality."""
        callback_mock = Mock()
        callback_config = ProgressCallback(
            callback_func=callback_mock,
            update_interval=0.1,
            min_progress_delta=0.05
        )
        
        tracker = StreamingProgressTracker(
            total_tiles=20,
            progress_callbacks=[callback_config]
        )
        
        tracker.start_processing()
        
        # Process tiles to trigger callbacks
        for i in range(5):
            tracker.record_tile_processed(0.1, 1024, True, False)
            time.sleep(0.05)  # Small delay
        
        # Get progress to trigger callback check
        progress = tracker.get_current_progress()
        
        # Callback should have been called at least once
        self.assertGreater(callback_mock.call_count, 0)
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        self.tracker.start_processing()
        
        # Process some tiles
        for i in range(5):
            self.tracker.record_tile_processed(0.1, 1024, True, False)
        
        progress = self.tracker.get_current_progress()
        
        # Memory usage should be tracked
        self.assertGreaterEqual(progress.memory_usage_gb, 0.0)
        self.assertGreaterEqual(progress.peak_memory_usage_gb, 0.0)
    
    def test_data_quality_calculation(self):
        """Test data quality score calculation."""
        self.tracker.start_processing()
        
        # Process mix of successful, failed, and skipped tiles
        self.tracker.record_tile_processed(0.1, 1024, True, False)   # Success
        self.tracker.record_tile_processed(0.1, 1024, True, False)   # Success
        self.tracker.record_tile_processed(0.1, 1024, False, False)  # Failed
        self.tracker.record_tile_processed(0.1, 1024, True, True)    # Skipped
        
        progress = self.tracker.get_current_progress()
        
        # Quality score should be 2 successful / 4 total = 0.5
        expected_quality = 2.0 / 4.0
        self.assertAlmostEqual(progress.data_quality_score, expected_quality, places=2)


class TestWSIStreamReaderProgressIntegration(unittest.TestCase):
    """Test cases for WSIStreamReader progress tracking integration."""
    
    def setUp(self):
        """Set up test fixtures with mocked dependencies."""
        self.mock_wsi_path = "test_slide.svs"
        
        # Mock WSIReader
        self.mock_wsi_reader = Mock()
        self.mock_wsi_reader.dimensions = (10000, 10000)
        self.mock_wsi_reader.level_count = 3
        self.mock_wsi_reader.level_dimensions = [(10000, 10000), (5000, 5000), (2500, 2500)]
        self.mock_wsi_reader.get_magnification.return_value = 40.0
        self.mock_wsi_reader.get_mpp.return_value = (0.25, 0.25)
        
        # Configure tile buffer
        self.config = TileBufferConfig(
            max_memory_gb=1.0,
            tile_size=1024,
            adaptive_sizing_enabled=True
        )
    
    @patch('src.data.wsi_pipeline.wsi_stream_reader.WSIReader')
    def test_progress_tracking_initialization(self, mock_wsi_reader_class):
        """Test progress tracking initialization in WSIStreamReader."""
        mock_wsi_reader_class.return_value = self.mock_wsi_reader
        
        # Create reader with progress callbacks
        callback_mock = Mock()
        progress_callbacks = [ProgressCallback(callback_func=callback_mock)]
        
        reader = WSIStreamReader(
            self.mock_wsi_path,
            config=self.config,
            progress_callbacks=progress_callbacks
        )
        
        # Initialize streaming
        metadata = reader.initialize_streaming(
            target_processing_time=30.0,
            confidence_threshold=0.95
        )
        
        # Check that progress tracker was created
        self.assertIsNotNone(reader._progress_tracker)
        self.assertEqual(reader._progress_tracker.total_tiles, metadata.estimated_patches)
        self.assertEqual(reader._progress_tracker.confidence_threshold, 0.95)
        self.assertEqual(reader._progress_tracker.target_processing_time, 30.0)
    
    @patch('src.data.wsi_pipeline.wsi_stream_reader.WSIReader')
    def test_progress_updates_during_streaming(self, mock_wsi_reader_class):
        """Test progress updates during tile streaming."""
        mock_wsi_reader_class.return_value = self.mock_wsi_reader
        
        reader = WSIStreamReader(self.mock_wsi_path, config=self.config)
        
        # Mock tile reading
        mock_tile_data = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        reader._read_tile = Mock(return_value=mock_tile_data)
        
        # Initialize streaming
        reader.initialize_streaming()
        
        # Get initial progress
        initial_progress = reader.get_progress()
        self.assertEqual(initial_progress.tiles_processed, 0)
        self.assertEqual(initial_progress.progress_ratio, 0.0)
        
        # Process a few batches
        batch_count = 0
        for batch in reader.stream_tiles(batch_size=4):
            batch_count += 1
            if batch_count >= 3:  # Process 3 batches
                break
        
        # Check progress after processing
        final_progress = reader.get_progress()
        self.assertGreater(final_progress.tiles_processed, 0)
        self.assertGreater(final_progress.progress_ratio, 0.0)
        self.assertGreater(final_progress.elapsed_time, 0.0)
    
    @patch('src.data.wsi_pipeline.wsi_stream_reader.WSIReader')
    def test_confidence_based_early_stopping(self, mock_wsi_reader_class):
        """Test early stopping based on confidence threshold."""
        mock_wsi_reader_class.return_value = self.mock_wsi_reader
        
        reader = WSIStreamReader(self.mock_wsi_path, config=self.config)
        
        # Mock tile reading
        mock_tile_data = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        reader._read_tile = Mock(return_value=mock_tile_data)
        
        # Initialize streaming with low confidence threshold for testing
        reader.initialize_streaming(confidence_threshold=0.8)
        
        # Simulate high confidence after a few tiles
        batch_count = 0
        for batch in reader.stream_tiles(batch_size=2):
            batch_count += 1
            
            # Update confidence to trigger early stopping
            if batch_count == 2:
                reader.update_confidence(0.85)  # Above threshold
            
            # Check if early stopping is recommended
            progress = reader.get_progress()
            if progress.early_stop_recommended:
                break
            
            if batch_count >= 10:  # Safety limit
                break
        
        # Verify early stopping was triggered
        final_progress = reader.get_progress()
        self.assertTrue(final_progress.early_stop_recommended)
        self.assertGreaterEqual(final_progress.current_confidence, 0.8)
    
    @patch('src.data.wsi_pipeline.wsi_stream_reader.WSIReader')
    def test_progress_callback_integration(self, mock_wsi_reader_class):
        """Test progress callback integration."""
        mock_wsi_reader_class.return_value = self.mock_wsi_reader
        
        # Create callback mock
        callback_mock = Mock()
        
        reader = WSIStreamReader(self.mock_wsi_path, config=self.config)
        
        # Add progress callback
        reader.add_progress_callback(
            callback_func=callback_mock,
            update_interval=0.1,
            min_progress_delta=0.01
        )
        
        # Mock tile reading
        mock_tile_data = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        reader._read_tile = Mock(return_value=mock_tile_data)
        
        # Initialize and process
        reader.initialize_streaming()
        
        batch_count = 0
        for batch in reader.stream_tiles(batch_size=2):
            batch_count += 1
            if batch_count >= 3:
                break
            time.sleep(0.05)  # Small delay to allow callbacks
        
        # Verify callback was called
        self.assertGreater(callback_mock.call_count, 0)
        
        # Verify callback received StreamingProgress objects
        for call in callback_mock.call_args_list:
            args, kwargs = call
            self.assertIsInstance(args[0], StreamingProgress)
    
    @patch('src.data.wsi_pipeline.wsi_stream_reader.WSIReader')
    def test_detailed_progress_stats(self, mock_wsi_reader_class):
        """Test detailed progress statistics."""
        mock_wsi_reader_class.return_value = self.mock_wsi_reader
        
        reader = WSIStreamReader(self.mock_wsi_path, config=self.config)
        reader.initialize_streaming()
        
        # Get detailed stats
        stats = reader.get_detailed_progress_stats()
        
        # Verify required fields are present
        required_fields = [
            'tiles_processed', 'total_tiles', 'progress_ratio',
            'elapsed_time', 'estimated_time_remaining',
            'throughput_tiles_per_second', 'current_confidence',
            'slide_path', 'slide_id', 'level'
        ]
        
        for field in required_fields:
            self.assertIn(field, stats)
    
    @patch('src.data.wsi_pipeline.wsi_stream_reader.WSIReader')
    def test_performance_summary(self, mock_wsi_reader_class):
        """Test performance summary generation."""
        mock_wsi_reader_class.return_value = self.mock_wsi_reader
        
        reader = WSIStreamReader(self.mock_wsi_path, config=self.config)
        reader.initialize_streaming()
        
        # Get performance summary
        summary = reader.get_performance_summary()
        
        # Verify required performance metrics
        required_metrics = [
            'total_processing_time', 'tiles_per_second',
            'memory_efficiency', 'data_quality_score',
            'processing_stages', 'resource_usage'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, summary)
        
        # Verify nested structures
        self.assertIn('loading_time', summary['processing_stages'])
        self.assertIn('peak_memory_gb', summary['resource_usage'])


class TestProgressTrackingEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for progress tracking."""
    
    def test_progress_without_initialization(self):
        """Test getting progress before initialization."""
        config = TileBufferConfig(max_memory_gb=1.0)
        
        with patch('src.data.wsi_pipeline.wsi_stream_reader.WSIReader'):
            reader = WSIStreamReader("test.svs", config=config)
            
            # Get progress before initialization
            progress = reader.get_progress()
            
            # Should return default/empty progress
            self.assertEqual(progress.tiles_processed, 0)
            self.assertEqual(progress.total_tiles, 0)
            self.assertEqual(progress.progress_ratio, 0.0)
    
    def test_confidence_tracking_without_tracker(self):
        """Test confidence updates without progress tracker."""
        config = TileBufferConfig(max_memory_gb=1.0)
        
        with patch('src.data.wsi_pipeline.wsi_stream_reader.WSIReader'):
            reader = WSIStreamReader("test.svs", config=config)
            
            # Should not raise error even without tracker
            reader.update_confidence(0.8)
    
    def test_eta_with_zero_processing_time(self):
        """Test ETA calculation with zero processing time."""
        tracker = StreamingProgressTracker(total_tiles=100)
        tracker.start_processing()
        
        # Record tile with zero processing time
        tracker.record_tile_processed(0.0, 1024, True, False)
        
        progress = tracker.get_current_progress()
        
        # Should handle zero processing time gracefully
        self.assertGreaterEqual(progress.estimated_time_remaining, 0.0)
    
    def test_memory_tracking_failure(self):
        """Test graceful handling of memory tracking failures."""
        with patch('psutil.Process') as mock_process:
            mock_process.side_effect = Exception("Memory access failed")
            
            tracker = StreamingProgressTracker(total_tiles=10)
            tracker.start_processing()
            
            # Should not raise error even if memory tracking fails
            progress = tracker.get_current_progress()
            self.assertGreaterEqual(progress.memory_usage_gb, 0.0)


if __name__ == '__main__':
    unittest.main()