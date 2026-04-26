"""Unit tests for ProgressiveVisualizer."""

import pytest
import numpy as np
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.streaming.progressive_visualizer import (
    ProgressiveVisualizer,
    VisualizationUpdate
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def visualizer(temp_output_dir):
    """Create ProgressiveVisualizer instance."""
    return ProgressiveVisualizer(
        output_dir=temp_output_dir,
        slide_dimensions=(10240, 10240),
        tile_size=1024,
        update_interval=0.1
    )


class TestVisualizationUpdate:
    """Test VisualizationUpdate dataclass."""
    
    def test_initialization(self):
        """Test VisualizationUpdate creation."""
        update = VisualizationUpdate(
            timestamp=time.time(),
            patches_processed=100,
            confidence=0.85
        )
        
        assert update.patches_processed == 100
        assert update.confidence == 0.85
        assert update.attention_weights is None
        assert update.coordinates is None
    
    def test_with_optional_fields(self):
        """Test VisualizationUpdate with optional fields."""
        weights = np.array([0.1, 0.2, 0.3])
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        
        update = VisualizationUpdate(
            timestamp=time.time(),
            patches_processed=3,
            confidence=0.75,
            attention_weights=weights,
            coordinates=coords
        )
        
        assert update.attention_weights is not None
        assert update.coordinates is not None
        np.testing.assert_array_equal(update.attention_weights, weights)
        np.testing.assert_array_equal(update.coordinates, coords)


class TestProgressiveVisualizer:
    """Test ProgressiveVisualizer class."""
    
    def test_initialization(self, visualizer, temp_output_dir):
        """Test visualizer initialization."""
        assert visualizer.slide_dimensions == (10240, 10240)
        assert visualizer.tile_size == 1024
        assert visualizer.heatmap_width == 10
        assert visualizer.heatmap_height == 10
        assert visualizer.attention_heatmap.shape == (10, 10)
        assert visualizer.coverage_mask.shape == (10, 10)
        assert Path(temp_output_dir).exists()
    
    def test_heatmap_dimensions_calculation(self, temp_output_dir):
        """Test heatmap dimension calculation with various sizes."""
        # Exact multiple
        viz1 = ProgressiveVisualizer(temp_output_dir, (2048, 2048), tile_size=512)
        assert viz1.heatmap_width == 4
        assert viz1.heatmap_height == 4
        
        # Non-exact multiple (should round up)
        viz2 = ProgressiveVisualizer(temp_output_dir, (2100, 2100), tile_size=512)
        assert viz2.heatmap_width == 5
        assert viz2.heatmap_height == 5
    
    def test_colormap_creation(self, visualizer):
        """Test custom colormap creation."""
        cmap = visualizer.colormap
        assert cmap is not None
        assert cmap.N == 256
        
        # Test colormap values
        colors = cmap(np.linspace(0, 1, 5))
        assert colors.shape == (5, 4)  # RGBA
    
    def test_update_attention_heatmap(self, visualizer):
        """Test attention heatmap update."""
        weights = np.array([0.5, 0.8, 0.3])
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        
        visualizer.update_attention_heatmap(
            attention_weights=weights,
            coordinates=coords,
            confidence=0.85,
            patches_processed=3
        )
        
        # Check heatmap updated
        assert visualizer.attention_heatmap[0, 0] == 0.5
        assert visualizer.attention_heatmap[1, 1] == 0.8
        assert visualizer.attention_heatmap[2, 2] == 0.3
        
        # Check coverage mask
        assert visualizer.coverage_mask[0, 0]
        assert visualizer.coverage_mask[1, 1]
        assert visualizer.coverage_mask[2, 2]
        
        # Check confidence history
        assert len(visualizer.confidence_history) == 1
        assert visualizer.confidence_history[0][1] == 0.85
    
    def test_update_attention_heatmap_accumulation(self, visualizer):
        """Test attention weight accumulation."""
        # First update
        weights1 = np.array([0.5])
        coords1 = np.array([[0, 0]])
        visualizer.update_attention_heatmap(weights1, coords1, 0.7, 1)
        
        # Second update to same location
        weights2 = np.array([0.3])
        coords2 = np.array([[0, 0]])
        visualizer.update_attention_heatmap(weights2, coords2, 0.8, 2)
        
        # Should accumulate
        assert visualizer.attention_heatmap[0, 0] == 0.8  # 0.5 + 0.3
    
    def test_update_attention_heatmap_validation(self, visualizer):
        """Test input validation."""
        # Mismatched sizes
        weights = np.array([0.5, 0.8])
        coords = np.array([[0, 0]])  # Only 1 coord for 2 weights
        
        with pytest.raises(ValueError, match="same batch size"):
            visualizer.update_attention_heatmap(weights, coords, 0.85, 2)
    
    def test_update_attention_heatmap_out_of_bounds(self, visualizer):
        """Test handling of out-of-bounds coordinates."""
        weights = np.array([0.5, 0.8, 0.3])
        coords = np.array([[0, 0], [100, 100], [2, 2]])  # [100, 100] out of bounds
        
        # Should not raise, just skip out-of-bounds
        visualizer.update_attention_heatmap(weights, coords, 0.85, 3)
        
        # Check only valid coords updated
        assert visualizer.attention_heatmap[0, 0] == 0.5
        assert visualizer.attention_heatmap[2, 2] == 0.3
        # Don't check out-of-bounds index
    
    def test_confidence_tracking(self, visualizer):
        """Test confidence history tracking."""
        for i in range(5):
            weights = np.array([0.5])
            coords = np.array([[i, i]])
            confidence = 0.5 + i * 0.1
            visualizer.update_attention_heatmap(weights, coords, confidence, i+1)
        
        assert len(visualizer.confidence_history) == 5
        
        # Check confidence values
        confidences = [c for _, c in visualizer.confidence_history]
        expected = [0.5, 0.6, 0.7, 0.8, 0.9]
        np.testing.assert_allclose(confidences, expected, rtol=1e-5)
    
    @patch('matplotlib.pyplot.savefig')
    def test_save_attention_heatmap(self, mock_savefig, visualizer):
        """Test saving attention heatmap."""
        # Add some data
        weights = np.array([0.5, 0.8])
        coords = np.array([[0, 0], [1, 1]])
        visualizer.update_attention_heatmap(weights, coords, 0.85, 2)
        
        # Save heatmap
        visualizer._save_attention_heatmap(patches_processed=2)
        
        # Check savefig called
        assert mock_savefig.called
        call_args = mock_savefig.call_args
        assert 'attention_heatmap_realtime.png' in str(call_args[0][0])
    
    @patch('matplotlib.pyplot.savefig')
    def test_save_confidence_plot(self, mock_savefig, visualizer):
        """Test saving confidence plot."""
        # Add confidence history
        for i in range(3):
            weights = np.array([0.5])
            coords = np.array([[i, i]])
            visualizer.update_attention_heatmap(weights, coords, 0.7 + i*0.1, i+1)
        
        # Save plot
        visualizer._save_confidence_plot()
        
        # Check savefig called
        assert mock_savefig.called
        call_args = mock_savefig.call_args
        assert 'confidence_progression.png' in str(call_args[0][0])
    
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.savefig')
    def test_save_confidence_plot_insufficient_data(self, mock_savefig, mock_close, visualizer):
        """Test confidence plot with insufficient data."""
        mock_savefig.reset_mock()  # Reset from previous tests
        
        # Only 1 data point - call _save_confidence_plot directly
        visualizer.confidence_history = [(time.time(), 0.85)]
        
        # Should not save with < 2 points
        visualizer._save_confidence_plot()
        assert not mock_savefig.called
    
    def test_get_statistics(self, visualizer):
        """Test statistics retrieval."""
        # Add some data
        weights = np.array([0.5, 0.8, 0.3])
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        visualizer.update_attention_heatmap(weights, coords, 0.85, 3)
        
        stats = visualizer.get_statistics()
        
        assert stats['heatmap_dimensions'] == (10, 10)
        assert stats['coverage_percent'] == 3.0  # 3 out of 100 tiles
        assert stats['total_updates'] == 1
        assert stats['current_confidence'] == 0.85
        assert 'output_directory' in stats
    
    def test_get_statistics_empty(self, visualizer):
        """Test statistics with no data."""
        stats = visualizer.get_statistics()
        
        assert stats['heatmap_dimensions'] == (10, 10)
        assert stats['coverage_percent'] == 0.0
        assert stats['total_updates'] == 0
        assert stats['current_confidence'] == 0.0
    
    @patch('matplotlib.pyplot.savefig')
    def test_save_final_visualizations(self, mock_savefig, visualizer):
        """Test final visualization export."""
        # Add data
        weights = np.array([0.5, 0.8])
        coords = np.array([[0, 0], [1, 1]])
        visualizer.update_attention_heatmap(weights, coords, 0.85, 2)
        
        # Save final visualizations
        visualizer.save_final_visualizations(export_formats=['png'])
        
        # Should save multiple files
        assert mock_savefig.call_count >= 3  # heatmap, confidence, dashboard
    
    @patch('matplotlib.pyplot.savefig')
    def test_save_final_visualizations_multiple_formats(self, mock_savefig, visualizer):
        """Test final visualization export in multiple formats."""
        # Add data
        weights = np.array([0.5])
        coords = np.array([[0, 0]])
        visualizer.update_attention_heatmap(weights, coords, 0.85, 1)
        
        # Save in multiple formats
        visualizer.save_final_visualizations(export_formats=['png', 'pdf'])
        
        # Check both formats saved
        call_args_list = [str(call[0][0]) for call in mock_savefig.call_args_list]
        assert any('.png' in arg for arg in call_args_list)
        assert any('.pdf' in arg for arg in call_args_list)
    
    def test_async_updates_start_stop(self, visualizer):
        """Test async update thread lifecycle."""
        # Start
        visualizer.start_async_updates()
        assert visualizer.running
        assert visualizer.visualization_thread is not None
        assert visualizer.visualization_thread.is_alive()
        
        # Stop
        visualizer.stop_async_updates()
        assert not visualizer.running
        
        # Wait a bit for thread to finish
        time.sleep(0.2)
        assert not visualizer.visualization_thread.is_alive()
    
    def test_async_updates_double_start(self, visualizer):
        """Test starting async updates twice."""
        visualizer.start_async_updates()
        
        # Second start should warn but not crash
        visualizer.start_async_updates()
        
        visualizer.stop_async_updates()
    
    @patch.object(ProgressiveVisualizer, '_process_update')
    def test_async_update_processing(self, mock_process, visualizer):
        """Test async update queue processing."""
        visualizer.start_async_updates()
        
        # Add update
        weights = np.array([0.5])
        coords = np.array([[0, 0]])
        visualizer.update_attention_heatmap(weights, coords, 0.85, 1)
        
        # Wait for processing
        time.sleep(0.3)
        
        # Check update processed
        assert mock_process.called
        
        visualizer.stop_async_updates()
    
    def test_update_interval_throttling(self, visualizer):
        """Test update interval throttling."""
        visualizer.update_interval = 1.0  # 1 second
        
        # First update should process
        weights = np.array([0.5])
        coords = np.array([[0, 0]])
        visualizer.update_attention_heatmap(weights, coords, 0.85, 1)
        first_time = visualizer.last_update_time
        
        # Immediate second update should be throttled
        time.sleep(0.1)
        visualizer.update_attention_heatmap(weights, coords, 0.90, 2)
        
        # last_update_time should not change (throttled)
        assert visualizer.last_update_time == first_time
    
    def test_context_manager(self, visualizer):
        """Test context manager protocol."""
        with visualizer as viz:
            assert viz.running
            
            # Add data
            weights = np.array([0.5])
            coords = np.array([[0, 0]])
            viz.update_attention_heatmap(weights, coords, 0.85, 1)
        
        # Should stop and save on exit
        assert not visualizer.running
    
    @patch('matplotlib.pyplot.savefig')
    def test_context_manager_saves_final(self, mock_savefig, visualizer):
        """Test context manager saves final visualizations."""
        with visualizer as viz:
            weights = np.array([0.5])
            coords = np.array([[0, 0]])
            viz.update_attention_heatmap(weights, coords, 0.85, 1)
        
        # Should have saved final visualizations
        assert mock_savefig.called


class TestIntegration:
    """Integration tests for ProgressiveVisualizer."""
    
    @patch('matplotlib.pyplot.savefig')
    def test_end_to_end_visualization(self, mock_savefig, temp_output_dir):
        """Test complete visualization workflow."""
        viz = ProgressiveVisualizer(
            output_dir=temp_output_dir,
            slide_dimensions=(5120, 5120),
            tile_size=512,
            update_interval=0.05
        )
        
        # Simulate streaming updates
        for i in range(10):
            batch_size = 5
            weights = np.random.rand(batch_size)
            coords = np.random.randint(0, 10, size=(batch_size, 2))
            confidence = 0.5 + i * 0.05
            
            viz.update_attention_heatmap(weights, coords, confidence, (i+1)*batch_size)
            time.sleep(0.01)
        
        # Save final
        viz.save_final_visualizations(export_formats=['png'])
        
        # Check statistics
        stats = viz.get_statistics()
        assert stats['total_updates'] == 10
        assert stats['coverage_percent'] > 0
        assert stats['current_confidence'] > 0.5
        
        # Check files saved
        assert mock_savefig.called
    
    @patch('matplotlib.pyplot.savefig')
    def test_async_visualization_workflow(self, mock_savefig, temp_output_dir):
        """Test async visualization with context manager."""
        with ProgressiveVisualizer(
            output_dir=temp_output_dir,
            slide_dimensions=(2048, 2048),
            tile_size=256,
            update_interval=0.05
        ) as viz:
            # Simulate rapid updates
            for i in range(20):
                weights = np.random.rand(3)
                coords = np.random.randint(0, 8, size=(3, 2))
                confidence = 0.6 + i * 0.02
                
                viz.update_attention_heatmap(weights, coords, confidence, (i+1)*3)
                time.sleep(0.01)
            
            # Wait for async processing
            time.sleep(0.2)
        
        # Should have processed updates and saved final
        assert mock_savefig.called
