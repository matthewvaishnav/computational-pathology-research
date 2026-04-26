"""Unit tests for ProgressiveVisualizer."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

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
    slide_dimensions = (10000, 8000)
    tile_size = 1024
    return ProgressiveVisualizer(
        output_dir=temp_output_dir,
        slide_dimensions=slide_dimensions,
        tile_size=tile_size,
        update_interval=0.1
    )


def test_visualizer_initialization(visualizer, temp_output_dir):
    """Test visualizer initialization."""
    assert visualizer.output_dir == Path(temp_output_dir)
    assert visualizer.slide_dimensions == (10000, 8000)
    assert visualizer.tile_size == 1024
    assert visualizer.update_interval == 0.1
    
    # Check heatmap dimensions
    expected_width = (10000 + 1024 - 1) // 1024
    expected_height = (8000 + 1024 - 1) // 1024
    assert visualizer.heatmap_width == expected_width
    assert visualizer.heatmap_height == expected_height
    
    # Check initialization
    assert visualizer.attention_heatmap.shape == (expected_height, expected_width)
    assert visualizer.coverage_mask.shape == (expected_height, expected_width)
    assert len(visualizer.confidence_history) == 0


def test_update_attention_heatmap(visualizer):
    """Test attention heatmap update."""
    # Create test data
    batch_size = 4
    attention_weights = np.array([0.1, 0.3, 0.5, 0.2])
    coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    confidence = 0.85
    patches_processed = 100
    
    # Update heatmap
    visualizer.update_attention_heatmap(
        attention_weights, coordinates, confidence, patches_processed
    )
    
    # Check heatmap updated
    assert visualizer.attention_heatmap[0, 0] == 0.1
    assert visualizer.attention_heatmap[0, 1] == 0.3
    assert visualizer.attention_heatmap[1, 0] == 0.5
    assert visualizer.attention_heatmap[1, 1] == 0.2
    
    # Check coverage mask
    assert visualizer.coverage_mask[0, 0]
    assert visualizer.coverage_mask[0, 1]
    assert visualizer.coverage_mask[1, 0]
    assert visualizer.coverage_mask[1, 1]
    
    # Check confidence history
    assert len(visualizer.confidence_history) == 1
    assert visualizer.confidence_history[0][1] == 0.85


def test_update_attention_heatmap_accumulation(visualizer):
    """Test attention weight accumulation."""
    # First update
    attention_weights = np.array([0.1])
    coordinates = np.array([[0, 0]])
    visualizer.update_attention_heatmap(attention_weights, coordinates, 0.8, 10)
    
    # Second update (same location)
    attention_weights = np.array([0.2])
    coordinates = np.array([[0, 0]])
    visualizer.update_attention_heatmap(attention_weights, coordinates, 0.9, 20)
    
    # Check accumulation
    assert visualizer.attention_heatmap[0, 0] == 0.3  # 0.1 + 0.2


def test_update_attention_heatmap_validation(visualizer):
    """Test input validation."""
    # Mismatched sizes
    attention_weights = np.array([0.1, 0.2])
    coordinates = np.array([[0, 0]])  # Only 1 coordinate
    
    with pytest.raises(ValueError, match="same batch size"):
        visualizer.update_attention_heatmap(attention_weights, coordinates, 0.8, 10)


def test_update_attention_heatmap_bounds(visualizer):
    """Test coordinate bounds checking."""
    # Out of bounds coordinates (should be ignored)
    attention_weights = np.array([0.1, 0.2, 0.3])
    coordinates = np.array([
        [0, 0],  # Valid
        [1000, 1000],  # Out of bounds
        [1, 1]  # Valid
    ])
    
    visualizer.update_attention_heatmap(attention_weights, coordinates, 0.8, 10)
    
    # Check only valid coordinates updated
    assert visualizer.attention_heatmap[0, 0] == 0.1
    assert visualizer.attention_heatmap[1, 1] == 0.3
    assert visualizer.coverage_mask[0, 0]
    assert visualizer.coverage_mask[1, 1]


def test_get_statistics(visualizer):
    """Test statistics retrieval."""
    # Add some data
    attention_weights = np.array([0.1, 0.2])
    coordinates = np.array([[0, 0], [1, 1]])
    visualizer.update_attention_heatmap(attention_weights, coordinates, 0.85, 10)
    
    stats = visualizer.get_statistics()
    
    assert 'heatmap_dimensions' in stats
    assert 'coverage_percent' in stats
    assert 'total_updates' in stats
    assert 'current_confidence' in stats
    assert 'output_directory' in stats
    
    assert stats['total_updates'] == 1
    assert stats['current_confidence'] == 0.85
    assert stats['coverage_percent'] > 0


def test_save_final_visualizations(visualizer, temp_output_dir):
    """Test final visualization export."""
    # Add some data
    attention_weights = np.random.rand(10)
    coordinates = np.array([[i % 5, i // 5] for i in range(10)])
    
    for i in range(5):
        visualizer.update_attention_heatmap(
            attention_weights[i*2:(i+1)*2],
            coordinates[i*2:(i+1)*2],
            0.8 + i * 0.02,
            (i+1) * 10
        )
    
    # Save visualizations
    visualizer.save_final_visualizations(export_formats=['png'])
    
    # Check files created
    output_dir = Path(temp_output_dir)
    assert (output_dir / 'attention_heatmap_final.png').exists()
    assert (output_dir / 'confidence_progression_final.png').exists()
    assert (output_dir / 'processing_dashboard.png').exists()


def test_save_final_visualizations_multiple_formats(visualizer, temp_output_dir):
    """Test export in multiple formats."""
    # Add minimal data
    attention_weights = np.array([0.5])
    coordinates = np.array([[0, 0]])
    visualizer.update_attention_heatmap(attention_weights, coordinates, 0.9, 10)
    visualizer.update_attention_heatmap(attention_weights, coordinates, 0.95, 20)
    
    # Save in multiple formats
    visualizer.save_final_visualizations(export_formats=['png', 'pdf', 'svg'])
    
    # Check all formats created
    output_dir = Path(temp_output_dir)
    for fmt in ['png', 'pdf', 'svg']:
        assert (output_dir / f'attention_heatmap_final.{fmt}').exists()
        assert (output_dir / f'confidence_progression_final.{fmt}').exists()
        assert (output_dir / f'processing_dashboard.{fmt}').exists()


def test_context_manager(temp_output_dir):
    """Test context manager usage."""
    slide_dimensions = (5000, 4000)
    
    with ProgressiveVisualizer(temp_output_dir, slide_dimensions, tile_size=512) as viz:
        # Add data
        attention_weights = np.array([0.5])
        coordinates = np.array([[0, 0]])
        viz.update_attention_heatmap(attention_weights, coordinates, 0.9, 10)
        viz.update_attention_heatmap(attention_weights, coordinates, 0.95, 20)
    
    # Check final visualizations saved
    output_dir = Path(temp_output_dir)
    assert (output_dir / 'attention_heatmap_final.png').exists()


def test_async_updates(visualizer):
    """Test async update mechanism."""
    # Start async updates
    visualizer.start_async_updates()
    assert visualizer.running
    assert visualizer.visualization_thread is not None
    
    # Stop async updates
    visualizer.stop_async_updates()
    assert not visualizer.running


def test_colormap_creation(visualizer):
    """Test custom colormap creation."""
    cmap = visualizer.colormap
    assert cmap is not None
    assert cmap.N == 256  # 256 color bins


def test_empty_confidence_history(visualizer, temp_output_dir):
    """Test handling of empty confidence history."""
    # Try to save with no data
    visualizer.save_final_visualizations(export_formats=['png'])
    
    # Should create heatmap but not confidence plot
    output_dir = Path(temp_output_dir)
    assert (output_dir / 'attention_heatmap_final.png').exists()
    # Confidence plot may not exist with <2 data points


def test_visualization_update_dataclass():
    """Test VisualizationUpdate dataclass."""
    update = VisualizationUpdate(
        timestamp=1234.5,
        patches_processed=100,
        confidence=0.95,
        attention_weights=np.array([0.1, 0.2]),
        coordinates=np.array([[0, 0], [1, 1]])
    )
    
    assert update.timestamp == 1234.5
    assert update.patches_processed == 100
    assert update.confidence == 0.95
    assert update.attention_weights is not None
    assert update.coordinates is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
