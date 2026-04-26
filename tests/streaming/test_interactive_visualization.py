"""Unit tests for interactive visualization features in ProgressiveVisualizer."""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.streaming.progressive_visualizer import (
    ProgressiveVisualizer,
    InteractiveConfig,
    PLOTLY_AVAILABLE
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def interactive_config():
    """Create interactive configuration."""
    return InteractiveConfig(
        enable_zoom_pan=True,
        enable_overlay=True,
        enable_parameter_controls=True,
        initial_zoom_level=1.0,
        max_zoom_level=10.0,
        min_zoom_level=0.5
    )


@pytest.fixture
def visualizer_with_interactive(temp_output_dir, interactive_config):
    """Create ProgressiveVisualizer with interactive features."""
    return ProgressiveVisualizer(
        output_dir=temp_output_dir,
        slide_dimensions=(5120, 5120),
        tile_size=512,
        update_interval=0.1,
        interactive_config=interactive_config
    )


@pytest.fixture
def visualizer_with_data(visualizer_with_interactive):
    """Create visualizer with sample data."""
    # Add sample data
    for i in range(5):
        weights = np.random.rand(10) * 0.5 + 0.5
        coords = np.random.randint(0, 10, size=(10, 2))
        confidence = 0.6 + i * 0.08
        visualizer_with_interactive.update_attention_heatmap(
            weights, coords, confidence, (i+1)*10
        )
    
    return visualizer_with_interactive


class TestInteractiveConfig:
    """Test InteractiveConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = InteractiveConfig()
        
        assert config.enable_zoom_pan is True
        assert config.enable_overlay is True
        assert config.enable_parameter_controls is True
        assert config.initial_zoom_level == 1.0
        assert config.max_zoom_level == 10.0
        assert config.min_zoom_level == 0.5
        assert config.update_callback is None
    
    def test_custom_initialization(self):
        """Test custom configuration values."""
        callback = Mock()
        config = InteractiveConfig(
            enable_zoom_pan=False,
            enable_overlay=False,
            enable_parameter_controls=False,
            initial_zoom_level=2.0,
            max_zoom_level=20.0,
            min_zoom_level=0.1,
            update_callback=callback
        )
        
        assert config.enable_zoom_pan is False
        assert config.enable_overlay is False
        assert config.enable_parameter_controls is False
        assert config.initial_zoom_level == 2.0
        assert config.max_zoom_level == 20.0
        assert config.min_zoom_level == 0.1
        assert config.update_callback is callback


class TestVisualizerInteractiveInitialization:
    """Test ProgressiveVisualizer initialization with interactive features."""
    
    def test_initialization_with_interactive_config(self, temp_output_dir, interactive_config):
        """Test visualizer initialization with interactive config."""
        viz = ProgressiveVisualizer(
            output_dir=temp_output_dir,
            slide_dimensions=(2048, 2048),
            tile_size=256,
            interactive_config=interactive_config
        )
        
        assert viz.interactive_config == interactive_config
        assert viz.current_zoom_level == 1.0
        assert viz.current_pan_offset == (0, 0)
        assert viz.overlay_opacity == 0.6
        assert viz.parameter_values == {}
    
    def test_initialization_without_interactive_config(self, temp_output_dir):
        """Test visualizer initialization without interactive config."""
        viz = ProgressiveVisualizer(
            output_dir=temp_output_dir,
            slide_dimensions=(2048, 2048),
            tile_size=256
        )
        
        # Should create default config
        assert viz.interactive_config is not None
        assert isinstance(viz.interactive_config, InteractiveConfig)
        assert viz.interactive_config.enable_zoom_pan is True


class TestZoomAndPan:
    """Test zoom and pan capabilities."""
    
    def test_set_zoom_level(self, visualizer_with_interactive):
        """Test setting zoom level."""
        visualizer_with_interactive.set_zoom_level(2.5)
        assert visualizer_with_interactive.current_zoom_level == 2.5
        
        visualizer_with_interactive.set_zoom_level(5.0)
        assert visualizer_with_interactive.current_zoom_level == 5.0
    
    def test_set_zoom_level_clamping(self, visualizer_with_interactive):
        """Test zoom level clamping to valid range."""
        # Test max clamping
        visualizer_with_interactive.set_zoom_level(15.0)
        assert visualizer_with_interactive.current_zoom_level == 10.0  # max_zoom_level
        
        # Test min clamping
        visualizer_with_interactive.set_zoom_level(0.1)
        assert visualizer_with_interactive.current_zoom_level == 0.5  # min_zoom_level
    
    def test_set_zoom_level_disabled(self, temp_output_dir):
        """Test zoom level when feature is disabled."""
        config = InteractiveConfig(enable_zoom_pan=False)
        viz = ProgressiveVisualizer(
            temp_output_dir,
            (2048, 2048),
            interactive_config=config
        )
        
        initial_zoom = viz.current_zoom_level
        viz.set_zoom_level(3.0)
        
        # Should not change when disabled
        assert viz.current_zoom_level == initial_zoom
    
    def test_set_pan_offset(self, visualizer_with_interactive):
        """Test setting pan offset."""
        visualizer_with_interactive.set_pan_offset(5, 3)
        assert visualizer_with_interactive.current_pan_offset == (5, 3)
        
        visualizer_with_interactive.set_pan_offset(8, 7)
        assert visualizer_with_interactive.current_pan_offset == (8, 7)
    
    def test_set_pan_offset_clamping(self, visualizer_with_interactive):
        """Test pan offset clamping to valid range."""
        # Test max clamping (heatmap is 10x10)
        visualizer_with_interactive.set_pan_offset(20, 20)
        assert visualizer_with_interactive.current_pan_offset == (10, 10)
        
        # Test min clamping
        visualizer_with_interactive.set_pan_offset(-5, -5)
        assert visualizer_with_interactive.current_pan_offset == (0, 0)
    
    def test_set_pan_offset_disabled(self, temp_output_dir):
        """Test pan offset when feature is disabled."""
        config = InteractiveConfig(enable_zoom_pan=False)
        viz = ProgressiveVisualizer(
            temp_output_dir,
            (2048, 2048),
            interactive_config=config
        )
        
        initial_offset = viz.current_pan_offset
        viz.set_pan_offset(5, 5)
        
        # Should not change when disabled
        assert viz.current_pan_offset == initial_offset


class TestOverlay:
    """Test attention weight overlay features."""
    
    def test_set_overlay_opacity(self, visualizer_with_interactive):
        """Test setting overlay opacity."""
        visualizer_with_interactive.set_overlay_opacity(0.8)
        assert visualizer_with_interactive.overlay_opacity == 0.8
        
        visualizer_with_interactive.set_overlay_opacity(0.3)
        assert visualizer_with_interactive.overlay_opacity == 0.3
    
    def test_set_overlay_opacity_clamping(self, visualizer_with_interactive):
        """Test opacity clamping to valid range."""
        # Test max clamping
        visualizer_with_interactive.set_overlay_opacity(1.5)
        assert visualizer_with_interactive.overlay_opacity == 1.0
        
        # Test min clamping
        visualizer_with_interactive.set_overlay_opacity(-0.2)
        assert visualizer_with_interactive.overlay_opacity == 0.0
    
    def test_set_overlay_opacity_disabled(self, temp_output_dir):
        """Test overlay opacity when feature is disabled."""
        config = InteractiveConfig(enable_overlay=False)
        viz = ProgressiveVisualizer(
            temp_output_dir,
            (2048, 2048),
            interactive_config=config
        )
        
        initial_opacity = viz.overlay_opacity
        viz.set_overlay_opacity(0.9)
        
        # Should not change when disabled
        assert viz.overlay_opacity == initial_opacity


class TestParameterControls:
    """Test real-time parameter adjustment features."""
    
    def test_update_parameter(self, visualizer_with_interactive):
        """Test updating parameters."""
        visualizer_with_interactive.update_parameter('confidence_threshold', 0.95)
        assert visualizer_with_interactive.parameter_values['confidence_threshold'] == 0.95
        
        visualizer_with_interactive.update_parameter('batch_size', 64)
        assert visualizer_with_interactive.parameter_values['batch_size'] == 64
    
    def test_update_parameter_multiple(self, visualizer_with_interactive):
        """Test updating multiple parameters."""
        params = {
            'confidence_threshold': 0.95,
            'batch_size': 64,
            'learning_rate': 0.001,
            'temperature': 1.5
        }
        
        for name, value in params.items():
            visualizer_with_interactive.update_parameter(name, value)
        
        for name, value in params.items():
            assert visualizer_with_interactive.parameter_values[name] == value
    
    def test_update_parameter_overwrite(self, visualizer_with_interactive):
        """Test overwriting existing parameter."""
        visualizer_with_interactive.update_parameter('threshold', 0.8)
        assert visualizer_with_interactive.parameter_values['threshold'] == 0.8
        
        visualizer_with_interactive.update_parameter('threshold', 0.9)
        assert visualizer_with_interactive.parameter_values['threshold'] == 0.9
    
    def test_update_parameter_with_callback(self, temp_output_dir):
        """Test parameter update with callback."""
        callback = Mock()
        config = InteractiveConfig(
            enable_parameter_controls=True,
            update_callback=callback
        )
        viz = ProgressiveVisualizer(
            temp_output_dir,
            (2048, 2048),
            interactive_config=config
        )
        
        viz.update_parameter('test_param', 42)
        
        # Check callback was called
        assert callback.called
        call_args = callback.call_args[0][0]
        assert call_args['parameter'] == 'test_param'
        assert call_args['new_value'] == 42
        assert call_args['old_value'] is None
        assert 'test_param' in call_args['all_parameters']
    
    def test_update_parameter_callback_error_handling(self, temp_output_dir):
        """Test parameter update with failing callback."""
        def failing_callback(update_dict):
            raise ValueError("Callback error")
        
        config = InteractiveConfig(
            enable_parameter_controls=True,
            update_callback=failing_callback
        )
        viz = ProgressiveVisualizer(
            temp_output_dir,
            (2048, 2048),
            interactive_config=config
        )
        
        # Should not raise, just log error
        viz.update_parameter('test_param', 42)
        
        # Parameter should still be updated
        assert viz.parameter_values['test_param'] == 42
    
    def test_update_parameter_disabled(self, temp_output_dir):
        """Test parameter update when feature is disabled."""
        config = InteractiveConfig(enable_parameter_controls=False)
        viz = ProgressiveVisualizer(
            temp_output_dir,
            (2048, 2048),
            interactive_config=config
        )
        
        viz.update_parameter('test_param', 42)
        
        # Should not update when disabled
        assert 'test_param' not in viz.parameter_values
    
    def test_get_parameter(self, visualizer_with_interactive):
        """Test getting parameter values."""
        visualizer_with_interactive.update_parameter('threshold', 0.95)
        
        value = visualizer_with_interactive.get_parameter('threshold')
        assert value == 0.95
    
    def test_get_parameter_default(self, visualizer_with_interactive):
        """Test getting parameter with default value."""
        value = visualizer_with_interactive.get_parameter('nonexistent', default=0.5)
        assert value == 0.5
    
    def test_get_parameter_none_default(self, visualizer_with_interactive):
        """Test getting nonexistent parameter without default."""
        value = visualizer_with_interactive.get_parameter('nonexistent')
        assert value is None


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
class TestInteractiveVisualizations:
    """Test interactive visualization generation with Plotly."""
    
    def test_create_interactive_heatmap(self, visualizer_with_data):
        """Test creating interactive heatmap."""
        fig = visualizer_with_data.create_interactive_heatmap()
        
        assert fig is not None
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'Interactive Attention Heatmap'
    
    def test_create_interactive_heatmap_with_thumbnail(self, visualizer_with_data):
        """Test creating interactive heatmap with slide thumbnail."""
        # Create fake thumbnail
        thumbnail = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        
        fig = visualizer_with_data.create_interactive_heatmap(slide_thumbnail=thumbnail)
        
        assert fig is not None
        # Should have both thumbnail and heatmap traces
        assert len(fig.data) >= 2
    
    def test_create_interactive_heatmap_zoom_pan(self, visualizer_with_data):
        """Test interactive heatmap with zoom and pan."""
        visualizer_with_data.set_zoom_level(2.0)
        visualizer_with_data.set_pan_offset(2, 3)
        
        fig = visualizer_with_data.create_interactive_heatmap()
        
        assert fig is not None
        # Check that zoom/pan affects axis ranges
        assert fig.layout.xaxis.range is not None
        assert fig.layout.yaxis.range is not None
    
    def test_create_interactive_confidence_plot(self, visualizer_with_data):
        """Test creating interactive confidence plot."""
        fig = visualizer_with_data.create_interactive_confidence_plot()
        
        assert fig is not None
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'Real-Time Confidence Progression'
        # Check yaxis range is set correctly (tuple or list)
        assert fig.layout.yaxis.range[0] == 0.0
        assert fig.layout.yaxis.range[1] == 1.0
    
    def test_create_interactive_confidence_plot_insufficient_data(self, visualizer_with_interactive):
        """Test confidence plot with insufficient data."""
        # Only add 1 data point
        visualizer_with_interactive.update_attention_heatmap(
            np.array([0.5]), np.array([[0, 0]]), 0.7, 1
        )
        
        fig = visualizer_with_interactive.create_interactive_confidence_plot()
        
        # Should return None with < 2 points
        assert fig is None
    
    def test_create_interactive_dashboard(self, visualizer_with_data):
        """Test creating interactive dashboard."""
        fig = visualizer_with_data.create_interactive_dashboard()
        
        assert fig is not None
        # Should have multiple subplots
        assert len(fig.data) >= 4  # heatmap, confidence, coverage, histogram
        assert 'Dashboard' in fig.layout.title.text
    
    def test_create_interactive_dashboard_with_thumbnail(self, visualizer_with_data):
        """Test creating dashboard with slide thumbnail."""
        thumbnail = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        
        fig = visualizer_with_data.create_interactive_dashboard(slide_thumbnail=thumbnail)
        
        assert fig is not None
        assert len(fig.data) >= 4
    
    def test_create_interactive_dashboard_no_data(self, visualizer_with_interactive):
        """Test creating dashboard with no data."""
        fig = visualizer_with_interactive.create_interactive_dashboard()
        
        # Should return None with no data
        assert fig is None
    
    def test_save_interactive_html(self, visualizer_with_data):
        """Test saving interactive dashboard as HTML."""
        output_path = visualizer_with_data.save_interactive_html('test_dashboard.html')
        
        assert output_path is not None
        assert output_path.exists()
        assert output_path.suffix == '.html'
        
        # Check file has content
        assert output_path.stat().st_size > 0
    
    def test_save_interactive_html_custom_filename(self, visualizer_with_data):
        """Test saving with custom filename."""
        output_path = visualizer_with_data.save_interactive_html('custom_viz.html')
        
        assert output_path is not None
        assert output_path.name == 'custom_viz.html'
        assert output_path.exists()


class TestVisualizationState:
    """Test visualization state export and persistence."""
    
    def test_export_visualization_state(self, visualizer_with_data):
        """Test exporting visualization state."""
        # Set some interactive state
        visualizer_with_data.set_zoom_level(2.5)
        visualizer_with_data.set_pan_offset(3, 4)
        visualizer_with_data.set_overlay_opacity(0.7)
        visualizer_with_data.update_parameter('threshold', 0.95)
        
        state = visualizer_with_data.export_visualization_state()
        
        assert state['slide_dimensions'] == (5120, 5120)
        assert state['heatmap_dimensions'] == (10, 10)
        assert state['tile_size'] == 512
        assert state['zoom_level'] == 2.5
        assert state['pan_offset'] == (3, 4)
        assert state['overlay_opacity'] == 0.7
        assert state['parameters']['threshold'] == 0.95
        assert 'statistics' in state
        assert 'confidence_history' in state
        assert len(state['confidence_history']) > 0
    
    def test_export_visualization_state_empty(self, visualizer_with_interactive):
        """Test exporting state with no data."""
        state = visualizer_with_interactive.export_visualization_state()
        
        assert state['slide_dimensions'] == (5120, 5120)
        assert state['zoom_level'] == 1.0
        assert state['pan_offset'] == (0, 0)
        assert state['parameters'] == {}
        assert state['confidence_history'] == []
    
    def test_save_visualization_state(self, visualizer_with_data):
        """Test saving visualization state to JSON."""
        visualizer_with_data.update_parameter('test_param', 42)
        
        output_path = visualizer_with_data.save_visualization_state('state.json')
        
        assert output_path.exists()
        assert output_path.suffix == '.json'
        
        # Load and verify JSON
        with open(output_path, 'r') as f:
            loaded_state = json.load(f)
        
        assert loaded_state['slide_dimensions'] == [5120, 5120]
        assert loaded_state['parameters']['test_param'] == 42
        assert 'confidence_history' in loaded_state
    
    def test_save_visualization_state_custom_filename(self, visualizer_with_data):
        """Test saving state with custom filename."""
        output_path = visualizer_with_data.save_visualization_state('custom_state.json')
        
        assert output_path.name == 'custom_state.json'
        assert output_path.exists()


class TestInteractiveIntegration:
    """Integration tests for interactive features."""
    
    def test_complete_interactive_workflow(self, temp_output_dir, interactive_config):
        """Test complete interactive visualization workflow."""
        viz = ProgressiveVisualizer(
            temp_output_dir,
            (4096, 4096),
            tile_size=512,
            interactive_config=interactive_config
        )
        
        # Simulate streaming updates
        for i in range(10):
            weights = np.random.rand(5) * 0.5 + 0.5
            coords = np.random.randint(0, 8, size=(5, 2))
            confidence = 0.6 + i * 0.04
            viz.update_attention_heatmap(weights, coords, confidence, (i+1)*5)
        
        # Set interactive state
        viz.set_zoom_level(3.0)
        viz.set_pan_offset(2, 2)
        viz.set_overlay_opacity(0.8)
        viz.update_parameter('confidence_threshold', 0.95)
        viz.update_parameter('batch_size', 64)
        
        # Export state
        state = viz.export_visualization_state()
        assert state['zoom_level'] == 3.0
        assert state['pan_offset'] == (2, 2)
        assert state['overlay_opacity'] == 0.8
        assert len(state['parameters']) == 2
        
        # Save state
        state_path = viz.save_visualization_state()
        assert state_path.exists()
        
        # Create visualizations if Plotly available
        if PLOTLY_AVAILABLE:
            fig_heatmap = viz.create_interactive_heatmap()
            assert fig_heatmap is not None
            
            fig_confidence = viz.create_interactive_confidence_plot()
            assert fig_confidence is not None
            
            fig_dashboard = viz.create_interactive_dashboard()
            assert fig_dashboard is not None
            
            html_path = viz.save_interactive_html()
            assert html_path is not None
            assert html_path.exists()
    
    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_interactive_with_parameter_callback(self, temp_output_dir):
        """Test interactive features with parameter update callback."""
        callback_calls = []
        
        def track_callback(update_dict):
            callback_calls.append(update_dict.copy())
        
        config = InteractiveConfig(
            enable_parameter_controls=True,
            update_callback=track_callback
        )
        
        viz = ProgressiveVisualizer(
            temp_output_dir,
            (2048, 2048),
            interactive_config=config
        )
        
        # Add data
        weights = np.random.rand(10)
        coords = np.random.randint(0, 4, size=(10, 2))
        viz.update_attention_heatmap(weights, coords, 0.85, 10)
        
        # Update parameters
        viz.update_parameter('threshold', 0.95)
        viz.update_parameter('batch_size', 32)
        
        # Check callbacks were triggered
        assert len(callback_calls) == 2
        assert callback_calls[0]['parameter'] == 'threshold'
        assert callback_calls[0]['new_value'] == 0.95
        assert callback_calls[1]['parameter'] == 'batch_size'
        assert callback_calls[1]['new_value'] == 32
        
        # Create visualization
        fig = viz.create_interactive_dashboard()
        assert fig is not None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_interactive_features_without_plotly(self, visualizer_with_data, monkeypatch):
        """Test interactive features when Plotly is not available."""
        # Temporarily disable Plotly
        import src.streaming.progressive_visualizer as viz_module
        monkeypatch.setattr(viz_module, 'PLOTLY_AVAILABLE', False)
        
        # These should return None gracefully
        fig = visualizer_with_data.create_interactive_heatmap()
        assert fig is None
        
        fig = visualizer_with_data.create_interactive_confidence_plot()
        assert fig is None
        
        fig = visualizer_with_data.create_interactive_dashboard()
        assert fig is None
        
        path = visualizer_with_data.save_interactive_html()
        assert path is None
    
    def test_zoom_pan_with_empty_heatmap(self, visualizer_with_interactive):
        """Test zoom/pan with no data."""
        visualizer_with_interactive.set_zoom_level(2.0)
        visualizer_with_interactive.set_pan_offset(5, 5)
        
        # Should not crash
        assert visualizer_with_interactive.current_zoom_level == 2.0
        assert visualizer_with_interactive.current_pan_offset == (5, 5)
    
    def test_overlay_with_invalid_thumbnail(self, visualizer_with_data):
        """Test overlay with invalid thumbnail shape."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("Plotly not available")
        
        # Wrong shape thumbnail
        invalid_thumbnail = np.random.rand(10, 10)  # Missing channel dimension
        
        # Should handle gracefully (may skip thumbnail or raise)
        try:
            fig = visualizer_with_data.create_interactive_heatmap(slide_thumbnail=invalid_thumbnail)
            # If it doesn't raise, check it created something
            assert fig is not None
        except (ValueError, IndexError):
            # Expected for invalid shape
            pass
    
    def test_parameter_update_with_none_value(self, visualizer_with_interactive):
        """Test updating parameter with None value."""
        visualizer_with_interactive.update_parameter('test_param', None)
        
        assert visualizer_with_interactive.parameter_values['test_param'] is None
        
        value = visualizer_with_interactive.get_parameter('test_param')
        assert value is None
    
    def test_state_export_with_special_values(self, visualizer_with_interactive):
        """Test state export with special parameter values."""
        visualizer_with_interactive.update_parameter('int_param', 42)
        visualizer_with_interactive.update_parameter('float_param', 3.14)
        visualizer_with_interactive.update_parameter('str_param', 'test')
        visualizer_with_interactive.update_parameter('bool_param', True)
        visualizer_with_interactive.update_parameter('none_param', None)
        
        state = visualizer_with_interactive.export_visualization_state()
        
        # All should be JSON-serializable
        json_str = json.dumps(state)
        assert json_str is not None
        
        # Verify values preserved
        assert state['parameters']['int_param'] == 42
        assert state['parameters']['float_param'] == 3.14
        assert state['parameters']['str_param'] == 'test'
        assert state['parameters']['bool_param'] is True
        assert state['parameters']['none_param'] is None
