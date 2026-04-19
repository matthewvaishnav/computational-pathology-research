"""Unit tests for interpretability dashboard."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Check if Flask is available
try:
    from flask import Flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from src.interpretability.dashboard import (
    InMemoryCache,
    InterpretabilityDashboard,
    start_dashboard
)


class TestInMemoryCache:
    """Test in-memory cache implementation."""
    
    def test_cache_initialization(self):
        """Test cache initializes with correct max_size."""
        cache = InMemoryCache(max_size=50)
        assert cache.max_size == 50
        assert cache.size() == 0
    
    def test_cache_set_and_get(self):
        """Test setting and getting cache values."""
        cache = InMemoryCache()
        cache.set('key1', {'data': 'value1'})
        
        result = cache.get('key1')
        assert result == {'data': 'value1'}
    
    def test_cache_miss(self):
        """Test cache returns None for missing keys."""
        cache = InMemoryCache()
        result = cache.get('nonexistent')
        assert result is None
    
    def test_cache_eviction(self):
        """Test cache evicts oldest item when full."""
        cache = InMemoryCache(max_size=2)
        
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.set('key3', 'value3')  # Should evict key1
        
        assert cache.size() == 2
        assert cache.get('key1') is None
        assert cache.get('key2') == 'value2'
        assert cache.get('key3') == 'value3'
    
    def test_cache_clear(self):
        """Test clearing cache."""
        cache = InMemoryCache()
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        cache.clear()
        assert cache.size() == 0
        assert cache.get('key1') is None


@pytest.mark.skipif(not FLASK_AVAILABLE, reason="Flask not installed")
class TestInterpretabilityDashboard:
    """Test interpretability dashboard."""
    
    def test_dashboard_initialization(self):
        """Test dashboard initializes correctly."""
        dashboard = InterpretabilityDashboard(port=5001)
        
        assert dashboard.app is not None
        assert dashboard.port == 5001
        assert dashboard.cache is not None
        assert isinstance(dashboard.cache, InMemoryCache)
    
    def test_dashboard_with_components(self):
        """Test dashboard initializes with interpretability components."""
        mock_gradcam = Mock()
        mock_attention = Mock()
        mock_failure = Mock()
        mock_feature = Mock()
        
        dashboard = InterpretabilityDashboard(
            gradcam_generator=mock_gradcam,
            attention_visualizer=mock_attention,
            failure_analyzer=mock_failure,
            feature_importance=mock_feature
        )
        
        assert dashboard.gradcam_generator is mock_gradcam
        assert dashboard.attention_visualizer is mock_attention
        assert dashboard.failure_analyzer is mock_failure
        assert dashboard.feature_importance is mock_feature
    
    def test_dashboard_routes_registered(self):
        """Test all required routes are registered."""
        dashboard = InterpretabilityDashboard()
        
        # Get all registered routes
        routes = [rule.rule for rule in dashboard.app.url_map.iter_rules()]
        
        # Check required routes exist
        assert '/' in routes
        assert '/api/samples' in routes
        assert '/api/sample/<sample_id>' in routes
        assert '/api/filter' in routes
        assert '/api/compare' in routes
        assert '/api/export' in routes
    
    def test_index_route(self):
        """Test index route returns dashboard info."""
        dashboard = InterpretabilityDashboard()
        client = dashboard.app.test_client()
        
        response = client.get('/')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['message'] == 'Interpretability Dashboard'
        assert 'endpoints' in data
        assert data['status'] == 'running'
    
    def test_list_samples_route(self):
        """Test list samples route."""
        dashboard = InterpretabilityDashboard()
        client = dashboard.app.test_client()
        
        response = client.get('/api/samples?limit=10&offset=0')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'samples' in data
        assert data['limit'] == 10
        assert data['offset'] == 0
    
    def test_get_sample_route(self):
        """Test get sample route."""
        dashboard = InterpretabilityDashboard()
        client = dashboard.app.test_client()
        
        response = client.get('/api/sample/test_sample_123')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['sample_id'] == 'test_sample_123'
        assert 'data' in data
    
    def test_get_sample_caching(self):
        """Test sample data is cached on second request."""
        dashboard = InterpretabilityDashboard()
        client = dashboard.app.test_client()
        
        # First request
        response1 = client.get('/api/sample/test_sample_456')
        data1 = json.loads(response1.data)
        assert data1['cached'] is False
        
        # Second request should be cached
        response2 = client.get('/api/sample/test_sample_456')
        data2 = json.loads(response2.data)
        assert data2['cached'] is True
    
    def test_filter_samples_route(self):
        """Test filter samples route."""
        dashboard = InterpretabilityDashboard()
        client = dashboard.app.test_client()
        
        filter_data = {
            'min_confidence': 0.7,
            'max_confidence': 0.95,
            'correctness': True
        }
        
        response = client.post(
            '/api/filter',
            data=json.dumps(filter_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'sample_ids' in data
        assert data['filters'] == filter_data
    
    def test_compare_samples_route(self):
        """Test compare samples route."""
        dashboard = InterpretabilityDashboard()
        client = dashboard.app.test_client()
        
        compare_data = {
            'sample_ids': ['sample1', 'sample2', 'sample3'],
            'comparison_type': 'side_by_side'
        }
        
        response = client.post(
            '/api/compare',
            data=json.dumps(compare_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'comparison' in data
        assert len(data['sample_ids']) == 3
    
    def test_compare_samples_max_limit(self):
        """Test compare samples enforces max 4 samples."""
        dashboard = InterpretabilityDashboard()
        client = dashboard.app.test_client()
        
        compare_data = {
            'sample_ids': ['s1', 's2', 's3', 's4', 's5']  # 5 samples
        }
        
        response = client.post(
            '/api/compare',
            data=json.dumps(compare_data),
            content_type='application/json'
        )
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'Maximum 4 samples' in data['error']
    
    def test_export_route(self):
        """Test export visualization route."""
        dashboard = InterpretabilityDashboard()
        client = dashboard.app.test_client()
        
        export_data = {
            'sample_id': 'test_sample',
            'format': 'png',
            'dpi': 300
        }
        
        response = client.post(
            '/api/export',
            data=json.dumps(export_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        # Response should be a file download
    
    def test_export_invalid_format(self):
        """Test export rejects invalid formats."""
        dashboard = InterpretabilityDashboard()
        client = dashboard.app.test_client()
        
        export_data = {
            'sample_id': 'test_sample',
            'format': 'invalid_format'
        }
        
        response = client.post(
            '/api/export',
            data=json.dumps(export_data),
            content_type='application/json'
        )
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'Invalid format' in data['error']
    
    def test_load_sample(self):
        """Test load_sample method."""
        dashboard = InterpretabilityDashboard()
        
        sample_data = dashboard.load_sample('test_sample')
        
        assert sample_data['sample_id'] == 'test_sample'
        assert 'gradcam_heatmaps' in sample_data
        assert 'attention_weights' in sample_data
        assert 'prediction' in sample_data
        assert 'confidence' in sample_data
        assert 'clinical_features' in sample_data
    
    def test_filter_samples(self):
        """Test filter_samples method."""
        dashboard = InterpretabilityDashboard()
        
        filtered = dashboard.filter_samples(
            min_confidence=0.8,
            max_confidence=0.95,
            correctness=True
        )
        
        assert isinstance(filtered, list)
    
    def test_compare_samples(self):
        """Test compare_samples method."""
        dashboard = InterpretabilityDashboard()
        
        comparison = dashboard.compare_samples(
            sample_ids=['s1', 's2'],
            comparison_type='side_by_side'
        )
        
        assert 'samples' in comparison
        assert comparison['comparison_type'] == 'side_by_side'
        assert len(comparison['samples']) == 2
    
    def test_compare_samples_validates_count(self):
        """Test compare_samples validates max 4 samples."""
        dashboard = InterpretabilityDashboard()
        
        with pytest.raises(ValueError, match="Maximum 4 samples"):
            dashboard.compare_samples(
                sample_ids=['s1', 's2', 's3', 's4', 's5']
            )
    
    def test_compare_samples_validates_type(self):
        """Test compare_samples validates comparison type."""
        dashboard = InterpretabilityDashboard()
        
        with pytest.raises(ValueError, match="Invalid comparison_type"):
            dashboard.compare_samples(
                sample_ids=['s1', 's2'],
                comparison_type='invalid_type'
            )
    
    def test_export_visualization(self):
        """Test export_visualization method."""
        dashboard = InterpretabilityDashboard()
        
        output_path = dashboard.export_visualization(
            sample_id='test_sample',
            output_format='png',
            dpi=300
        )
        
        assert output_path.exists()
        assert output_path.suffix == '.png'
    
    def test_export_validates_format(self):
        """Test export_visualization validates format."""
        dashboard = InterpretabilityDashboard()
        
        with pytest.raises(ValueError, match="Invalid format"):
            dashboard.export_visualization(
                sample_id='test_sample',
                output_format='invalid'
            )
    
    def test_clear_cache(self):
        """Test clearing dashboard cache."""
        dashboard = InterpretabilityDashboard()
        
        # Add some cached data
        dashboard.cache.set('key1', 'value1')
        assert dashboard.cache.size() > 0
        
        # Clear cache
        dashboard.clear_cache()
        assert dashboard.cache.size() == 0
    
    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        dashboard = InterpretabilityDashboard()
        
        stats = dashboard.get_cache_stats()
        
        assert 'backend' in stats
        assert stats['backend'] == 'memory'
        assert 'size' in stats
        assert 'max_size' in stats


@pytest.mark.skipif(not FLASK_AVAILABLE, reason="Flask not installed")
def test_start_dashboard_function():
    """Test start_dashboard convenience function."""
    # We can't actually start the server in tests, but we can verify
    # the function exists and accepts the right parameters
    assert callable(start_dashboard)


def test_flask_not_available_error():
    """Test dashboard raises error when Flask is not available."""
    with patch('src.interpretability.dashboard.FLASK_AVAILABLE', False):
        with pytest.raises(RuntimeError, match="Flask is not installed"):
            InterpretabilityDashboard()
