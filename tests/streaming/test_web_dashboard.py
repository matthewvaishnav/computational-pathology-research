"""Unit tests for web-based dashboard."""

import pytest
import asyncio
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json

from src.streaming.web_dashboard import (
    app,
    dashboard_state,
    connection_manager,
    ProcessingStatus,
    HeatmapData,
    ConfidenceData,
    ProcessingParameters,
    ProcessingRequest,
    DashboardState,
    update_dashboard_status,
    update_dashboard_error,
    update_dashboard_complete
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def reset_dashboard_state():
    """Reset dashboard state before each test."""
    dashboard_state.reset()
    yield
    dashboard_state.reset()


@pytest.fixture
def sample_heatmap():
    """Create sample heatmap data."""
    heatmap = np.random.rand(10, 10).astype(np.float32)
    coverage = np.random.rand(10, 10) > 0.5
    return heatmap, coverage


class TestDashboardState:
    """Tests for DashboardState class."""
    
    def test_initialization(self):
        """Test DashboardState initialization."""
        state = DashboardState()
        
        assert state.slide_id == ""
        assert state.status == "idle"
        assert state.patches_processed == 0
        assert state.total_patches == 0
        assert state.current_confidence == 0.0
        assert state.confidence_history == []
        assert state.parameters is not None
    
    def test_reset(self):
        """Test state reset."""
        state = DashboardState()
        
        # Modify state
        state.slide_id = "test_slide"
        state.status = "processing"
        state.patches_processed = 100
        state.total_patches = 1000
        state.current_confidence = 0.85
        state.confidence_history = [(1.0, 0.5), (2.0, 0.7)]
        
        # Reset
        state.reset()
        
        assert state.slide_id == ""
        assert state.status == "idle"
        assert state.patches_processed == 0
        assert state.total_patches == 0
        assert state.current_confidence == 0.0
        assert state.confidence_history == []
    
    def test_get_progress_percent(self):
        """Test progress percentage calculation."""
        state = DashboardState()
        
        # No patches
        assert state.get_progress_percent() == 0.0
        
        # 50% progress
        state.patches_processed = 500
        state.total_patches = 1000
        assert state.get_progress_percent() == 50.0
        
        # 100% progress
        state.patches_processed = 1000
        assert state.get_progress_percent() == 100.0
    
    def test_get_elapsed_time(self):
        """Test elapsed time calculation."""
        state = DashboardState()
        
        # No start time
        assert state.get_elapsed_time() == 0.0
        
        # With start time
        import time
        state.start_time = time.time() - 5.0
        elapsed = state.get_elapsed_time()
        assert 4.9 < elapsed < 5.1  # Allow small tolerance
    
    def test_get_estimated_remaining(self):
        """Test estimated remaining time calculation."""
        state = DashboardState()
        
        # No progress
        assert state.get_estimated_remaining() == 0.0
        
        # 50% progress after 10 seconds
        import time
        state.start_time = time.time() - 10.0
        state.patches_processed = 500
        state.total_patches = 1000
        
        remaining = state.get_estimated_remaining()
        assert 9.0 < remaining < 11.0  # Should be ~10 seconds remaining
    
    def test_get_throughput(self):
        """Test throughput calculation."""
        state = DashboardState()
        
        # No time elapsed
        assert state.get_throughput() == 0.0
        
        # 100 patches in 10 seconds
        import time
        state.start_time = time.time() - 10.0
        state.patches_processed = 100
        
        throughput = state.get_throughput()
        assert 9.0 < throughput < 11.0  # Should be ~10 patches/sec


class TestRESTEndpoints:
    """Tests for REST API endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "active_connections" in data
    
    def test_get_status_idle(self, client, reset_dashboard_state):
        """Test status endpoint when idle."""
        response = client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"
        assert data["patches_processed"] == 0
        assert data["total_patches"] == 0
        assert data["progress_percent"] == 0.0
        assert data["current_confidence"] == 0.0
    
    def test_get_status_processing(self, client, reset_dashboard_state):
        """Test status endpoint during processing."""
        # Set up processing state
        dashboard_state.slide_id = "test_slide"
        dashboard_state.status = "processing"
        dashboard_state.patches_processed = 500
        dashboard_state.total_patches = 1000
        dashboard_state.current_confidence = 0.85
        
        import time
        dashboard_state.start_time = time.time() - 10.0
        
        response = client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["slide_id"] == "test_slide"
        assert data["patches_processed"] == 500
        assert data["total_patches"] == 1000
        assert data["progress_percent"] == 50.0
        assert data["current_confidence"] == 0.85
        assert data["elapsed_time"] > 0
        assert data["throughput"] > 0
    
    def test_get_heatmap_no_data(self, client, reset_dashboard_state):
        """Test heatmap endpoint with no data."""
        response = client.get("/api/heatmap")
        
        assert response.status_code == 404
        assert "No heatmap data available" in response.json()["detail"]
    
    def test_get_heatmap_with_data(self, client, reset_dashboard_state, sample_heatmap):
        """Test heatmap endpoint with data."""
        heatmap, coverage = sample_heatmap
        
        dashboard_state.slide_id = "test_slide"
        dashboard_state.attention_heatmap = heatmap
        dashboard_state.coverage_mask = coverage
        dashboard_state.heatmap_dimensions = heatmap.shape
        
        response = client.get("/api/heatmap")
        
        assert response.status_code == 200
        data = response.json()
        assert data["slide_id"] == "test_slide"
        assert len(data["heatmap"]) == heatmap.shape[0]
        assert len(data["heatmap"][0]) == heatmap.shape[1]
        assert data["dimensions"] == list(heatmap.shape)
        assert 0 <= data["coverage_percent"] <= 100
    
    def test_get_confidence_no_data(self, client, reset_dashboard_state):
        """Test confidence endpoint with no data."""
        response = client.get("/api/confidence")
        
        assert response.status_code == 404
        assert "No confidence data available" in response.json()["detail"]
    
    def test_get_confidence_with_data(self, client, reset_dashboard_state):
        """Test confidence endpoint with data."""
        dashboard_state.slide_id = "test_slide"
        dashboard_state.confidence_history = [
            (0.0, 0.5),
            (1.0, 0.7),
            (2.0, 0.85),
            (3.0, 0.92)
        ]
        
        response = client.get("/api/confidence")
        
        assert response.status_code == 200
        data = response.json()
        assert data["slide_id"] == "test_slide"
        assert len(data["timestamps"]) == 4
        assert len(data["confidences"]) == 4
        assert data["timestamps"][0] == 0.0  # Normalized to start at 0
        assert data["confidences"] == [0.5, 0.7, 0.85, 0.92]
        assert data["target_threshold"] == 0.95
    
    def test_get_parameters(self, client, reset_dashboard_state):
        """Test get parameters endpoint."""
        response = client.get("/api/parameters")
        
        assert response.status_code == 200
        data = response.json()
        assert "confidence_threshold" in data
        assert "batch_size" in data
        assert "tile_size" in data
        assert "update_interval" in data
        assert "enable_early_stopping" in data
    
    @pytest.mark.asyncio
    async def test_update_parameters(self, client, reset_dashboard_state):
        """Test update parameters endpoint."""
        new_params = {
            "confidence_threshold": 0.90,
            "batch_size": 128,
            "tile_size": 2048,
            "update_interval": 2.0,
            "enable_early_stopping": False
        }
        
        response = client.post("/api/parameters", json=new_params)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["parameters"]["confidence_threshold"] == 0.90
        assert data["parameters"]["batch_size"] == 128
        
        # Verify parameters were updated
        assert dashboard_state.parameters.confidence_threshold == 0.90
        assert dashboard_state.parameters.batch_size == 128
    
    def test_update_parameters_validation(self, client):
        """Test parameter validation."""
        # Invalid confidence threshold
        invalid_params = {
            "confidence_threshold": 1.5,  # > 1.0
            "batch_size": 64,
            "tile_size": 1024,
            "update_interval": 1.0,
            "enable_early_stopping": True
        }
        
        response = client.post("/api/parameters", json=invalid_params)
        assert response.status_code == 422  # Validation error
    
    def test_stop_processing_not_running(self, client, reset_dashboard_state):
        """Test stop processing when not running."""
        response = client.post("/api/stop")
        
        assert response.status_code == 400
        assert "No processing in progress" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_stop_processing_running(self, client, reset_dashboard_state):
        """Test stop processing when running."""
        dashboard_state.status = "processing"
        
        response = client.post("/api/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert dashboard_state.status == "idle"


class TestPydanticModels:
    """Tests for Pydantic models."""
    
    def test_processing_status_model(self):
        """Test ProcessingStatus model."""
        status = ProcessingStatus(
            slide_id="test",
            status="processing",
            patches_processed=100,
            total_patches=1000,
            progress_percent=10.0,
            current_confidence=0.75,
            elapsed_time=5.0,
            estimated_remaining=45.0,
            throughput=20.0
        )
        
        assert status.slide_id == "test"
        assert status.patches_processed == 100
        assert status.progress_percent == 10.0
    
    def test_processing_status_validation(self):
        """Test ProcessingStatus validation."""
        # Invalid progress percent
        with pytest.raises(Exception):
            ProcessingStatus(
                slide_id="test",
                status="processing",
                patches_processed=100,
                total_patches=1000,
                progress_percent=150.0,  # > 100
                current_confidence=0.75,
                elapsed_time=5.0,
                estimated_remaining=45.0,
                throughput=20.0
            )
    
    def test_processing_parameters_model(self):
        """Test ProcessingParameters model."""
        params = ProcessingParameters(
            confidence_threshold=0.95,
            batch_size=64,
            tile_size=1024,
            update_interval=1.0,
            enable_early_stopping=True
        )
        
        assert params.confidence_threshold == 0.95
        assert params.batch_size == 64
        assert params.enable_early_stopping is True
    
    def test_processing_parameters_defaults(self):
        """Test ProcessingParameters defaults."""
        params = ProcessingParameters()
        
        assert params.confidence_threshold == 0.95
        assert params.batch_size == 64
        assert params.tile_size == 1024
        assert params.update_interval == 1.0
        assert params.enable_early_stopping is True
    
    def test_heatmap_data_model(self):
        """Test HeatmapData model."""
        heatmap = [[0.1, 0.2], [0.3, 0.4]]
        
        data = HeatmapData(
            slide_id="test",
            heatmap=heatmap,
            dimensions=(2, 2),
            coverage_percent=75.0,
            timestamp=123456.0
        )
        
        assert data.slide_id == "test"
        assert data.heatmap == heatmap
        assert data.dimensions == (2, 2)
        assert data.coverage_percent == 75.0
    
    def test_confidence_data_model(self):
        """Test ConfidenceData model."""
        data = ConfidenceData(
            slide_id="test",
            timestamps=[0.0, 1.0, 2.0],
            confidences=[0.5, 0.7, 0.9],
            target_threshold=0.95
        )
        
        assert data.slide_id == "test"
        assert len(data.timestamps) == 3
        assert len(data.confidences) == 3
        assert data.target_threshold == 0.95


class TestUpdateFunctions:
    """Tests for dashboard update functions."""
    
    @pytest.mark.asyncio
    async def test_update_dashboard_status(self, reset_dashboard_state):
        """Test updating dashboard status."""
        attention_weights = np.array([0.1, 0.2, 0.3])
        coordinates = np.array([[0, 0], [1, 1], [2, 2]])
        
        await update_dashboard_status(
            patches_processed=100,
            total_patches=1000,
            confidence=0.85,
            attention_weights=attention_weights,
            coordinates=coordinates
        )
        
        assert dashboard_state.patches_processed == 100
        assert dashboard_state.total_patches == 1000
        assert dashboard_state.current_confidence == 0.85
        assert len(dashboard_state.confidence_history) == 1
        assert dashboard_state.confidence_history[0][1] == 0.85
    
    @pytest.mark.asyncio
    async def test_update_dashboard_status_no_heatmap(self, reset_dashboard_state):
        """Test updating status without heatmap data."""
        await update_dashboard_status(
            patches_processed=50,
            total_patches=500,
            confidence=0.70
        )
        
        assert dashboard_state.patches_processed == 50
        assert dashboard_state.current_confidence == 0.70
        assert dashboard_state.attention_heatmap is None
    
    @pytest.mark.asyncio
    async def test_update_dashboard_error(self, reset_dashboard_state):
        """Test updating dashboard with error."""
        error_msg = "Test error message"
        
        await update_dashboard_error(error_msg)
        
        assert dashboard_state.status == "error"
        assert dashboard_state.last_error == error_msg
    
    @pytest.mark.asyncio
    async def test_update_dashboard_complete(self, reset_dashboard_state):
        """Test marking processing as complete."""
        dashboard_state.slide_id = "test_slide"
        dashboard_state.current_confidence = 0.96
        
        await update_dashboard_complete()
        
        assert dashboard_state.status == "completed"


class TestConnectionManager:
    """Tests for WebSocket connection manager."""
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self):
        """Test connecting WebSocket."""
        manager = connection_manager
        initial_count = len(manager.active_connections)
        
        # Mock WebSocket
        mock_ws = AsyncMock()
        await manager.connect(mock_ws)
        
        assert len(manager.active_connections) == initial_count + 1
        assert mock_ws in manager.active_connections
        
        # Cleanup
        manager.disconnect(mock_ws)
    
    def test_disconnect_websocket(self):
        """Test disconnecting WebSocket."""
        manager = connection_manager
        
        # Mock WebSocket
        mock_ws = Mock()
        manager.active_connections.append(mock_ws)
        initial_count = len(manager.active_connections)
        
        manager.disconnect(mock_ws)
        
        assert len(manager.active_connections) == initial_count - 1
        assert mock_ws not in manager.active_connections
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self):
        """Test broadcasting message to all connections."""
        manager = connection_manager
        
        # Create mock WebSockets
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        
        await manager.connect(mock_ws1)
        await manager.connect(mock_ws2)
        
        # Broadcast message
        test_message = {"type": "test", "data": "hello"}
        await manager.broadcast(test_message)
        
        # Verify both received message
        mock_ws1.send_json.assert_called_once_with(test_message)
        mock_ws2.send_json.assert_called_once_with(test_message)
        
        # Cleanup
        manager.disconnect(mock_ws1)
        manager.disconnect(mock_ws2)
    
    @pytest.mark.asyncio
    async def test_send_personal_message(self):
        """Test sending message to specific connection."""
        manager = connection_manager
        
        mock_ws = AsyncMock()
        await manager.connect(mock_ws)
        
        test_message = {"type": "test", "data": "personal"}
        await manager.send_personal(test_message, mock_ws)
        
        mock_ws.send_json.assert_called_once_with(test_message)
        
        # Cleanup
        manager.disconnect(mock_ws)


class TestIntegration:
    """Integration tests for web dashboard."""
    
    @pytest.mark.asyncio
    async def test_full_processing_workflow(self, client, reset_dashboard_state):
        """Test complete processing workflow."""
        # 1. Check initial status
        response = client.get("/api/status")
        assert response.json()["status"] == "idle"
        
        # 2. Update parameters
        params = {
            "confidence_threshold": 0.90,
            "batch_size": 128,
            "tile_size": 1024,
            "update_interval": 1.0,
            "enable_early_stopping": True
        }
        response = client.post("/api/parameters", json=params)
        assert response.status_code == 200
        
        # 3. Simulate processing updates
        for i in range(5):
            await update_dashboard_status(
                patches_processed=(i + 1) * 200,
                total_patches=1000,
                confidence=0.5 + (i * 0.1)
            )
        
        # 4. Check status during processing
        response = client.get("/api/status")
        data = response.json()
        assert data["patches_processed"] == 1000
        assert data["current_confidence"] == 0.9
        
        # 5. Check confidence history
        response = client.get("/api/confidence")
        assert response.status_code == 200
        data = response.json()
        assert len(data["confidences"]) == 5
        
        # 6. Complete processing
        await update_dashboard_complete()
        assert dashboard_state.status == "completed"
    
    def test_cors_headers(self, client):
        """Test CORS middleware is configured."""
        # Note: TestClient doesn't trigger CORS middleware
        # This test verifies the middleware is configured in the app
        from fastapi.middleware.cors import CORSMiddleware
        
        # Check that CORS middleware is in the app
        cors_middleware_found = False
        for middleware in app.user_middleware:
            if middleware.cls == CORSMiddleware:
                cors_middleware_found = True
                break
        
        assert cors_middleware_found, "CORS middleware not configured"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
