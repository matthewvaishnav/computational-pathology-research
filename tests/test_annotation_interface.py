"""
Tests for Expert Annotation Interface
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

from src.annotation_interface.backend.annotation_api import app
from src.annotation_interface.backend.annotation_models import (
    AnnotationCreate,
    AnnotationUpdate,
    AnnotationType,
    AnnotationLabel,
    Point,
    AnnotationGeometry,
    AnnotationQueueItem
)


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_annotation_data():
    """Sample annotation data for testing"""
    return {
        "slide_id": "test_slide_001",
        "task_id": "test_task_001",
        "label": "tumor",
        "geometry": {
            "type": "polygon",
            "points": [
                {"x": 100.0, "y": 100.0},
                {"x": 200.0, "y": 100.0},
                {"x": 200.0, "y": 200.0},
                {"x": 100.0, "y": 200.0}
            ]
        },
        "confidence": 0.95,
        "comments": "Clear tumor region",
        "expert_id": "expert_001"
    }


@pytest.fixture
def sample_queue_item():
    """Sample queue item for testing"""
    return AnnotationQueueItem(
        task_id="test_task_001",
        slide_id="test_slide_001",
        priority=0.9,
        uncertainty_score=0.85,
        ai_prediction={"diagnosis": "tumor", "confidence": 0.65},
        status="pending",
        created_at=datetime.now()
    )


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test health check returns 200"""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "annotations_count" in data


class TestAnnotationCRUD:
    """Test annotation CRUD operations"""
    
    def test_create_annotation(self, client, sample_annotation_data):
        """Test creating a new annotation"""
        response = client.post("/api/annotations", json=sample_annotation_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["annotation"] is not None
        
        annotation = data["annotation"]
        assert annotation["slide_id"] == sample_annotation_data["slide_id"]
        assert annotation["label"] == sample_annotation_data["label"]
        assert annotation["confidence"] == sample_annotation_data["confidence"]
        assert "id" in annotation
        assert "created_at" in annotation
    
    def test_get_annotations(self, client, sample_annotation_data):
        """Test retrieving annotations"""
        # Create annotation first
        create_response = client.post("/api/annotations", json=sample_annotation_data)
        assert create_response.status_code == 200
        
        # Get all annotations
        response = client.get("/api/annotations")
        assert response.status_code == 200
        
        annotations = response.json()
        assert isinstance(annotations, list)
        assert len(annotations) > 0
    
    def test_get_annotations_by_slide(self, client, sample_annotation_data):
        """Test filtering annotations by slide_id"""
        # Create annotation
        client.post("/api/annotations", json=sample_annotation_data)
        
        # Get annotations for specific slide
        response = client.get(
            "/api/annotations",
            params={"slide_id": sample_annotation_data["slide_id"]}
        )
        assert response.status_code == 200
        
        annotations = response.json()
        assert all(a["slide_id"] == sample_annotation_data["slide_id"] for a in annotations)
    
    def test_get_annotation_by_id(self, client, sample_annotation_data):
        """Test retrieving specific annotation"""
        # Create annotation
        create_response = client.post("/api/annotations", json=sample_annotation_data)
        annotation_id = create_response.json()["annotation"]["id"]
        
        # Get specific annotation
        response = client.get(f"/api/annotations/{annotation_id}")
        assert response.status_code == 200
        
        annotation = response.json()
        assert annotation["id"] == annotation_id
    
    def test_update_annotation(self, client, sample_annotation_data):
        """Test updating an annotation"""
        # Create annotation
        create_response = client.post("/api/annotations", json=sample_annotation_data)
        annotation_id = create_response.json()["annotation"]["id"]
        
        # Update annotation
        update_data = {
            "label": "normal",
            "confidence": 0.98,
            "comments": "Updated comment"
        }
        response = client.put(f"/api/annotations/{annotation_id}", json=update_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        
        annotation = data["annotation"]
        assert annotation["label"] == update_data["label"]
        assert annotation["confidence"] == update_data["confidence"]
        assert annotation["comments"] == update_data["comments"]
    
    def test_delete_annotation(self, client, sample_annotation_data):
        """Test deleting an annotation"""
        # Create annotation
        create_response = client.post("/api/annotations", json=sample_annotation_data)
        annotation_id = create_response.json()["annotation"]["id"]
        
        # Delete annotation
        response = client.delete(f"/api/annotations/{annotation_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        
        # Verify deletion
        get_response = client.get(f"/api/annotations/{annotation_id}")
        assert get_response.status_code == 404
    
    def test_get_nonexistent_annotation(self, client):
        """Test getting annotation that doesn't exist"""
        response = client.get("/api/annotations/nonexistent_id")
        assert response.status_code == 404


class TestAnnotationQueue:
    """Test annotation queue endpoints"""
    
    def test_get_empty_queue(self, client):
        """Test getting empty queue"""
        response = client.get("/api/queue")
        assert response.status_code == 200
        
        queue = response.json()
        assert isinstance(queue, list)
    
    def test_get_queue_with_limit(self, client):
        """Test queue pagination"""
        response = client.get("/api/queue", params={"limit": 5})
        assert response.status_code == 200
        
        queue = response.json()
        assert len(queue) <= 5
    
    def test_assign_task(self, client, sample_queue_item):
        """Test assigning task to expert"""
        # Add task to queue first (would be done by active learning system)
        from src.annotation_interface.backend.annotation_api import add_task_to_queue
        add_task_to_queue(sample_queue_item)
        
        # Assign task
        response = client.post(
            f"/api/queue/{sample_queue_item.task_id}/assign",
            params={"expert_id": "expert_001"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["task"]["assigned_expert"] == "expert_001"
        assert data["task"]["status"] == "in_progress"
    
    def test_complete_task(self, client, sample_queue_item):
        """Test completing a task"""
        # Add task to queue
        from src.annotation_interface.backend.annotation_api import add_task_to_queue
        add_task_to_queue(sample_queue_item)
        
        # Complete task
        response = client.post(f"/api/queue/{sample_queue_item.task_id}/complete")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["task"]["status"] == "completed"


class TestSlideEndpoints:
    """Test slide-related endpoints"""
    
    def test_get_slide_info_not_found(self, client):
        """Test getting info for non-existent slide"""
        response = client.get("/api/slides/nonexistent_slide")
        assert response.status_code == 404
    
    def test_get_slide_tile(self, client):
        """Test getting slide tile (placeholder)"""
        response = client.get("/api/slides/test_slide/tile/0/0/0")
        assert response.status_code == 200
        
        data = response.json()
        assert "slide_id" in data
        assert "z" in data
        assert "x" in data
        assert "y" in data
    
    def test_get_ai_prediction(self, client):
        """Test getting AI prediction overlay"""
        response = client.get("/api/slides/test_slide/ai-prediction")
        assert response.status_code == 200
        
        data = response.json()
        assert "slide_id" in data
        assert "prediction_type" in data
        assert "confidence" in data


class TestDataModels:
    """Test Pydantic data models"""
    
    def test_annotation_geometry_polygon(self):
        """Test polygon geometry model"""
        geometry = AnnotationGeometry(
            type=AnnotationType.POLYGON,
            points=[
                Point(x=0, y=0),
                Point(x=100, y=0),
                Point(x=100, y=100),
                Point(x=0, y=100)
            ]
        )
        assert geometry.type == AnnotationType.POLYGON
        assert len(geometry.points) == 4
    
    def test_annotation_geometry_circle(self):
        """Test circle geometry model"""
        geometry = AnnotationGeometry(
            type=AnnotationType.CIRCLE,
            center=Point(x=50, y=50),
            radius=25.0
        )
        assert geometry.type == AnnotationType.CIRCLE
        assert geometry.center.x == 50
        assert geometry.radius == 25.0
    
    def test_annotation_create_validation(self):
        """Test annotation creation validation"""
        data = AnnotationCreate(
            slide_id="test_slide",
            label=AnnotationLabel.TUMOR,
            geometry=AnnotationGeometry(
                type=AnnotationType.POLYGON,
                points=[Point(x=0, y=0), Point(x=100, y=100)]
            ),
            confidence=0.95,
            expert_id="expert_001"
        )
        assert data.confidence == 0.95
        assert data.label == AnnotationLabel.TUMOR
    
    def test_annotation_confidence_bounds(self):
        """Test confidence value bounds"""
        # Valid confidence
        data = AnnotationCreate(
            slide_id="test",
            label=AnnotationLabel.TUMOR,
            geometry=AnnotationGeometry(type=AnnotationType.POINT, points=[]),
            confidence=0.5,
            expert_id="expert"
        )
        assert 0.0 <= data.confidence <= 1.0
        
        # Invalid confidence should raise validation error
        with pytest.raises(Exception):
            AnnotationCreate(
                slide_id="test",
                label=AnnotationLabel.TUMOR,
                geometry=AnnotationGeometry(type=AnnotationType.POINT, points=[]),
                confidence=1.5,  # Invalid: > 1.0
                expert_id="expert"
            )


class TestWebSocketConnection:
    """Test WebSocket functionality"""
    
    def test_websocket_connection(self, client):
        """Test WebSocket connection establishment"""
        with client.websocket_connect("/ws/test_slide") as websocket:
            # Connection should be established
            assert websocket is not None
    
    def test_websocket_broadcast(self, client):
        """Test WebSocket message broadcasting"""
        with client.websocket_connect("/ws/test_slide") as websocket:
            # Send a message
            test_message = {"type": "test", "data": "hello"}
            websocket.send_json(test_message)
            
            # Receive echoed message
            data = websocket.receive_json()
            assert data["type"] == "user_action"
            assert data["data"] == test_message


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_complete_annotation_workflow(self, client, sample_annotation_data, sample_queue_item):
        """Test complete workflow from queue to annotation"""
        # 1. Add task to queue
        from src.annotation_interface.backend.annotation_api import add_task_to_queue
        add_task_to_queue(sample_queue_item)
        
        # 2. Get queue
        queue_response = client.get("/api/queue")
        assert queue_response.status_code == 200
        queue = queue_response.json()
        assert len(queue) > 0
        
        # 3. Assign task
        assign_response = client.post(
            f"/api/queue/{sample_queue_item.task_id}/assign",
            params={"expert_id": "expert_001"}
        )
        assert assign_response.status_code == 200
        
        # 4. Create annotation
        create_response = client.post("/api/annotations", json=sample_annotation_data)
        assert create_response.status_code == 200
        annotation_id = create_response.json()["annotation"]["id"]
        
        # 5. Get annotations for slide
        annotations_response = client.get(
            "/api/annotations",
            params={"slide_id": sample_annotation_data["slide_id"]}
        )
        assert annotations_response.status_code == 200
        annotations = annotations_response.json()
        assert len(annotations) > 0
        
        # 6. Complete task
        complete_response = client.post(f"/api/queue/{sample_queue_item.task_id}/complete")
        assert complete_response.status_code == 200
        
        # 7. Verify annotation exists
        get_response = client.get(f"/api/annotations/{annotation_id}")
        assert get_response.status_code == 200


class TestAnnotationTimeTracking:
    """Test annotation time tracking features"""
    
    def test_task_time_tracking(self, client, sample_queue_item):
        """Test that task assignment and completion track time"""
        from src.annotation_interface.backend.annotation_api import add_task_to_queue
        add_task_to_queue(sample_queue_item)
        
        # Assign task
        assign_response = client.post(
            f"/api/queue/{sample_queue_item.task_id}/assign",
            params={"expert_id": "expert_001"}
        )
        assert assign_response.status_code == 200
        task_data = assign_response.json()["task"]
        assert task_data["started_at"] is not None
        
        # Complete task
        import time
        time.sleep(0.1)  # Small delay to ensure time difference
        complete_response = client.post(f"/api/queue/{sample_queue_item.task_id}/complete")
        assert complete_response.status_code == 200
        
        completed_task = complete_response.json()["task"]
        assert completed_task["completed_at"] is not None
        assert completed_task["annotation_time_seconds"] is not None
        assert completed_task["annotation_time_seconds"] > 0
    
    def test_annotation_time_analytics_endpoint(self, client, sample_annotation_data):
        """Test annotation time analytics endpoint"""
        # Create annotation with time data
        client.post("/api/annotations", json=sample_annotation_data)
        
        # Get time analytics
        response = client.get("/api/quality/annotation-time-analytics")
        assert response.status_code == 200
        
        data = response.json()
        assert "time_window_days" in data
        assert "total_annotations" in data
        assert "overall_stats" in data or "message" in data
    
    def test_annotation_time_analytics_by_expert(self, client, sample_annotation_data):
        """Test annotation time analytics filtered by expert"""
        # Create annotation
        client.post("/api/annotations", json=sample_annotation_data)
        
        # Get time analytics for specific expert
        response = client.get(
            "/api/quality/annotation-time-analytics",
            params={"expert_id": sample_annotation_data["expert_id"]}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["expert_id"] == sample_annotation_data["expert_id"]
    
    def test_quality_trends_endpoint(self, client, sample_annotation_data):
        """Test quality trends over time endpoint"""
        # Create some annotations
        client.post("/api/annotations", json=sample_annotation_data)
        
        # Get quality trends
        response = client.get("/api/quality/quality-trends")
        assert response.status_code == 200
        
        data = response.json()
        assert "time_window_days" in data
        assert "granularity" in data
        assert "trends" in data
        assert isinstance(data["trends"], list)
    
    def test_quality_trends_weekly_granularity(self, client, sample_annotation_data):
        """Test quality trends with weekly granularity"""
        # Create annotation
        client.post("/api/annotations", json=sample_annotation_data)
        
        # Get weekly trends
        response = client.get(
            "/api/quality/quality-trends",
            params={"granularity": "weekly"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["granularity"] == "weekly"
    
    def test_expert_performance_includes_time_metrics(self, client, sample_annotation_data):
        """Test that expert performance includes time metrics"""
        # Create annotation
        client.post("/api/annotations", json=sample_annotation_data)
        
        # Get expert performance
        response = client.get(
            f"/api/quality/expert/{sample_annotation_data['expert_id']}/performance"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "avg_annotation_time_seconds" in data
        assert "median_annotation_time_seconds" in data


class TestQualityControlEndpoints:
    """Test quality control API endpoints"""
    
    def test_validate_annotation_endpoint(self, client, sample_annotation_data):
        """Test annotation validation endpoint"""
        # Create annotation
        create_response = client.post("/api/annotations", json=sample_annotation_data)
        annotation_id = create_response.json()["annotation"]["id"]
        
        # Validate annotation
        response = client.get(f"/api/quality/validate/{annotation_id}")
        assert response.status_code == 200
        
        validation = response.json()
        assert "is_valid" in validation
        assert "quality_score" in validation
        assert "quality_status" in validation
    
    def test_get_quality_metrics(self, client, sample_annotation_data):
        """Test quality metrics endpoint"""
        # Create some annotations
        client.post("/api/annotations", json=sample_annotation_data)
        
        # Get metrics
        response = client.get("/api/quality/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "metrics" in data
        assert "alerts" in data
        
        metrics = data["metrics"]
        assert "total_annotations" in metrics
        assert "avg_confidence" in metrics
        assert "validation_rate" in metrics
    
    def test_get_metrics_history(self, client):
        """Test metrics history endpoint"""
        response = client.get("/api/quality/metrics/history")
        assert response.status_code == 200
        
        data = response.json()
        assert "history" in data
        assert isinstance(data["history"], list)
    
    def test_inter_rater_agreement_endpoint(self, client):
        """Test inter-rater agreement endpoint"""
        # Create annotations from two experts
        annotation_data_1 = {
            "slide_id": "test_slide_001",
            "task_id": None,
            "label": "tumor",
            "geometry": {
                "type": "circle",
                "center": {"x": 50.0, "y": 50.0},
                "radius": 20.0
            },
            "confidence": 0.9,
            "comments": "Expert 1 annotation",
            "expert_id": "expert_001"
        }
        
        annotation_data_2 = {
            "slide_id": "test_slide_001",
            "task_id": None,
            "label": "tumor",
            "geometry": {
                "type": "circle",
                "center": {"x": 55.0, "y": 55.0},
                "radius": 20.0
            },
            "confidence": 0.85,
            "comments": "Expert 2 annotation",
            "expert_id": "expert_002"
        }
        
        client.post("/api/annotations", json=annotation_data_1)
        client.post("/api/annotations", json=annotation_data_2)
        
        # Calculate inter-rater agreement
        response = client.get(
            "/api/quality/inter-rater-agreement",
            params={
                "slide_id": "test_slide_001",
                "expert_id_1": "expert_001",
                "expert_id_2": "expert_002"
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "agreement_metrics" in data
        assert "kappa" in data["agreement_metrics"]
    
    def test_get_quality_alerts(self, client, sample_annotation_data):
        """Test quality alerts endpoint"""
        # Create annotation
        client.post("/api/annotations", json=sample_annotation_data)
        
        # Get alerts
        response = client.get("/api/quality/alerts")
        assert response.status_code == 200
        
        data = response.json()
        assert "alerts" in data
        assert "alert_count" in data
        assert isinstance(data["alerts"], list)
    
    def test_expert_performance_endpoint(self, client, sample_annotation_data):
        """Test expert performance endpoint"""
        # Create annotations
        client.post("/api/annotations", json=sample_annotation_data)
        
        # Get expert performance
        response = client.get(
            f"/api/quality/expert/{sample_annotation_data['expert_id']}/performance"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "expert_id" in data
        assert "total_annotations" in data
        assert "avg_confidence" in data
        assert "avg_quality_score" in data
        assert "quality_distribution" in data
    
    def test_expert_performance_not_found(self, client):
        """Test expert performance for non-existent expert"""
        response = client.get("/api/quality/expert/nonexistent_expert/performance")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
