"""
Integration tests for clinical workflow integration.

Tests the complete workflow from active learning to annotation interface.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.annotation_interface.workflow.active_learning_connector import ActiveLearningConnector
from src.annotation_interface.workflow.clinical_integration import ClinicalWorkflowIntegrator
from src.annotation_interface.workflow.notification_service import (
    NotificationChannel,
    NotificationPriority,
    NotificationService,
)
from src.annotation_interface.workflow.pacs_connector import PACSConnector
from src.clinical.pacs.data_models import OperationResult
from src.clinical.pacs.pacs_adapter import PACSAdapter
from src.continuous_learning.active_learning import (
    ActiveLearningSystem,
    AnnotationTask,
    CaseForReview,
    ExpertAnnotation,
)


class TestActiveLearningConnector:
    """Test active learning connector."""

    @pytest.fixture
    def mock_al_system(self):
        """Create mock active learning system."""
        al_system = Mock(spec=ActiveLearningSystem)
        al_system.get_annotation_queue = Mock(return_value=[])
        al_system.get_statistics = Mock(
            return_value={
                "cases_identified": 10,
                "annotations_received": 5,
                "retraining_triggered": 1,
            }
        )
        return al_system

    @pytest.fixture
    def connector(self, mock_al_system):
        """Create active learning connector."""
        return ActiveLearningConnector(mock_al_system)

    @pytest.mark.asyncio
    async def test_connector_start_stop(self, connector):
        """Test connector can start and stop."""
        await connector.start()
        assert connector._running is True

        await connector.stop()
        assert connector._running is False

    @pytest.mark.asyncio
    async def test_submit_expert_feedback(self, connector, mock_al_system):
        """Test submitting expert feedback."""
        # Create mock task
        mock_task = Mock(spec=AnnotationTask)
        mock_task.task_id = "task_123"
        mock_task.case_data = Mock(spec=CaseForReview)
        mock_task.case_data.case_id = "case_123"

        mock_al_system.get_annotation_queue.return_value = [mock_task]
        mock_al_system.receive_expert_feedback = Mock(return_value=True)

        # Submit feedback
        success = await connector.submit_expert_feedback(
            task_id="task_123",
            expert_id="expert_1",
            diagnosis="malignant",
            confidence=0.95,
            annotation_time=120.0,
        )

        assert success is True
        mock_al_system.receive_expert_feedback.assert_called_once()


class TestPACSConnector:
    """Test PACS connector."""

    @pytest.fixture
    def mock_pacs_adapter(self):
        """Create mock PACS adapter."""
        adapter = Mock(spec=PACSAdapter)
        adapter.retrieve_study = Mock(
            return_value=OperationResult.success_result(
                operation_id="retrieve_1", message="Success"
            )
        )
        adapter.query_studies = Mock(
            return_value=(
                [],
                OperationResult.success_result(operation_id="query_1", message="Success"),
            )
        )
        adapter.get_adapter_statistics = Mock(return_value={"endpoints_configured": 1})
        return adapter

    @pytest.fixture
    def connector(self, mock_pacs_adapter, tmp_path):
        """Create PACS connector."""
        return PACSConnector(mock_pacs_adapter, cache_directory=str(tmp_path))

    @pytest.mark.asyncio
    async def test_retrieve_slide_for_annotation(self, connector, tmp_path):
        """Test retrieving slide from PACS."""
        # Create mock DICOM file
        study_dir = tmp_path / "study_123"
        study_dir.mkdir()
        slide_file = study_dir / "slide_123.dcm"
        slide_file.write_text("mock dicom data")

        # Retrieve slide
        slide_info = await connector.retrieve_slide_for_annotation(
            study_uid="study_123", slide_id="slide_123"
        )

        assert slide_info is not None
        assert slide_info.slide_id == "slide_123"

    @pytest.mark.asyncio
    async def test_query_studies_for_patient(self, connector):
        """Test querying studies for patient."""
        studies = await connector.query_studies_for_patient("patient_123")

        assert isinstance(studies, list)
        connector.pacs_adapter.query_studies.assert_called_once()

    def test_get_statistics(self, connector):
        """Test getting connector statistics."""
        stats = connector.get_statistics()

        assert "cached_slides" in stats
        assert "cache_directory" in stats
        assert "pacs_adapter" in stats


class TestNotificationService:
    """Test notification service."""

    @pytest.fixture
    def service(self):
        """Create notification service."""
        return NotificationService()

    @pytest.mark.asyncio
    async def test_notify_new_annotation_task(self, service):
        """Test notifying about new annotation task."""
        # Mock email sending
        service._send_email_notification = AsyncMock()
        service._send_webhook_notification = AsyncMock()

        await service.notify_new_annotation_task(
            expert_id="expert_1",
            task_id="task_123",
            slide_id="slide_123",
            priority=0.8,
            uncertainty_score=0.9,
        )

        # Check notification was recorded
        assert len(service.notification_history) > 0

    @pytest.mark.asyncio
    async def test_notify_urgent_case(self, service):
        """Test notifying about urgent case."""
        service._send_email_notification = AsyncMock()
        service._send_webhook_notification = AsyncMock()

        await service.notify_urgent_case(
            expert_ids=["expert_1", "expert_2"],
            task_id="task_123",
            slide_id="slide_123",
            reason="Very high uncertainty",
        )

        # Check notifications were sent to both experts
        assert service._send_email_notification.call_count == 2

    def test_register_webhook(self, service):
        """Test registering webhook."""
        service.register_webhook("expert_1", "https://example.com/webhook")

        assert "expert_1" in service.webhooks
        assert "https://example.com/webhook" in service.webhooks["expert_1"]

    def test_determine_notification_priority(self, service):
        """Test determining notification priority."""
        # Urgent case
        priority = service._determine_notification_priority(0.9, 0.95)
        assert priority == NotificationPriority.URGENT

        # High priority case
        priority = service._determine_notification_priority(0.7, 0.86)
        assert priority == NotificationPriority.HIGH

        # Normal case
        priority = service._determine_notification_priority(0.5, 0.7)
        assert priority == NotificationPriority.NORMAL

        # Low priority case
        priority = service._determine_notification_priority(0.3, 0.6)
        assert priority == NotificationPriority.LOW


class TestClinicalWorkflowIntegrator:
    """Test clinical workflow integrator."""

    @pytest.fixture
    def mock_al_system(self):
        """Create mock active learning system."""
        al_system = Mock(spec=ActiveLearningSystem)
        al_system.get_annotation_queue = Mock(return_value=[])
        al_system.get_statistics = Mock(return_value={"annotations_received": 10})
        al_system.min_annotations_for_retraining = 50
        return al_system

    @pytest.fixture
    def mock_pacs_adapter(self):
        """Create mock PACS adapter."""
        adapter = Mock(spec=PACSAdapter)
        adapter.test_connection = Mock(
            return_value=OperationResult.success_result(
                operation_id="test_1", message="Connection successful"
            )
        )
        adapter.get_adapter_statistics = Mock(return_value={})
        return adapter

    @pytest.fixture
    def mock_notification_service(self):
        """Create mock notification service."""
        service = Mock(spec=NotificationService)
        service.get_statistics = Mock(return_value={"total_notifications_sent": 5})
        return service

    @pytest.fixture
    def integrator(self, mock_al_system, mock_pacs_adapter, mock_notification_service):
        """Create clinical workflow integrator."""
        return ClinicalWorkflowIntegrator(
            active_learning_system=mock_al_system,
            pacs_adapter=mock_pacs_adapter,
            notification_service=mock_notification_service,
            auto_start=False,
        )

    @pytest.mark.asyncio
    async def test_integrator_start_stop(self, integrator):
        """Test integrator can start and stop."""
        await integrator.start()
        assert integrator._running is True

        await integrator.stop()
        assert integrator._running is False

    @pytest.mark.asyncio
    async def test_handle_annotation_completion(self, integrator):
        """Test handling annotation completion."""
        integrator.al_connector.submit_expert_feedback = AsyncMock(return_value=True)

        success = await integrator.handle_annotation_completion(
            task_id="task_123",
            expert_id="expert_1",
            diagnosis="malignant",
            confidence=0.95,
            annotation_time=120.0,
        )

        assert success is True

    def test_get_workflow_statistics(self, integrator):
        """Test getting workflow statistics."""
        stats = integrator.get_workflow_statistics()

        assert "status" in stats
        assert "active_learning" in stats
        assert "pacs" in stats
        assert "notifications" in stats

    def test_register_expert_webhook(self, integrator):
        """Test registering expert webhook."""
        integrator.register_expert_webhook("expert_1", "https://example.com/webhook")

        integrator.notification_service.register_webhook.assert_called_once_with(
            "expert_1", "https://example.com/webhook"
        )

    @pytest.mark.asyncio
    async def test_test_workflow_integration(self, integrator):
        """Test workflow integration testing."""
        results = await integrator.test_workflow_integration()

        assert "timestamp" in results
        assert "tests" in results
        assert "overall_success" in results
        assert "pacs_connection" in results["tests"]
        assert "active_learning" in results["tests"]
        assert "notifications" in results["tests"]


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, tmp_path):
        """Test complete workflow from case identification to annotation."""
        # Create mock components
        mock_al_system = Mock(spec=ActiveLearningSystem)
        mock_al_system.get_annotation_queue = Mock(return_value=[])
        mock_al_system.get_statistics = Mock(return_value={"annotations_received": 10})
        mock_al_system.min_annotations_for_retraining = 50

        mock_pacs_adapter = Mock(spec=PACSAdapter)
        mock_pacs_adapter.retrieve_study = Mock(
            return_value=OperationResult.success_result(
                operation_id="retrieve_1", message="Success"
            )
        )
        mock_pacs_adapter.test_connection = Mock(
            return_value=OperationResult.success_result(operation_id="test_1", message="Success")
        )
        mock_pacs_adapter.get_adapter_statistics = Mock(return_value={})

        mock_notification_service = Mock(spec=NotificationService)
        mock_notification_service.notify_new_annotation_task = AsyncMock()
        mock_notification_service.get_statistics = Mock(return_value={})

        # Create integrator
        integrator = ClinicalWorkflowIntegrator(
            active_learning_system=mock_al_system,
            pacs_adapter=mock_pacs_adapter,
            notification_service=mock_notification_service,
            auto_start=False,
        )

        # Create mock slide file
        study_dir = tmp_path / "study_123"
        study_dir.mkdir()
        slide_file = study_dir / "slide_123.dcm"
        slide_file.write_text("mock dicom data")

        # Update cache directory
        integrator.pacs_connector.cache_directory = tmp_path

        # Process new case
        result = await integrator.process_new_case(
            study_uid="study_123",
            slide_id="slide_123",
            patient_id="patient_123",
            priority=0.8,
            expert_id="expert_1",
        )

        # Verify workflow steps
        assert result["success"] is True
        assert "pacs_retrieval" in result["steps_completed"]
        assert "ai_analysis" in result["steps_completed"]
        assert "processing_time" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
