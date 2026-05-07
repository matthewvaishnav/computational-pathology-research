"""Property-based tests for PACS Workflow Orchestrator.

Feature: pacs-integration-system
Property 28: Automatic Study Queuing
Property 29: Workflow Operation Sequencing
Property 30: Status Tracking Completeness
Property 31: Priority-Based Processing Order
"""

import time
from datetime import datetime, timedelta
from typing import List
from unittest.mock import Mock, patch

import pytest
import torch

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from src.clinical.pacs.data_models import (
    DicomPriority,
    OperationResult,
    StudyInfo,
)
from src.clinical.pacs.workflow_orchestrator import WorkflowOrchestrator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_test_study(
    study_uid: str = "1.2.3.4.5",
    patient_id: str = "PAT001",
    priority: DicomPriority = DicomPriority.MEDIUM,
) -> StudyInfo:
    """Create test study info."""
    return StudyInfo(
        study_instance_uid=study_uid,
        patient_id=patient_id,
        patient_name="Test Patient",
        study_date=datetime.now(),
        study_description="Test WSI Study",
        modality="SM",
        series_count=1,
        priority=priority,
    )


def _make_mock_pacs_adapter():
    """Create mock PACS adapter."""
    mock_adapter = Mock()

    # Mock query_studies
    mock_adapter.query_studies.return_value = (
        [],
        OperationResult.success_result(operation_id="query_1", message="Query successful"),
    )

    # Mock retrieve_study
    mock_adapter.retrieve_study.return_value = OperationResult.success_result(
        operation_id="retrieve_1",
        message="Retrieval successful",
        data={"retrieved_files": ["/path/to/file1.dcm"]},
    )

    # Mock store_analysis_results
    mock_adapter.store_analysis_results.return_value = OperationResult.success_result(
        operation_id="store_1", message="Storage successful"
    )

    return mock_adapter


def _make_mock_clinical_workflow():
    """Create mock clinical workflow system."""
    mock_workflow = Mock()

    # Mock process_case
    mock_workflow.process_case.return_value = {
        "primary_diagnosis": {"disease_name": "Normal", "probability": 0.95},
        "model_version": "1.0.0",
        "probability_distribution": {"Normal": 0.95, "Abnormal": 0.05},
    }

    return mock_workflow


# ---------------------------------------------------------------------------
# Property 28 — Automatic Study Queuing
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 28: Automatic Study Queuing
# For any newly detected WSI studies, they SHALL be automatically queued for
# retrieval and processing without manual intervention.


def test_property_28_new_studies_automatically_queued():
    """New studies must be automatically queued without manual intervention."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
        poll_interval=timedelta(seconds=1),
    )

    # Create new studies
    studies = [_make_test_study(study_uid=f"1.2.3.{i}", patient_id=f"PAT{i:03d}") for i in range(3)]

    # Process studies (simulates automatic queuing)
    results = orchestrator.process_new_studies(studies)

    # All studies must be queued and processed
    assert len(results) == 3
    assert all(r.success for r in results)

    # Verify studies tracked as processed
    assert len(orchestrator._processed_studies) == 3


@given(study_count=st.integers(min_value=1, max_value=10))
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_28_multiple_studies_queued_automatically(study_count):
    """Multiple new studies must be automatically queued."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    # Create multiple studies
    studies = [
        _make_test_study(study_uid=f"1.2.3.{i}", patient_id=f"PAT{i:03d}")
        for i in range(study_count)
    ]

    # Process studies
    results = orchestrator.process_new_studies(studies)

    # All must be queued
    assert len(results) == study_count


def test_property_28_no_manual_intervention_required():
    """Studies must be processed without manual intervention."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study = _make_test_study()

    # Process without any manual steps
    results = orchestrator.process_new_studies([study])

    # Must complete automatically
    assert len(results) == 1
    assert results[0].success


def test_property_28_already_processed_studies_skipped():
    """Already processed studies must not be requeued."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study = _make_test_study(study_uid="1.2.3.4.5")

    # Process first time
    results1 = orchestrator.process_new_studies([study])
    assert len(results1) == 1

    # Process again (should skip)
    results2 = orchestrator.process_new_studies([study], force_reprocess=False)
    assert len(results2) == 0  # Skipped


# ---------------------------------------------------------------------------
# Property 29 — Workflow Operation Sequencing
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 29: Workflow Operation Sequencing
# For any workflow execution, operations SHALL occur in the correct sequence:
# query, retrieve, analyze, store.


def test_property_29_operations_execute_in_correct_sequence():
    """Operations must execute in sequence: query, retrieve, analyze, store."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study = _make_test_study()

    # Track call order
    call_order = []

    def track_retrieve(*args, **kwargs):
        call_order.append("retrieve")
        return OperationResult.success_result(
            operation_id="retrieve_1",
            message="Retrieved",
            data={"retrieved_files": ["/path/file.dcm"]},
        )

    def track_process(*args, **kwargs):
        call_order.append("analyze")
        return {
            "primary_diagnosis": {"disease_name": "Normal", "probability": 0.95},
            "model_version": "1.0.0",
            "probability_distribution": {},
        }

    def track_store(*args, **kwargs):
        call_order.append("store")
        return OperationResult.success_result(operation_id="store_1", message="Stored")

    mock_pacs.retrieve_study.side_effect = track_retrieve
    mock_workflow.process_case.side_effect = track_process
    mock_pacs.store_analysis_results.side_effect = track_store

    # Process study
    results = orchestrator.process_new_studies([study])

    # Verify sequence
    assert call_order == ["retrieve", "analyze", "store"]
    assert results[0].success


def test_property_29_retrieve_before_analyze():
    """Retrieve must occur before analyze."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study = _make_test_study()

    retrieve_time = None
    analyze_time = None

    def track_retrieve(*args, **kwargs):
        nonlocal retrieve_time
        retrieve_time = time.time()
        return OperationResult.success_result(
            operation_id="retrieve_1",
            message="Retrieved",
            data={"retrieved_files": ["/path/file.dcm"]},
        )

    def track_analyze(*args, **kwargs):
        nonlocal analyze_time
        analyze_time = time.time()
        return {
            "primary_diagnosis": {"disease_name": "Normal", "probability": 0.95},
            "model_version": "1.0.0",
            "probability_distribution": {},
        }

    mock_pacs.retrieve_study.side_effect = track_retrieve
    mock_workflow.process_case.side_effect = track_analyze

    # Process
    orchestrator.process_new_studies([study])

    # Retrieve must happen before analyze
    assert retrieve_time is not None
    assert analyze_time is not None
    assert retrieve_time < analyze_time


def test_property_29_analyze_before_store():
    """Analyze must occur before store."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study = _make_test_study()

    analyze_time = None
    store_time = None

    def track_analyze(*args, **kwargs):
        nonlocal analyze_time
        analyze_time = time.time()
        return {
            "primary_diagnosis": {"disease_name": "Normal", "probability": 0.95},
            "model_version": "1.0.0",
            "probability_distribution": {},
        }

    def track_store(*args, **kwargs):
        nonlocal store_time
        store_time = time.time()
        return OperationResult.success_result(operation_id="store_1", message="Stored")

    mock_workflow.process_case.side_effect = track_analyze
    mock_pacs.store_analysis_results.side_effect = track_store

    # Process
    orchestrator.process_new_studies([study])

    # Analyze must happen before store
    assert analyze_time is not None
    assert store_time is not None
    assert analyze_time < store_time


def test_property_29_failure_stops_sequence():
    """Failure in sequence must stop subsequent operations."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    # Make retrieval fail
    mock_pacs.retrieve_study.return_value = OperationResult.error_result(
        operation_id="retrieve_1",
        message="Retrieval failed",
        errors=["Network error"],
    )

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study = _make_test_study()

    # Process
    results = orchestrator.process_new_studies([study])

    # Must fail
    assert len(results) == 1
    assert not results[0].success

    # Analyze and store must not be called
    mock_workflow.process_case.assert_not_called()
    mock_pacs.store_analysis_results.assert_not_called()


# ---------------------------------------------------------------------------
# Property 30 — Status Tracking Completeness
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 30: Status Tracking Completeness
# For any processing completion, study status updates SHALL be recorded in the
# local database with accurate state information.


def test_property_30_successful_processing_tracked():
    """Successful processing must be tracked with accurate state."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study = _make_test_study(study_uid="1.2.3.4.5")

    # Process
    results = orchestrator.process_new_studies([study])

    # Status must be tracked
    assert study.study_instance_uid in orchestrator._processed_studies
    assert orchestrator._stats["studies_processed"] == 1
    assert orchestrator._stats["studies_failed"] == 0


def test_property_30_failed_processing_tracked():
    """Failed processing must be tracked with error details."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    # Make retrieval fail
    mock_pacs.retrieve_study.return_value = OperationResult.error_result(
        operation_id="retrieve_1",
        message="Retrieval failed",
        errors=["Network error"],
    )

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study = _make_test_study(study_uid="1.2.3.4.5")

    # Process
    results = orchestrator.process_new_studies([study])

    # Failure must be tracked
    assert study.study_instance_uid in orchestrator._failed_studies
    assert orchestrator._stats["studies_failed"] == 1
    assert orchestrator._stats["studies_processed"] == 0

    # Error details must be recorded
    failure_info = orchestrator._failed_studies[study.study_instance_uid]
    assert "error" in failure_info
    assert "timestamp" in failure_info


@given(study_count=st.integers(min_value=1, max_value=10))
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_30_multiple_studies_tracked_accurately(study_count):
    """Multiple studies must be tracked with accurate counts."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    studies = [
        _make_test_study(study_uid=f"1.2.3.{i}", patient_id=f"PAT{i:03d}")
        for i in range(study_count)
    ]

    # Process
    orchestrator.process_new_studies(studies)

    # All must be tracked
    assert len(orchestrator._processed_studies) == study_count
    assert orchestrator._stats["studies_processed"] == study_count


def test_property_30_status_retrievable_via_api():
    """Status must be retrievable via get_processing_status."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study = _make_test_study()

    # Process
    orchestrator.process_new_studies([study])

    # Get status
    status = orchestrator.get_processing_status()

    # Must contain accurate information
    assert status["processed_studies"] == 1
    assert status["failed_studies"] == 0
    assert "statistics" in status
    assert status["statistics"]["studies_processed"] == 1


# ---------------------------------------------------------------------------
# Property 31 — Priority-Based Processing Order
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 31: Priority-Based Processing Order
# For any set of studies with different DICOM priority tags, urgent studies
# SHALL be processed before lower-priority studies.


def test_property_31_urgent_processed_before_low():
    """Urgent studies must be processed before low priority studies."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    # Create studies with different priorities
    studies = [
        _make_test_study(study_uid="1.2.3.1", priority=DicomPriority.LOW),
        _make_test_study(study_uid="1.2.3.2", priority=DicomPriority.URGENT),
        _make_test_study(study_uid="1.2.3.3", priority=DicomPriority.MEDIUM),
    ]

    # Track processing order
    processing_order = []

    def track_retrieve(study_instance_uid, **kwargs):
        processing_order.append(study_instance_uid)
        return OperationResult.success_result(
            operation_id=f"retrieve_{study_instance_uid}",
            message="Retrieved",
            data={"retrieved_files": ["/path/file.dcm"]},
        )

    mock_pacs.retrieve_study.side_effect = track_retrieve

    # Process
    orchestrator.process_new_studies(studies)

    # Urgent must be first
    assert processing_order[0] == "1.2.3.2"  # URGENT
    assert processing_order[1] == "1.2.3.3"  # MEDIUM
    assert processing_order[2] == "1.2.3.1"  # LOW


@given(
    priorities=st.lists(
        st.sampled_from(
            [DicomPriority.LOW, DicomPriority.MEDIUM, DicomPriority.HIGH, DicomPriority.URGENT]
        ),
        min_size=3,
        max_size=10,
    )
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_31_priority_ordering_maintained(priorities):
    """Priority ordering must be maintained for any priority mix."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
        max_concurrent_studies=1,  # Sequential processing to avoid race conditions
    )

    # Create studies with given priorities
    studies = [
        _make_test_study(study_uid=f"1.2.3.{i}", priority=priority)
        for i, priority in enumerate(priorities)
    ]

    # Track processing order
    processing_order = []

    def track_retrieve(study_instance_uid, **kwargs):
        processing_order.append(study_instance_uid)
        return OperationResult.success_result(
            operation_id=f"retrieve_{study_instance_uid}",
            message="Retrieved",
            data={"retrieved_files": ["/path/file.dcm"]},
        )

    mock_pacs.retrieve_study.side_effect = track_retrieve

    # Process
    orchestrator.process_new_studies(studies)

    # Verify priority order maintained
    priority_order = {
        DicomPriority.URGENT: 0,
        DicomPriority.HIGH: 1,
        DicomPriority.MEDIUM: 2,
        DicomPriority.LOW: 3,
    }

    # Extract priorities from processing order
    processed_priorities = [priorities[int(uid.split(".")[-1])] for uid in processing_order]

    # Check that priorities are non-increasing
    for i in range(len(processed_priorities) - 1):
        current_priority = priority_order[processed_priorities[i]]
        next_priority = priority_order[processed_priorities[i + 1]]
        assert current_priority <= next_priority


def test_property_31_high_before_medium():
    """High priority studies must be processed before medium priority."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    studies = [
        _make_test_study(study_uid="1.2.3.1", priority=DicomPriority.MEDIUM),
        _make_test_study(study_uid="1.2.3.2", priority=DicomPriority.HIGH),
    ]

    processing_order = []

    def track_retrieve(study_instance_uid, **kwargs):
        processing_order.append(study_instance_uid)
        return OperationResult.success_result(
            operation_id=f"retrieve_{study_instance_uid}",
            message="Retrieved",
            data={"retrieved_files": ["/path/file.dcm"]},
        )

    mock_pacs.retrieve_study.side_effect = track_retrieve

    # Process
    orchestrator.process_new_studies(studies)

    # HIGH must be first
    assert processing_order[0] == "1.2.3.2"
    assert processing_order[1] == "1.2.3.1"


def test_property_31_medium_before_low():
    """Medium priority studies must be processed before low priority."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    studies = [
        _make_test_study(study_uid="1.2.3.1", priority=DicomPriority.LOW),
        _make_test_study(study_uid="1.2.3.2", priority=DicomPriority.MEDIUM),
    ]

    processing_order = []

    def track_retrieve(study_instance_uid, **kwargs):
        processing_order.append(study_instance_uid)
        return OperationResult.success_result(
            operation_id=f"retrieve_{study_instance_uid}",
            message="Retrieved",
            data={"retrieved_files": ["/path/file.dcm"]},
        )

    mock_pacs.retrieve_study.side_effect = track_retrieve

    # Process
    orchestrator.process_new_studies(studies)

    # MEDIUM must be first
    assert processing_order[0] == "1.2.3.2"
    assert processing_order[1] == "1.2.3.1"


# ---------------------------------------------------------------------------
# Additional Unit Tests
# ---------------------------------------------------------------------------


def test_orchestrator_initialization():
    """Orchestrator must initialize correctly."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
        poll_interval=timedelta(minutes=5),
        max_concurrent_studies=10,
    )

    assert orchestrator.pacs_adapter is mock_pacs
    assert orchestrator.clinical_workflow is mock_workflow
    assert orchestrator.poll_interval == timedelta(minutes=5)
    assert orchestrator.max_concurrent_studies == 10
    assert not orchestrator._is_running


def test_start_stop_automated_polling():
    """Automated polling must start and stop correctly."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
        poll_interval=timedelta(seconds=1),
    )

    # Start polling
    orchestrator.start_automated_polling()
    assert orchestrator._is_running
    assert orchestrator._polling_thread is not None

    # Stop polling
    orchestrator.stop_automated_polling()
    assert not orchestrator._is_running


def test_handle_processing_failure_network_error():
    """Network errors must trigger retry_later recovery action."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study_uid = "1.2.3.4.5"
    error = Exception("Connection timeout")

    recovery_action = orchestrator.handle_processing_failure(study_uid, error)

    assert recovery_action == "retry_later"
    assert study_uid in orchestrator._failed_studies


def test_handle_processing_failure_disk_error():
    """Disk errors must trigger pause_processing recovery action."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study_uid = "1.2.3.4.5"
    error = Exception("Disk space full")

    recovery_action = orchestrator.handle_processing_failure(study_uid, error)

    assert recovery_action == "pause_processing"


def test_handle_processing_failure_authentication_error():
    """Authentication errors must trigger manual_intervention recovery action."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study_uid = "1.2.3.4.5"
    error = Exception("Authentication failed")

    recovery_action = orchestrator.handle_processing_failure(study_uid, error)

    assert recovery_action == "manual_intervention"


def test_handle_processing_failure_max_retries():
    """Max retries must trigger move_to_dead_letter recovery action."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study_uid = "1.2.3.4.5"
    error = Exception("Generic error")

    # Fail 3 times
    for _ in range(3):
        orchestrator.handle_processing_failure(study_uid, error)

    # 4th failure should move to dead letter
    recovery_action = orchestrator.handle_processing_failure(study_uid, error)

    assert recovery_action == "move_to_dead_letter"


def test_get_processing_status_structure():
    """Processing status must contain all required fields."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    status = orchestrator.get_processing_status()

    # Verify structure
    assert "is_running" in status
    assert "poll_interval_minutes" in status
    assert "max_concurrent_studies" in status
    assert "active_processing" in status
    assert "queued_studies" in status
    assert "processed_studies" in status
    assert "failed_studies" in status
    assert "statistics" in status
    assert "active_studies" in status


def test_context_manager_support():
    """Orchestrator must support context manager protocol."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    with WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    ) as orchestrator:
        assert orchestrator is not None

    # Should clean up after exit
    assert not orchestrator._is_running


def test_force_reprocess_flag():
    """Force reprocess flag must allow reprocessing of completed studies."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    study = _make_test_study(study_uid="1.2.3.4.5")

    # Process first time
    results1 = orchestrator.process_new_studies([study])
    assert len(results1) == 1

    # Process again with force_reprocess=True
    results2 = orchestrator.process_new_studies([study], force_reprocess=True)
    assert len(results2) == 1  # Reprocessed


def test_concurrent_study_limit():
    """Concurrent study processing must respect max_concurrent_studies limit."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
        max_concurrent_studies=2,
    )

    # Create 5 studies
    studies = [_make_test_study(study_uid=f"1.2.3.{i}", patient_id=f"PAT{i:03d}") for i in range(5)]

    # Process
    results = orchestrator.process_new_studies(studies)

    # All should complete
    assert len(results) == 5
    assert all(r.success for r in results)


def test_priority_order_helper():
    """Priority order helper must return correct numeric values."""
    mock_pacs = _make_mock_pacs_adapter()
    mock_workflow = _make_mock_clinical_workflow()

    orchestrator = WorkflowOrchestrator(
        pacs_adapter=mock_pacs,
        clinical_workflow=mock_workflow,
    )

    # Verify priority ordering
    assert orchestrator._get_priority_order(DicomPriority.URGENT) == 0
    assert orchestrator._get_priority_order(DicomPriority.HIGH) == 1
    assert orchestrator._get_priority_order(DicomPriority.MEDIUM) == 2
    assert orchestrator._get_priority_order(DicomPriority.LOW) == 3
