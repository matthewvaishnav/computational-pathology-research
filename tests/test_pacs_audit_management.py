"""Property-based tests for PACS Audit Management.

Feature: pacs-integration-system
Property 47: Configurable Retention Period Support
Property 48: Audit Search and Reporting Accuracy
"""

import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from src.clinical.pacs.audit_logger import (
    AuditMessage,
    AuditParticipant,
    AuditSearchIndex,
    AuditStudyObject,
    LogRetentionManager,
    PACSAuditLogger,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_test_audit_message(
    event_type: str = "DICOM_QUERY",
    event_datetime: datetime = None,
    patient_id: str = "PAT001",
    user_id: str = "USER001",
    outcome: int = 0,
) -> AuditMessage:
    """Create test audit message."""
    if event_datetime is None:
        event_datetime = datetime.utcnow()

    return AuditMessage(
        message_id=f"msg_{event_datetime.timestamp()}",
        event_id="110112",
        event_action="R",
        event_outcome=outcome,
        event_datetime=event_datetime,
        event_type=event_type,
        participants=[
            AuditParticipant(
                user_id=user_id,
                user_name=user_id,
                user_role="Pathologist",
                is_requestor=True,
            )
        ],
        study_objects=[
            AuditStudyObject(
                study_instance_uid="1.2.3.4.5",
                patient_id=patient_id,
                patient_name="Test Patient",
            )
        ],
        description=f"Test {event_type} event",
        phi_accessed=True,
        phi_fields=["PatientID", "PatientName"],
    )


# ---------------------------------------------------------------------------
# Property 47 — Configurable Retention Period Support
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 47: Configurable Retention Period Support
# For any configured retention period (1-10 years), audit logs SHALL be
# retained for the specified duration.


@given(retention_years=st.integers(min_value=1, max_value=10))
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_47_retention_period_configurable(retention_years):
    """Retention period must be configurable between 1-10 years."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)

        # Create retention manager with specified period
        retention_mgr = LogRetentionManager(
            storage_path=storage_path,
            retention_years=retention_years,
        )

        # Verify retention period set correctly
        assert retention_mgr.retention_years == retention_years


def test_property_47_retention_period_validation():
    """Retention period must be validated (1-10 years)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)

        # Invalid: too low
        with pytest.raises(ValueError, match="between 1 and 10"):
            LogRetentionManager(storage_path=storage_path, retention_years=0)

        # Invalid: too high
        with pytest.raises(ValueError, match="between 1 and 10"):
            LogRetentionManager(storage_path=storage_path, retention_years=11)

        # Valid: boundary values
        LogRetentionManager(storage_path=storage_path, retention_years=1)
        LogRetentionManager(storage_path=storage_path, retention_years=10)


def test_property_47_entries_retained_within_period():
    """Entries within retention period must not be deleted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)

        # Create logger with 2-year retention
        logger = PACSAuditLogger(
            storage_path=storage_path,
            retention_years=2,
        )

        # Create entry from 1 year ago (within retention)
        old_date = datetime.utcnow() - timedelta(days=365)
        message = _make_test_audit_message(event_datetime=old_date)

        # Log message
        logger._log_message(message)

        # Check retention manager
        retention_mgr = logger._retention
        entry_date = old_date.date()

        # Should NOT be marked for deletion
        assert not retention_mgr.should_delete(entry_date)


def test_property_47_entries_deleted_after_period():
    """Entries beyond retention period must be marked for deletion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)

        # Create retention manager with 2-year retention
        retention_mgr = LogRetentionManager(
            storage_path=storage_path,
            retention_years=2,
        )

        # Entry from 3 years ago (beyond retention)
        old_date = date.today() - timedelta(days=3 * 365 + 1)

        # Should be marked for deletion
        assert retention_mgr.should_delete(old_date)


@given(retention_years=st.integers(min_value=1, max_value=10))
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_47_retention_status_accurate(retention_years):
    """Retention status must accurately reflect configured period."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)

        logger = PACSAuditLogger(
            storage_path=storage_path,
            retention_years=retention_years,
        )

        # Get retention status
        status = logger._retention.get_retention_status()

        # Must reflect configured period
        assert status["retention_years"] == retention_years


def test_property_47_archive_old_entries():
    """Old entries must be archivable after 1 year."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "entries"
        archive_path = Path(tmpdir) / "archive"

        # Create retention manager
        retention_mgr = LogRetentionManager(
            storage_path=storage_path,
            retention_years=7,
        )

        # Create entry from 2 years ago
        old_date = datetime.utcnow() - timedelta(days=2 * 365)
        day_dir = storage_path / old_date.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        # Create test file
        test_file = day_dir / "test_entry.json"
        test_file.write_text('{"test": "data"}')

        # Archive old entries
        moved = retention_mgr.archive_old_entries(archive_path)

        # Entry should be archived
        assert moved == 1
        assert (archive_path / old_date.strftime("%Y%m%d") / "test_entry.json").exists()
        assert not test_file.exists()


def test_property_47_delete_expired_entries():
    """Expired entries must be deletable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "entries"

        # Create retention manager with 2-year retention
        retention_mgr = LogRetentionManager(
            storage_path=storage_path,
            retention_years=2,
        )

        # Create entry from 3 years ago (expired)
        old_date = datetime.utcnow() - timedelta(days=3 * 365 + 1)
        day_dir = storage_path / old_date.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        # Create test file
        test_file = day_dir / "expired_entry.json"
        test_file.write_text('{"test": "data"}')

        # Delete expired entries
        deleted = retention_mgr.delete_expired_entries()

        # Entry should be deleted
        assert deleted == 1
        assert not test_file.exists()


# ---------------------------------------------------------------------------
# Property 48 — Audit Search and Reporting Accuracy
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 48: Audit Search and Reporting Accuracy
# For any audit log search or report request, results SHALL accurately reflect
# the queried criteria and time ranges.


def test_property_48_search_by_date_range():
    """Search must accurately filter by date range."""
    index = AuditSearchIndex()

    # Create messages across different dates
    base_date = datetime(2026, 1, 1, 12, 0, 0)

    messages = [
        _make_test_audit_message(event_datetime=base_date + timedelta(days=i)) for i in range(10)
    ]

    # Add to index
    for msg in messages:
        index.add_entry(msg, f"path/{msg.message_id}.json")

    # Search for middle 5 days
    start_date = base_date + timedelta(days=2)
    end_date = base_date + timedelta(days=6)

    results = index.search(start_date=start_date, end_date=end_date, limit=0)

    # Should return exactly 5 entries (days 2-6 inclusive)
    assert len(results) == 5

    # Verify all results within range
    for result in results:
        result_date = datetime.fromisoformat(result["event_datetime"])
        assert start_date <= result_date <= end_date


@given(
    event_types=st.lists(
        st.sampled_from(["DICOM_QUERY", "DICOM_RETRIEVE", "DICOM_STORE", "PHI_ACCESS"]),
        min_size=5,
        max_size=20,
    )
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_48_search_by_event_type(event_types):
    """Search must accurately filter by event type."""
    index = AuditSearchIndex()

    # Create messages with different event types
    for i, event_type in enumerate(event_types):
        msg = _make_test_audit_message(
            event_type=event_type,
            event_datetime=datetime.utcnow() + timedelta(seconds=i),
        )
        index.add_entry(msg, f"path/{msg.message_id}.json")

    # Search for specific event type
    target_type = "DICOM_QUERY"
    results = index.search(event_type=target_type, limit=0)

    # All results must match target type
    expected_count = event_types.count(target_type)
    assert len(results) == expected_count

    for result in results:
        assert result["event_type"] == target_type


def test_property_48_search_by_patient_id():
    """Search must accurately filter by patient ID."""
    index = AuditSearchIndex()

    # Create messages for different patients
    patient_ids = ["PAT001", "PAT002", "PAT003", "PAT001", "PAT002"]

    for i, patient_id in enumerate(patient_ids):
        msg = _make_test_audit_message(
            patient_id=patient_id,
            event_datetime=datetime.utcnow() + timedelta(seconds=i),
        )
        index.add_entry(msg, f"path/{msg.message_id}.json")

    # Search for specific patient
    results = index.search(patient_id="PAT001", limit=0)

    # Should return 2 entries for PAT001
    assert len(results) == 2

    for result in results:
        assert "PAT001" in result["patient_ids"]


def test_property_48_search_by_user_id():
    """Search must accurately filter by user ID."""
    index = AuditSearchIndex()

    # Create messages from different users
    user_ids = ["USER001", "USER002", "USER003", "USER001"]

    for i, user_id in enumerate(user_ids):
        msg = _make_test_audit_message(
            user_id=user_id,
            event_datetime=datetime.utcnow() + timedelta(seconds=i),
        )
        index.add_entry(msg, f"path/{msg.message_id}.json")

    # Search for specific user
    results = index.search(user_id="USER001", limit=0)

    # Should return 2 entries for USER001
    assert len(results) == 2

    for result in results:
        assert "USER001" in result["user_ids"]


def test_property_48_search_by_outcome():
    """Search must accurately filter by outcome."""
    index = AuditSearchIndex()

    # Create messages with different outcomes
    outcomes = [0, 0, 4, 0, 8, 0]  # 0=success, 4=warning, 8=failure

    for i, outcome in enumerate(outcomes):
        msg = _make_test_audit_message(
            outcome=outcome,
            event_datetime=datetime.utcnow() + timedelta(seconds=i),
        )
        index.add_entry(msg, f"path/{msg.message_id}.json")

    # Search for failures only
    results = index.search(outcome=8, limit=0)

    # Should return 1 failure
    assert len(results) == 1
    assert results[0]["event_outcome"] == 8


def test_property_48_search_limit_respected():
    """Search limit must be respected."""
    index = AuditSearchIndex()

    # Create 20 messages
    for i in range(20):
        msg = _make_test_audit_message(event_datetime=datetime.utcnow() + timedelta(seconds=i))
        index.add_entry(msg, f"path/{msg.message_id}.json")

    # Search with limit=5
    results = index.search(limit=5)

    # Should return exactly 5 results
    assert len(results) == 5


def test_property_48_search_returns_all_when_limit_zero():
    """Search with limit=0 must return all results."""
    index = AuditSearchIndex()

    # Create 15 messages
    for i in range(15):
        msg = _make_test_audit_message(event_datetime=datetime.utcnow() + timedelta(seconds=i))
        index.add_entry(msg, f"path/{msg.message_id}.json")

    # Search with limit=0
    results = index.search(limit=0)

    # Should return all 15 results
    assert len(results) == 15


def test_property_48_report_summary_accurate():
    """Summary report must accurately aggregate statistics."""
    index = AuditSearchIndex()

    # Create diverse set of messages
    messages = [
        _make_test_audit_message(event_type="DICOM_QUERY", user_id="USER001", outcome=0),
        _make_test_audit_message(event_type="DICOM_QUERY", user_id="USER001", outcome=0),
        _make_test_audit_message(event_type="DICOM_RETRIEVE", user_id="USER002", outcome=0),
        _make_test_audit_message(event_type="DICOM_STORE", user_id="USER001", outcome=4),
        _make_test_audit_message(event_type="DICOM_QUERY", user_id="USER003", outcome=8),
    ]

    for i, msg in enumerate(messages):
        msg.event_datetime = datetime.utcnow() + timedelta(seconds=i)
        index.add_entry(msg, f"path/{msg.message_id}.json")

    # Generate summary report
    start_date = datetime.utcnow() - timedelta(hours=1)
    end_date = datetime.utcnow() + timedelta(hours=1)

    report = index.generate_report(start_date, end_date, report_type="summary")

    # Verify aggregations
    assert report["total_events"] == 5
    assert report["by_event_type"]["DICOM_QUERY"] == 3
    assert report["by_event_type"]["DICOM_RETRIEVE"] == 1
    assert report["by_event_type"]["DICOM_STORE"] == 1
    assert report["by_outcome"][0] == 3  # Success
    assert report["by_outcome"][4] == 1  # Warning
    assert report["by_outcome"][8] == 1  # Failure
    assert report["by_user"]["USER001"] == 3
    assert report["by_user"]["USER002"] == 1
    assert report["by_user"]["USER003"] == 1


def test_property_48_report_phi_access_accurate():
    """PHI access report must accurately filter PHI events."""
    index = AuditSearchIndex()

    # Create mix of PHI and non-PHI events
    messages = [
        _make_test_audit_message(event_type="DICOM_QUERY"),  # PHI
        _make_test_audit_message(event_type="DICOM_RETRIEVE"),  # PHI
        _make_test_audit_message(event_type="SYSTEM_EVENT"),  # Non-PHI
    ]

    # Set PHI flags
    messages[0].phi_accessed = True
    messages[1].phi_accessed = True
    messages[2].phi_accessed = False

    for i, msg in enumerate(messages):
        msg.event_datetime = datetime.utcnow() + timedelta(seconds=i)
        index.add_entry(msg, f"path/{msg.message_id}.json")

    # Generate PHI access report
    start_date = datetime.utcnow() - timedelta(hours=1)
    end_date = datetime.utcnow() + timedelta(hours=1)

    report = index.generate_report(start_date, end_date, report_type="phi_access")

    # Should only include PHI events
    assert report["total_phi_events"] == 2
    assert len(report["entries"]) == 2


def test_property_48_report_failures_accurate():
    """Failures report must accurately filter failed events."""
    index = AuditSearchIndex()

    # Create mix of success and failure events
    messages = [
        _make_test_audit_message(outcome=0),  # Success
        _make_test_audit_message(outcome=4),  # Warning (failure)
        _make_test_audit_message(outcome=0),  # Success
        _make_test_audit_message(outcome=8),  # Failure
    ]

    for i, msg in enumerate(messages):
        msg.event_datetime = datetime.utcnow() + timedelta(seconds=i)
        index.add_entry(msg, f"path/{msg.message_id}.json")

    # Generate failures report
    start_date = datetime.utcnow() - timedelta(hours=1)
    end_date = datetime.utcnow() + timedelta(hours=1)

    report = index.generate_report(start_date, end_date, report_type="failures")

    # Should only include non-zero outcomes
    assert report["total_failures"] == 2
    assert len(report["entries"]) == 2


@given(
    start_offset=st.integers(min_value=0, max_value=10),
    end_offset=st.integers(min_value=11, max_value=20),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_48_date_range_boundaries_accurate(start_offset, end_offset):
    """Date range boundaries must be accurately enforced."""
    index = AuditSearchIndex()

    # Create messages across 30 days
    base_date = datetime(2026, 1, 1, 12, 0, 0)

    for i in range(30):
        msg = _make_test_audit_message(event_datetime=base_date + timedelta(days=i))
        index.add_entry(msg, f"path/{msg.message_id}.json")

    # Search with specific range
    start_date = base_date + timedelta(days=start_offset)
    end_date = base_date + timedelta(days=end_offset)

    results = index.search(start_date=start_date, end_date=end_date, limit=0)

    # Verify all results within boundaries
    for result in results:
        result_date = datetime.fromisoformat(result["event_datetime"])
        assert start_date <= result_date <= end_date


# ---------------------------------------------------------------------------
# Additional Unit Tests
# ---------------------------------------------------------------------------


def test_retention_manager_initialization():
    """Retention manager must initialize correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)

        retention_mgr = LogRetentionManager(
            storage_path=storage_path,
            retention_years=7,
        )

        assert retention_mgr.storage_path == storage_path
        assert retention_mgr.retention_years == 7


def test_retention_manager_should_archive():
    """Archive check must correctly identify old entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)
        retention_mgr = LogRetentionManager(storage_path=storage_path, retention_years=7)

        # Recent entry (should not archive)
        recent_date = date.today() - timedelta(days=180)
        assert not retention_mgr.should_archive(recent_date)

        # Old entry (should archive)
        old_date = date.today() - timedelta(days=400)
        assert retention_mgr.should_archive(old_date)


def test_retention_status_structure():
    """Retention status must contain all required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)

        logger = PACSAuditLogger(storage_path=storage_path, retention_years=7)

        status = logger._retention.get_retention_status()

        # Verify structure
        assert "total_entries" in status
        assert "archivable_count" in status
        assert "expired_count" in status
        assert "oldest_entry_date" in status
        assert "retention_years" in status


def test_search_index_initialization():
    """Search index must initialize correctly."""
    index = AuditSearchIndex()

    assert index._entries == []


def test_search_index_add_entry():
    """Entries must be addable to search index."""
    index = AuditSearchIndex()

    msg = _make_test_audit_message()
    index.add_entry(msg, "path/test.json")

    # Entry should be in index
    assert len(index._entries) == 1


def test_search_results_sorted_by_date():
    """Search results must be sorted by date (newest first)."""
    index = AuditSearchIndex()

    # Create messages in random order
    base_date = datetime(2026, 1, 1, 12, 0, 0)
    dates = [base_date + timedelta(days=i) for i in [5, 2, 8, 1, 9]]

    for dt in dates:
        msg = _make_test_audit_message(event_datetime=dt)
        index.add_entry(msg, f"path/{msg.message_id}.json")

    # Search all
    results = index.search(limit=0)

    # Should be sorted newest first
    result_dates = [datetime.fromisoformat(r["event_datetime"]) for r in results]
    assert result_dates == sorted(result_dates, reverse=True)


def test_compliance_report_metadata():
    """Compliance reports must include metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)

        logger = PACSAuditLogger(storage_path=storage_path)

        # Generate report
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()

        report = logger.generate_compliance_report(
            start_date=start_date,
            end_date=end_date,
            report_type="summary",
        )

        # Must include metadata
        assert "metadata" in report
        assert "generated_at" in report["metadata"]
        assert "start_date" in report["metadata"]
        assert "end_date" in report["metadata"]
        assert "source_system" in report["metadata"]


def test_invalid_report_type_raises_error():
    """Invalid report type must raise error."""
    index = AuditSearchIndex()

    start_date = datetime.utcnow() - timedelta(days=7)
    end_date = datetime.utcnow()

    with pytest.raises(ValueError, match="Unknown report_type"):
        index.generate_report(start_date, end_date, report_type="invalid")
