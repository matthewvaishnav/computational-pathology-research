"""Tests for HIPAA-compliant PACS audit logger."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

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
    TamperEvidentStorage,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_logger(tmp_path: Path, **kwargs) -> PACSAuditLogger:
    """Create a PACSAuditLogger wired to a temp directory."""
    # Use a fixed signing key so integrity can be verified within a test session
    return PACSAuditLogger(
        storage_path=tmp_path / "audit",
        signing_key=b"test-signing-key-32bytes-padding!",
        **kwargs,
    )


def _make_endpoint(host: str = "pacs.hospital.org", ae: str = "HOSPITAL_PACS") -> SimpleNamespace:
    return SimpleNamespace(host=host, ae_title=ae, port=11112)


def _make_study(uid: str = "1.2.3.4.5") -> SimpleNamespace:
    return SimpleNamespace(
        study_instance_uid=uid,
        patient_id="P001",
        patient_name="Smith^John",
        accession_number="ACC-001",
    )


# ---------------------------------------------------------------------------
# Property 43 — DICOM Operation Audit Completeness
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 43: DICOM Operation Audit Completeness
# For any DICOM operation, comprehensive audit logs SHALL be recorded with
# timestamps, user identifiers, and operation details.


@given(
    user_id=st.text(
        min_size=1, max_size=64, alphabet=st.characters(whitelist_categories=("L", "N", "P"))
    ),
    result_count=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_43_dicom_query_always_returns_message_id(tmp_path, user_id, result_count):
    """Any C-FIND call produces a non-empty message_id findable in the index."""
    # Use a safe path component by replacing invalid characters
    safe_user_id = "".join(c if c.isalnum() else "_" for c in user_id[:8])
    audit = _make_logger(tmp_path / safe_user_id)
    endpoint = _make_endpoint()
    mid = audit.log_dicom_query(
        user_id=user_id,
        endpoint=endpoint,
        query_params={"PatientID": "P001"},
        result_count=result_count,
    )
    assert mid and isinstance(mid, str)
    hits = audit.search_logs(user_id=user_id)
    assert any(h["message_id"] == mid for h in hits)
    assert any(h["event_type"] == "DICOM_QUERY" for h in hits if h["message_id"] == mid)


# ---------------------------------------------------------------------------
# Property 44 — HIPAA Audit Message Formatting
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 44: HIPAA Audit Message Formatting
# For any patient access event, audit logs SHALL be formatted according to
# DICOM Audit Message format for HIPAA compliance.

_REQUIRED_HIPAA_KEYS = {"event_type", "event_datetime", "outcome", "participants", "study_objects"}


@given(
    patient_id=st.text(
        min_size=1, max_size=32, alphabet=st.characters(whitelist_categories=("L", "N"))
    ),
    outcome=st.sampled_from([0, 4, 8, 12]),
)
@settings(max_examples=100)
def test_property_44_hipaa_format_has_required_keys(patient_id, outcome):
    """to_hipaa_format() on any AuditMessage contains all mandatory HIPAA keys."""
    msg = AuditMessage(
        message_id="test-id",
        event_id="110103",
        event_action="R",
        event_outcome=outcome,
        event_datetime=datetime.utcnow(),
        event_type="PHI_ACCESS",
        participants=[
            AuditParticipant(
                user_id="dr.smith",
                user_name="Dr. Smith",
                user_role="Pathologist",
            )
        ],
        study_objects=[
            AuditStudyObject(
                study_instance_uid="1.2.3",
                patient_id=patient_id,
                patient_name="REDACTED",
            )
        ],
        description="Test PHI access",
    )
    hipaa = msg.to_hipaa_format()
    assert _REQUIRED_HIPAA_KEYS.issubset(
        hipaa.keys()
    ), f"Missing keys: {_REQUIRED_HIPAA_KEYS - set(hipaa.keys())}"


# ---------------------------------------------------------------------------
# Property 45 — PHI Access Detail Logging
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 45: PHI Access Detail Logging
# For any Protected Health Information access, audit logs SHALL record the
# specific data elements that were viewed or modified.


@given(
    phi_fields=st.lists(
        st.sampled_from(["PatientID", "PatientName", "DOB", "SSN", "AccessionNumber"]),
        min_size=1,
        max_size=5,
        unique=True,
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_45_phi_fields_recorded(tmp_path, phi_fields):
    """log_phi_access stores all specified phi_fields and marks phi_accessed=True."""
    audit = _make_logger(tmp_path)
    mid = audit.log_phi_access(
        user_id="dr.jones",
        patient_id="P999",
        patient_name="Jones^Alice",
        accessed_fields=phi_fields,
        reason="clinical review",
    )
    hits = [h for h in audit.search_logs() if h["message_id"] == mid]
    assert hits, "Entry not found in index"
    assert hits[0]["phi_accessed"] is True
    recorded = set(hits[0]["phi_fields"])
    assert set(phi_fields).issubset(recorded)


# ---------------------------------------------------------------------------
# Property 46 — Tamper-Evident Log Integrity
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 46: Tamper-Evident Log Integrity
# For any audit log entry, cryptographic signatures SHALL provide tamper-evident
# storage and integrity verification.


@given(entries=st.lists(st.text(min_size=1, max_size=128), min_size=1, max_size=20))
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_46_tamper_evident_verify_all_clean(tmp_path, entries):
    """All freshly written entries verify as untampered."""
    storage = TamperEvidentStorage(tmp_path / "te", signing_key=b"x" * 32)
    paths = []
    for i, text in enumerate(entries):
        entry = {
            "message_id": f"msg-{i:04d}",
            "event_datetime": datetime.utcnow().isoformat(),
            "data": text,
        }
        rel = storage.write_entry(entry)
        paths.append(storage.storage_path / rel)
    for p in paths:
        assert storage.verify_entry(p), f"Entry unexpectedly tampered: {p}"


@given(entries=st.lists(st.text(min_size=1, max_size=128), min_size=2, max_size=10))
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_46_tamper_detected_after_modification(tmp_path, entries):
    """Modifying one file's JSON data causes verify_entry to return False for that file only."""
    storage = TamperEvidentStorage(tmp_path / "te2", signing_key=b"y" * 32)
    paths = []
    for i, text in enumerate(entries):
        entry = {
            "message_id": f"msg-{i:04d}",
            "event_datetime": datetime.utcnow().isoformat(),
            "data": text,
        }
        rel = storage.write_entry(entry)
        paths.append(storage.storage_path / rel)

    # Tamper with the first file
    victim = paths[0]
    payload = json.loads(victim.read_text(encoding="utf-8"))
    payload["data"]["data"] = "TAMPERED"
    victim.write_text(json.dumps(payload), encoding="utf-8")

    assert not storage.verify_entry(victim)
    # All other entries must still pass
    for p in paths[1:]:
        assert storage.verify_entry(p)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_audit_logger_creates_storage_directory(tmp_path):
    audit = _make_logger(tmp_path)
    assert (tmp_path / "audit" / "entries").is_dir()


def test_log_dicom_query_returns_message_id(tmp_path):
    audit = _make_logger(tmp_path)
    mid = audit.log_dicom_query(
        user_id="user1",
        endpoint=_make_endpoint(),
        query_params={"PatientID": "P001", "StudyDate": "20260101"},
        result_count=5,
    )
    assert mid and isinstance(mid, str)
    assert len(mid) == 36  # UUID4 canonical form


def test_log_dicom_retrieve_phi_marked(tmp_path):
    audit = _make_logger(tmp_path)
    mid = audit.log_dicom_retrieve(
        user_id="user1",
        endpoint=_make_endpoint(),
        study_info=_make_study(),
        file_count=12,
    )
    hits = [h for h in audit.search_logs() if h["message_id"] == mid]
    assert hits[0]["phi_accessed"] is True
    assert "PatientID" in hits[0]["phi_fields"]


def test_tamper_evident_storage_write_verify(tmp_path):
    storage = TamperEvidentStorage(tmp_path / "s", signing_key=b"k" * 32)
    entry = {
        "message_id": "abc123",
        "event_datetime": datetime.utcnow().isoformat(),
        "foo": "bar",
    }
    rel = storage.write_entry(entry)
    full = storage.storage_path / rel
    assert full.exists()
    assert storage.verify_entry(full)
    data = storage.read_entry(full)
    assert data is not None
    assert data["foo"] == "bar"


def test_tamper_evident_detects_modification(tmp_path):
    storage = TamperEvidentStorage(tmp_path / "s", signing_key=b"k" * 32)
    entry = {
        "message_id": "xyz",
        "event_datetime": datetime.utcnow().isoformat(),
        "secret": "original",
    }
    rel = storage.write_entry(entry)
    full = storage.storage_path / rel

    payload = json.loads(full.read_text(encoding="utf-8"))
    payload["data"]["secret"] = "altered"
    full.write_text(json.dumps(payload), encoding="utf-8")

    assert not storage.verify_entry(full)
    assert storage.read_entry(full) is None


def test_search_by_date_range(tmp_path):
    audit = _make_logger(tmp_path)
    now = datetime.utcnow()

    # Log 3 events at different times and search with a window that includes only one
    audit.log_system_event("startup", "System started")
    mid_target = audit.log_system_event("check", "Health check")
    audit.log_system_event("shutdown", "System stopped")

    # All three should be found in a wide range
    all_hits = audit.search_logs(
        start_date=now - timedelta(seconds=5),
        end_date=now + timedelta(seconds=5),
    )
    assert len(all_hits) == 3

    # Use the target entry's datetime to build a 1-second window around it
    target_hit = next(h for h in all_hits if h["message_id"] == mid_target)
    target_dt = datetime.fromisoformat(target_hit["event_datetime"])
    narrow_hits = audit.search_logs(
        start_date=target_dt - timedelta(milliseconds=1),
        end_date=target_dt + timedelta(milliseconds=1),
    )
    assert any(h["message_id"] == mid_target for h in narrow_hits)


def test_search_by_event_type(tmp_path):
    audit = _make_logger(tmp_path)
    audit.log_system_event("startup", "System started")
    audit.log_phi_access(
        user_id="dr.x",
        patient_id="P100",
        patient_name="Doe^Jane",
        accessed_fields=["PatientName"],
        reason="review",
    )
    system_hits = audit.search_logs(event_type="SYSTEM_EVENT")
    phi_hits = audit.search_logs(event_type="PHI_ACCESS")
    assert all(h["event_type"] == "SYSTEM_EVENT" for h in system_hits)
    assert all(h["event_type"] == "PHI_ACCESS" for h in phi_hits)
    assert len(system_hits) == 1
    assert len(phi_hits) == 1


def test_search_by_patient_id(tmp_path):
    audit = _make_logger(tmp_path)
    audit.log_phi_access(
        user_id="u1",
        patient_id="PAT-AAA",
        patient_name="Alpha^A",
        accessed_fields=["PatientID"],
        reason="review",
    )
    audit.log_phi_access(
        user_id="u2",
        patient_id="PAT-BBB",
        patient_name="Beta^B",
        accessed_fields=["PatientID"],
        reason="review",
    )
    hits_aaa = audit.search_logs(patient_id="PAT-AAA")
    hits_bbb = audit.search_logs(patient_id="PAT-BBB")
    assert len(hits_aaa) == 1
    assert len(hits_bbb) == 1
    assert hits_aaa[0]["patient_ids"] == ["PAT-AAA"]


def test_generate_summary_report(tmp_path):
    audit = _make_logger(tmp_path)
    now = datetime.utcnow()
    audit.log_system_event("startup", "Started")
    audit.log_system_event("check", "Healthy")
    audit.log_dicom_query(
        user_id="u1",
        endpoint=_make_endpoint(),
        query_params={"PatientID": "P1"},
        result_count=3,
    )
    report = audit.generate_compliance_report(
        start_date=now - timedelta(seconds=5),
        end_date=now + timedelta(seconds=5),
        report_type="summary",
    )
    assert report["report_type"] == "summary"
    assert report["total_events"] == 3
    assert report["by_event_type"].get("SYSTEM_EVENT") == 2
    assert report["by_event_type"].get("DICOM_QUERY") == 1
    assert "metadata" in report


def test_phi_hashing_in_log(tmp_path):
    audit = _make_logger(tmp_path, phi_protection_enabled=True)
    raw_name = "Smith^John"
    study = _make_study()
    study.patient_name = raw_name
    audit.log_dicom_retrieve(
        user_id="u1",
        endpoint=_make_endpoint(),
        study_info=study,
        file_count=1,
    )
    # Inspect the written JSON file directly
    entry_files = list((tmp_path / "audit" / "entries").glob("????????/*.json"))
    assert entry_files
    payload = json.loads(entry_files[0].read_text(encoding="utf-8"))
    study_objects = payload["data"].get("study_objects", [])
    # Raw name must NOT appear in any stored study object
    for so in study_objects:
        assert so.get("patient_name") != raw_name, "Raw PHI found in log — hashing failed"


def test_retention_manager_archive_threshold(tmp_path):
    rm = LogRetentionManager(tmp_path / "entries", retention_years=7)
    old_date = (datetime.utcnow() - timedelta(days=400)).date()
    recent_date = (datetime.utcnow() - timedelta(days=100)).date()
    assert rm.should_archive(old_date) is True
    assert rm.should_archive(recent_date) is False


def test_retention_manager_delete_threshold(tmp_path):
    rm = LogRetentionManager(tmp_path / "entries", retention_years=7)
    very_old = (datetime.utcnow() - timedelta(days=10 * 365)).date()
    just_old = (datetime.utcnow() - timedelta(days=400)).date()
    assert rm.should_delete(very_old) is True
    assert rm.should_delete(just_old) is False


def test_compliance_report_phi_access(tmp_path):
    audit = _make_logger(tmp_path)
    now = datetime.utcnow()
    audit.log_system_event("boot", "Boot complete")
    audit.log_phi_access(
        user_id="u1",
        patient_id="P200",
        patient_name="Doe^Jane",
        accessed_fields=["PatientID", "DOB"],
        reason="diagnosis",
    )
    audit.log_phi_access(
        user_id="u2",
        patient_id="P201",
        patient_name="Doe^John",
        accessed_fields=["PatientName"],
        reason="billing",
    )
    report = audit.generate_compliance_report(
        start_date=now - timedelta(seconds=5),
        end_date=now + timedelta(seconds=5),
        report_type="phi_access",
    )
    assert report["report_type"] == "phi_access"
    assert report["total_phi_events"] == 2
    entry_types = {e["event_type"] for e in report["entries"]}
    assert entry_types == {"PHI_ACCESS"}


def test_verify_log_integrity_all_valid(tmp_path):
    audit = _make_logger(tmp_path)
    for i in range(5):
        audit.log_system_event("event", f"Event {i}")
    result = audit.verify_log_integrity()
    assert result["total"] == 5
    assert result["valid"] == 5
    assert result["tampered"] == 0


def test_verify_log_integrity_detects_tampered_file(tmp_path):
    audit = _make_logger(tmp_path)
    audit.log_system_event("start", "Started")
    entries_dir = tmp_path / "audit" / "entries"
    json_files = list(entries_dir.glob("????????/*.json"))
    assert json_files
    victim = json_files[0]
    payload = json.loads(victim.read_text(encoding="utf-8"))
    payload["data"]["description"] = "TAMPERED"
    victim.write_text(json.dumps(payload), encoding="utf-8")

    result = audit.verify_log_integrity()
    assert result["tampered"] == 1
    assert result["valid"] == 0


def test_log_security_event(tmp_path):
    audit = _make_logger(tmp_path)
    mid = audit.log_security_event(
        {
            "event_type": "authentication_failure",
            "user_id": "hacker",
            "details": "Failed login attempt",
            "outcome": 8,
        }
    )
    assert mid
    hits = audit.search_logs(event_type="SECURITY_EVENT")
    assert any(h["message_id"] == mid for h in hits)


def test_log_dicom_store(tmp_path):
    audit = _make_logger(tmp_path)
    mid = audit.log_dicom_store(
        user_id="ai-system",
        endpoint=_make_endpoint(),
        study_instance_uid="1.2.3.4",
        sop_instance_uid="1.2.3.4.5",
        outcome=0,
    )
    assert mid
    hits = audit.search_logs(event_type="DICOM_STORE")
    assert any(h["message_id"] == mid for h in hits)


def test_search_limit(tmp_path):
    audit = _make_logger(tmp_path)
    for i in range(20):
        audit.log_system_event("ping", f"Ping {i}")
    hits = audit.search_logs(limit=5)
    assert len(hits) == 5


def test_search_results_sorted_descending(tmp_path):
    audit = _make_logger(tmp_path)
    for i in range(3):
        audit.log_system_event("e", f"Event {i}")
    hits = audit.search_logs()
    datetimes = [h["event_datetime"] for h in hits]
    assert datetimes == sorted(datetimes, reverse=True)


def test_hipaa_format_participants_and_study_objects(tmp_path):
    """to_hipaa_format includes non-empty participants and study_objects lists."""
    audit = _make_logger(tmp_path)
    audit.log_phi_access(
        user_id="dr.q",
        patient_id="P500",
        patient_name="Query^Doctor",
        accessed_fields=["PatientID"],
        reason="treatment",
    )
    # Retrieve the raw entry from storage to call to_hipaa_format
    entries_dir = tmp_path / "audit" / "entries"
    json_files = list(entries_dir.glob("????????/*.json"))
    assert json_files
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    data = payload["data"]
    # Reconstruct just enough for to_hipaa_format via to_dict round-trip
    msg = AuditMessage(
        message_id=data["message_id"],
        event_id=data["event_id"],
        event_action=data["event_action"],
        event_outcome=data["event_outcome"],
        event_datetime=datetime.fromisoformat(data["event_datetime"]),
        event_type=data["event_type"],
        participants=[AuditParticipant(**p) for p in data["participants"]],
        study_objects=[AuditStudyObject(**s) for s in data["study_objects"]],
        description=data["description"],
        phi_accessed=data["phi_accessed"],
        phi_fields=data["phi_fields"],
    )
    hipaa = msg.to_hipaa_format()
    assert hipaa["participants"]
    assert hipaa["study_objects"]
    assert hipaa["event_type"] == "PHI_ACCESS"
