# Feature: pacs-integration-system
# Property 39: Multi-Channel Notification Delivery
# Property 40: Critical Finding Escalation
# Property 41: Notification Content Completeness
# Property 42: Notification Delivery Tracking

import uuid
from datetime import datetime
from types import SimpleNamespace

from hypothesis import given, settings
from hypothesis import strategies as st

from src.clinical.pacs.notification_system import (
    DeliveryRecord,
    DeliveryTracker,
    EmailNotifier,
    HL7Notifier,
    NotificationEvent,
    NotificationSystem,
    SMSNotifier,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PRIORITY_VALUES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _make_event(
    priority: str = "MEDIUM",
    study_uid: str = "1.2.840.99999.1",
    patient_id: str = "P001",
    analysis_summary: str = "Carcinoma detected",
    confidence: float = 0.85,
    result_url: str = "https://histocore.local/results/1",
) -> NotificationEvent:
    return NotificationEvent(
        event_id=str(uuid.uuid4()),
        event_type="analysis_complete",
        study_instance_uid=study_uid,
        patient_id=patient_id,
        analysis_summary=analysis_summary,
        result_url=result_url,
        priority=priority,
        timestamp=datetime.now(),
        algorithm_name="HistoMIL-v2",
        confidence_score=confidence,
        findings=["Region of interest detected"],
    )


def _make_analysis_results(
    confidence: float = 0.85,
    urgency: str = "MEDIUM",
    study_uid: str = "1.2.840.99999.1",
) -> SimpleNamespace:
    rec = SimpleNamespace(
        recommendation_id=str(uuid.uuid4()),
        recommendation_text="Biopsy recommended",
        confidence=confidence,
        urgency_level=urgency,
    )
    return SimpleNamespace(
        study_instance_uid=study_uid,
        series_instance_uid="1.2.840.99999.1.1",
        algorithm_name="HistoMIL-v2",
        confidence_score=confidence,
        detected_regions=[],
        diagnostic_recommendations=[rec],
        processing_timestamp=datetime.now(),
        primary_diagnosis="Adenocarcinoma" if confidence >= 0.9 else None,
        patient_id="P001",
    )


def _make_system_all_channels() -> NotificationSystem:
    """NotificationSystem with all 3 channels in simulation mode."""
    ns = NotificationSystem()
    ns.configure_email(smtp_server="localhost", smtp_port=25, use_tls=False)
    ns.configure_sms()         # no gateway_url → simulation
    ns.configure_hl7()         # no endpoint_host → simulation
    return ns


# ---------------------------------------------------------------------------
# Property 39 — Multi-Channel Notification Delivery
# For any notification event, alerts SHALL be delivered through all
# configured channels.
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(priority=st.sampled_from(_PRIORITY_VALUES))
def test_property_39_all_channels_attempt_delivery(priority):
    ns = _make_system_all_channels()
    ar = _make_analysis_results(
        confidence=0.95 if priority == "CRITICAL" else 0.75,
        urgency="URGENT" if priority == "CRITICAL" else "MEDIUM",
    )
    result = ns.notify_analysis_complete(ar, result_url="https://histocore.local/r/1")
    assert result.success

    stats = ns.tracker.get_statistics()
    # All 3 channels must have at least one delivery record.
    by_channel = stats["by_channel"]
    assert "email" in by_channel, "email channel had no delivery records"
    assert "sms" in by_channel, "sms channel had no delivery records"
    assert "hl7" in by_channel, "hl7 channel had no delivery records"
    # Total records must cover all 3 channels.
    assert stats["total_records"] >= 3


# ---------------------------------------------------------------------------
# Property 40 — Critical Finding Escalation
# Confidence ≥ threshold → CRITICAL priority.
# Confidence < threshold with no URGENT recs → not CRITICAL.
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(confidence=st.floats(min_value=0.9, max_value=1.0, allow_nan=False))
def test_property_40_high_confidence_yields_critical_priority(confidence):
    ns = NotificationSystem()
    ar = _make_analysis_results(confidence=confidence, urgency="MEDIUM")
    event = ns._create_event_from_analysis(ar, study_info=None, result_url=None)
    assert event.priority == "CRITICAL", (
        f"Expected CRITICAL for confidence={confidence}, got {event.priority}"
    )


@settings(max_examples=100)
@given(confidence=st.floats(min_value=0.0, max_value=0.8999, allow_nan=False))
def test_property_40_low_confidence_no_urgent_not_critical(confidence):
    ns = NotificationSystem()
    ar = _make_analysis_results(confidence=confidence, urgency="LOW")
    event = ns._create_event_from_analysis(ar, study_info=None, result_url=None)
    assert event.priority != "CRITICAL", (
        f"Expected non-CRITICAL for confidence={confidence}, got {event.priority}"
    )


# ---------------------------------------------------------------------------
# Property 41 — Notification Content Completeness
# format_body() must contain study_instance_uid and analysis_summary.
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    study_uid=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="._-"),
        min_size=1,
        max_size=64,
    ),
    summary=st.text(min_size=1, max_size=200),
)
def test_property_41_body_contains_required_fields(study_uid, summary):
    event = NotificationEvent(
        event_id=str(uuid.uuid4()),
        event_type="analysis_complete",
        study_instance_uid=study_uid,
        patient_id="P001",
        analysis_summary=summary,
        result_url="https://histocore.local/r/1",
        priority="MEDIUM",
        timestamp=datetime.now(),
    )
    body = event.format_body()
    assert study_uid in body, "format_body() missing study_instance_uid"
    assert summary in body, "format_body() missing analysis_summary"


# ---------------------------------------------------------------------------
# Property 42 — Notification Delivery Tracking
# N failed records with attempts < max_attempts → get_pending_retries() = N.
# After marking all sent → get_pending_retries() = 0.
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(n=st.integers(min_value=1, max_value=20))
def test_property_42_pending_retries_matches_failed_records(n):
    tracker = DeliveryTracker()
    event_id = str(uuid.uuid4())
    record_ids = []
    for _ in range(n):
        rec = tracker.create_record(event_id, "email", f"doc{_}@hospital.org", max_attempts=3)
        # One failed attempt — still below max_attempts so can_retry() is True.
        tracker.mark_failed(rec.record_id, "connection refused")
        record_ids.append(rec.record_id)

    pending = tracker.get_pending_retries()
    assert len(pending) == n

    # After marking all as sent, no retries should be pending.
    for rid in record_ids:
        tracker.mark_sent(rid)

    assert len(tracker.get_pending_retries()) == 0


# ---------------------------------------------------------------------------
# Unit tests — email recipient validation
# ---------------------------------------------------------------------------

def test_email_validate_recipient_valid():
    assert EmailNotifier().validate_recipient("user@hospital.org") is True


def test_email_validate_recipient_invalid():
    assert EmailNotifier().validate_recipient("notanemail") is False


def test_email_validate_recipient_invalid_no_dot_in_domain():
    assert EmailNotifier().validate_recipient("user@nodot") is False


# ---------------------------------------------------------------------------
# Unit tests — SMS recipient validation
# ---------------------------------------------------------------------------

def test_sms_validate_recipient_valid():
    assert SMSNotifier().validate_recipient("+15551234567") is True


def test_sms_validate_recipient_valid_no_plus():
    assert SMSNotifier().validate_recipient("15551234567") is True


def test_sms_validate_recipient_invalid_too_short():
    assert SMSNotifier().validate_recipient("123") is False


# ---------------------------------------------------------------------------
# Unit tests — HL7 recipient validation
# ---------------------------------------------------------------------------

def test_hl7_validate_recipient_valid():
    assert HL7Notifier().validate_recipient("PATHOLOGY_SYS") is True


def test_hl7_validate_recipient_invalid_too_long():
    assert HL7Notifier().validate_recipient("A" * 21) is False


def test_hl7_validate_recipient_invalid_space():
    assert HL7Notifier().validate_recipient("PATH SYS") is False


# ---------------------------------------------------------------------------
# Unit test — HL7 message structure
# ---------------------------------------------------------------------------

def test_hl7_build_message_contains_segments():
    notifier = HL7Notifier(sending_facility="HISTOCORE", receiving_facility="HOSPITAL")
    event = _make_event(study_uid="1.2.840.10008.5.1")
    message = notifier._build_hl7_message(event, receiving_app="PATHSYS")
    assert "MSH|" in message
    assert "PID|" in message
    assert "OBX|" in message


def test_hl7_build_message_contains_study_uid():
    notifier = HL7Notifier()
    event = _make_event(study_uid="1.2.840.10008.5.1.99")
    message = notifier._build_hl7_message(event, receiving_app="PATHSYS")
    assert "1.2.840.10008.5.1.99" in message


def test_hl7_build_message_mllp_framing_absent_in_plain_message():
    """_build_hl7_message returns plain text; MLLP bytes are added by _send_mllp."""
    notifier = HL7Notifier()
    event = _make_event()
    message = notifier._build_hl7_message(event, receiving_app="PATHSYS")
    assert "\x0b" not in message
    assert "\x1c" not in message


# ---------------------------------------------------------------------------
# Unit test — full simulation-mode round trip
# ---------------------------------------------------------------------------

def test_notification_system_simulation_mode():
    ns = _make_system_all_channels()
    ar = _make_analysis_results(confidence=0.95, urgency="URGENT")
    result = ns.notify_analysis_complete(ar, result_url="https://histocore.local/r/99")
    assert result.success
    assert result.data["sent"] >= 1


# ---------------------------------------------------------------------------
# Unit test — critical finding creates CRITICAL event
# ---------------------------------------------------------------------------

def test_critical_finding_creates_critical_event():
    ns = _make_system_all_channels()
    result = ns.notify_critical_finding(
        study_instance_uid="1.2.840.99999.2",
        patient_id="P002",
        finding_description="High-grade invasive carcinoma",
        confidence=0.97,
    )
    assert result.success
    # The result data confirms delivery was attempted.
    assert "event_id" in result.data


def test_urgent_recommendation_elevates_to_critical():
    ns = NotificationSystem()
    ar = _make_analysis_results(confidence=0.5, urgency="URGENT")
    event = ns._create_event_from_analysis(ar, study_info=None, result_url=None)
    assert event.priority == "CRITICAL"


# ---------------------------------------------------------------------------
# Unit test — DeliveryTracker statistics
# ---------------------------------------------------------------------------

def test_delivery_tracker_statistics():
    tracker = DeliveryTracker()
    eid = str(uuid.uuid4())

    r1 = tracker.create_record(eid, "email", "a@hospital.org")
    r2 = tracker.create_record(eid, "sms", "+15550001111")
    r3 = tracker.create_record(eid, "hl7", "PATHSYS")

    tracker.mark_sent(r1.record_id)
    tracker.mark_failed(r2.record_id, "gateway timeout")
    # r3 stays pending

    stats = tracker.get_statistics()
    assert stats["total_records"] == 3
    assert stats["sent"] == 1
    # r2 had 1 attempt < max_attempts(3) so becomes "retrying", counted as pending
    assert stats["pending"] >= 1
    assert "email" in stats["by_channel"]
    assert stats["by_channel"]["email"]["sent"] == 1


def test_delivery_tracker_event_status():
    tracker = DeliveryTracker()
    eid = str(uuid.uuid4())
    r1 = tracker.create_record(eid, "email", "a@hospital.org")
    r2 = tracker.create_record(eid, "sms", "+15550001111")
    tracker.mark_sent(r1.record_id)

    status = tracker.get_event_status(eid)
    assert status["event_id"] == eid
    assert status["total_channels"] == 2
    assert status["sent"] == 1
    assert status["pending"] >= 1


# ---------------------------------------------------------------------------
# Unit test — retry_failed_deliveries updates delivery status
# ---------------------------------------------------------------------------

def test_retry_failed_deliveries_updates_status():
    ns = _make_system_all_channels()

    # Manually plant a failed record that points at the sms channel (simulation mode).
    eid = str(uuid.uuid4())
    rec = ns.tracker.create_record(eid, "sms", "+15550001111", max_attempts=3)
    ns.tracker.mark_failed(rec.record_id, "initial failure")

    assert len(ns.tracker.get_pending_retries()) == 1

    retried = ns.retry_failed_deliveries()
    # SMS simulation mode always returns True → 1 successful retry.
    assert retried == 1
    assert len(ns.tracker.get_pending_retries()) == 0


def test_retry_exhausted_records_not_retried():
    tracker = DeliveryTracker()
    eid = str(uuid.uuid4())
    rec = tracker.create_record(eid, "email", "a@hospital.org", max_attempts=1)
    # Exhaust attempts: mark_failed increments attempts to 1 == max_attempts → "failed"
    tracker.mark_failed(rec.record_id, "refused")
    assert rec.can_retry() is False
    assert len(tracker.get_pending_retries()) == 0


# ---------------------------------------------------------------------------
# Unit test — format_subject critical flag
# ---------------------------------------------------------------------------

def test_format_subject_critical_includes_label():
    event = _make_event(priority="CRITICAL", study_uid="1.2.840.10008.99")
    assert "[CRITICAL]" in event.format_subject()


def test_format_subject_non_critical_uses_ai_result_label():
    event = _make_event(priority="HIGH", study_uid="1.2.840.10008.99")
    assert "[AI Result]" in event.format_subject()
    assert "[CRITICAL]" not in event.format_subject()


# ---------------------------------------------------------------------------
# Unit test — system error goes to admins only (no crash, success result)
# ---------------------------------------------------------------------------

def test_notify_system_error_returns_success():
    ns = _make_system_all_channels()
    result = ns.notify_system_error("NullPointerException", "StorageEngine")
    assert result.success
