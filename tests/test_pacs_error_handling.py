"""Tests for PACS error handling and failover components."""

import time
from datetime import datetime

import pytest

from hypothesis import given, settings
from hypothesis import strategies as st
from src.clinical.pacs.data_models import (
    PACSEndpoint,
    PACSVendor,
    PerformanceConfig,
    SecurityConfig,
)
from src.clinical.pacs.error_handling import (
    DeadLetterQueue,
    DicomCFindError,
    DicomCStoreError,
    DicomErrorHandler,
    FailedOperation,
    NetworkErrorHandler,
    PACSAuthenticationError,
    PACSConnectionError,
)
from src.clinical.pacs.failover import CircuitBreaker, CircuitState, FailoverManager

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_endpoint(endpoint_id: str = "ep1", is_primary: bool = True) -> PACSEndpoint:
    return PACSEndpoint(
        endpoint_id=endpoint_id,
        ae_title="TEST_AE",
        host="pacs.hospital.local",
        port=11112,
        vendor=PACSVendor.GENERIC,
        security_config=SecurityConfig(
            tls_enabled=False,
            verify_certificates=False,
            mutual_authentication=False,
        ),
        performance_config=PerformanceConfig(),
        is_primary=is_primary,
    )


def _make_failed_op(
    operation_id: str = "op-1",
    retry_count: int = 0,
    max_retries: int = 3,
    endpoint_id: str = "ep1",
    operation_type: str = "c_move",
) -> FailedOperation:
    now = datetime.utcnow()
    return FailedOperation(
        operation_id=operation_id,
        operation_type=operation_type,
        study_instance_uid="1.2.3.4.5",
        endpoint_id=endpoint_id,
        error_message="Connection refused",
        error_code=None,
        retry_count=retry_count,
        max_retries=max_retries,
        first_failure_time=now,
        last_failure_time=now,
        queued_time=now,
    )


# ---------------------------------------------------------------------------
# Property 25 — Dead Letter Queue Management
# Feature: pacs-integration-system, Property 25: Dead Letter Queue Management
# For any operation that fails after all retry attempts, the operation SHALL be
# queued in the dead letter queue for manual review.
# ---------------------------------------------------------------------------


@given(
    operation_id=st.text(
        min_size=1, max_size=64, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))
    ),
    retry_count=st.integers(min_value=0, max_value=20),
    max_retries=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=100)
def test_property_25_dead_letter_queue_permanent_failures(
    operation_id: str, retry_count: int, max_retries: int
):
    """Operations exhausting all retries must appear in get_failed_permanent()."""
    # Force retry_count >= max_retries so this is a permanent failure.
    max_retries = min(retry_count, max_retries)

    dlq = DeadLetterQueue()
    op = _make_failed_op(
        operation_id=operation_id,
        retry_count=retry_count,
        max_retries=max_retries,
    )
    dlq.enqueue(op)

    permanent = dlq.get_failed_permanent()
    assert any(p.operation_id == operation_id for p in permanent)


# ---------------------------------------------------------------------------
# Property 26 — Comprehensive Error Logging
# Feature: pacs-integration-system, Property 26: Comprehensive Error Logging
# For any error condition, detailed logging SHALL include all relevant information:
# DICOM status codes, network error details, and operation context.
# ---------------------------------------------------------------------------


@given(
    status_code=st.one_of(
        st.integers(min_value=0xA000, max_value=0xAFFF),
        st.integers(min_value=0xC000, max_value=0xCFFF),
    )
)
@settings(max_examples=100)
def test_property_26_error_range_detected(status_code: int):
    """Any status code in DICOM error range must be identified as an error."""
    handler = DicomErrorHandler()
    assert handler.is_error_status(status_code) is True
    assert handler.is_warning_status(status_code) is False


@given(status_code=st.integers(min_value=0xB000, max_value=0xBFFF))
@settings(max_examples=100)
def test_property_26_warning_range_detected(status_code: int):
    """Any status code in DICOM warning range must be identified as a warning."""
    handler = DicomErrorHandler()
    assert handler.is_warning_status(status_code) is True
    assert handler.is_error_status(status_code) is False


# ---------------------------------------------------------------------------
# Property 27 — Automatic Operation Resumption
# Feature: pacs-integration-system, Property 27: Automatic Operation Resumption
# For any service recovery from error conditions, previously queued operations
# SHALL be automatically resumed without manual intervention.
# ---------------------------------------------------------------------------


@given(n=st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_property_27_all_retryable_ops_dequeued(n: int):
    """All retryable operations returned by dequeue_for_retry() are removed from the queue."""
    dlq = DeadLetterQueue()
    for i in range(n):
        dlq.enqueue(_make_failed_op(operation_id=f"op-{i}", retry_count=0, max_retries=3))

    returned = dlq.dequeue_for_retry(max_items=n)
    assert len(returned) == n
    assert dlq.size() == 0


# ---------------------------------------------------------------------------
# Unit tests — NetworkErrorHandler
# ---------------------------------------------------------------------------


def test_backoff_delay_increases_exponentially():
    handler = NetworkErrorHandler(base_delay=1.0, max_delay=3600.0, jitter=False)
    delays = [handler.calculate_backoff_delay(attempt) for attempt in range(4)]
    for i in range(1, len(delays)):
        assert delays[i] >= delays[i - 1], f"delay at attempt {i} did not increase"


def test_backoff_delay_capped_at_max():
    max_delay = 60.0
    handler = NetworkErrorHandler(base_delay=1.0, max_delay=max_delay, jitter=False)
    delay = handler.calculate_backoff_delay(100)
    assert delay <= max_delay


def test_should_retry_auth_error_false():
    handler = NetworkErrorHandler(max_attempts=5)
    error = PACSAuthenticationError("Bad credentials")
    assert handler.should_retry(attempt=0, error=error) is False


def test_should_retry_connection_error_true():
    handler = NetworkErrorHandler(max_attempts=5)
    error = PACSConnectionError("Refused")
    assert handler.should_retry(attempt=0, error=error) is True


# ---------------------------------------------------------------------------
# Unit tests — CircuitBreaker
# ---------------------------------------------------------------------------


def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
    for _ in range(5):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_circuit_breaker_half_open_after_timeout():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.05)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # Wait for the recovery window to lapse.
    time.sleep(0.1)

    # Calling `call()` with a no-op triggers the OPEN → HALF_OPEN transition.
    cb.call(lambda: None)
    assert cb.state in (CircuitState.HALF_OPEN, CircuitState.CLOSED)


def test_circuit_breaker_closes_after_successes():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.05, success_threshold=2)
    for _ in range(3):
        cb.record_failure()

    time.sleep(0.1)
    cb.call(lambda: None)  # first probe — transitions to HALF_OPEN then records success

    # One more success should close the circuit.
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Unit tests — FailoverManager
# ---------------------------------------------------------------------------


def test_failover_manager_selects_primary_first():
    primary = _make_endpoint("primary", is_primary=True)
    backup = _make_endpoint("backup", is_primary=False)
    mgr = FailoverManager([backup, primary])  # intentionally out of order

    selected = mgr.select_endpoint()
    assert selected.endpoint_id == "primary"


def test_failover_manager_falls_back_to_backup():
    primary = _make_endpoint("primary", is_primary=True)
    backup = _make_endpoint("backup", is_primary=False)
    mgr = FailoverManager([primary, backup])

    # Exhaust the primary's circuit breaker.
    for _ in range(5):
        mgr.mark_endpoint_failed(primary, PACSConnectionError("down"))

    selected = mgr.select_endpoint()
    assert selected.endpoint_id == "backup"


# ---------------------------------------------------------------------------
# Unit tests — DeadLetterQueue
# ---------------------------------------------------------------------------


def test_dead_letter_queue_prune_expired():
    dlq = DeadLetterQueue()

    # Enqueue an operation whose queued_time is far in the past.
    old_time = datetime(2000, 1, 1)
    op = FailedOperation(
        operation_id="old-op",
        operation_type="c_find",
        study_instance_uid="1.2.3",
        endpoint_id="ep1",
        error_message="old error",
        error_code=None,
        retry_count=0,
        max_retries=3,
        first_failure_time=old_time,
        last_failure_time=old_time,
        queued_time=old_time,
    )
    dlq.enqueue(op)
    assert dlq.size() == 1

    removed = dlq.prune_expired(max_age_hours=72)
    assert removed == 1
    assert dlq.size() == 0


# ---------------------------------------------------------------------------
# Unit tests — DicomErrorHandler
# ---------------------------------------------------------------------------


def test_cfind_error_status_raises():
    handler = DicomErrorHandler()
    with pytest.raises(DicomCFindError):
        handler.handle_c_find_status(0xA700)


def test_cmove_partial_retry():
    handler = DicomErrorHandler()
    action = handler.handle_c_move_status(
        0x0000,
        failed_instance_uids=["1.2.3.4.5.6"],
    )
    assert action == "partial_retry"


def test_cstore_warning_returns_warning_coercion():
    handler = DicomErrorHandler()
    action = handler.handle_c_store_status(0xB000, sop_instance_uid="1.2.3")
    assert action == "warning_coercion"
