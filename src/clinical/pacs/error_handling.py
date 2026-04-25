"""Error handling primitives for PACS integration: exceptions, handlers, and dead-letter queue."""

import json
import logging
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data_models import PACSEndpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class PACSConnectionError(Exception):
    """Raised when a network-level connection to a PACS endpoint fails."""


class PACSTimeoutError(PACSConnectionError):
    """Raised when a PACS operation exceeds its configured timeout."""


class PACSAssociationError(PACSConnectionError):
    """Raised when a DICOM association cannot be established or is rejected."""


class DicomProtocolError(Exception):
    """Base class for DICOM-protocol-level errors."""


class DicomCFindError(DicomProtocolError):
    """Raised when a C-FIND operation returns an unrecoverable error status."""


class DicomCMoveError(DicomProtocolError):
    """Raised when a C-MOVE operation returns an unrecoverable error status."""


class DicomCStoreError(DicomProtocolError):
    """Raised when a C-STORE operation returns an unrecoverable error status."""


class DiskSpaceError(Exception):
    """Raised when insufficient disk space prevents storing DICOM files."""


class PACSAuthenticationError(PACSConnectionError):
    """Raised when PACS credentials are rejected; retrying with the same creds is pointless."""


# ---------------------------------------------------------------------------
# DICOM status-code tables
# ---------------------------------------------------------------------------

CFIND_STATUS_PENDING = 0xFF00
CFIND_STATUS_SUCCESS = 0x0000
CFIND_STATUS_CANCEL = 0xFE00
CFIND_ERROR_CODES: Dict[int, str] = {
    0xA700: "Out of resources",
    0xA900: "Dataset does not match SOP class",
    0xC000: "Unable to process",
}

CMOVE_STATUS_PENDING = 0xFF00
CMOVE_STATUS_SUCCESS = 0x0000
CMOVE_ERROR_CODES: Dict[int, str] = {
    0xA701: "Out of resources - unable to calculate number of matches",
    0xA702: "Out of resources - unable to perform sub-operations",
    0xA801: "Move destination unknown",
    0xA900: "Identifier does not match SOP class",
    0xC000: "Unable to process",
}

CSTORE_STATUS_SUCCESS = 0x0000
CSTORE_WARNING_COERCION = 0xB000
CSTORE_WARNING_ELEMENTS_DISCARDED = 0xB006
CSTORE_WARNING_DATASET_COERCION = 0xB007
CSTORE_ERROR_CODES: Dict[int, str] = {
    0xA700: "Out of resources",
    0xA900: "Dataset does not match SOP class",
    0xC000: "Cannot understand",
}


# ---------------------------------------------------------------------------
# NetworkErrorHandler
# ---------------------------------------------------------------------------

_NON_RETRYABLE = (PACSAuthenticationError, DiskSpaceError, ValueError)
_RETRYABLE = (PACSConnectionError, PACSTimeoutError, OSError, RuntimeError)


class NetworkErrorHandler:
    """Calculates retry schedules and decides retry eligibility for network errors."""

    def __init__(
        self,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        # Tracks the current attempt count per endpoint for association decisions.
        self._attempt_counters: Dict[str, int] = {}

    def calculate_backoff_delay(self, attempt: int) -> float:
        """Return exponential backoff delay for *attempt* (0-indexed), capped at max_delay."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        if self.jitter:
            # Small random offset avoids thundering-herd when many clients retry simultaneously.
            delay += random.uniform(0, delay * 0.1)
        return delay

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Return True if the error is retryable and attempts remain."""
        if attempt >= self.max_attempts:
            return False
        if isinstance(error, _NON_RETRYABLE):
            return False
        return isinstance(error, _RETRYABLE)

    def handle_connection_timeout(self, endpoint: PACSEndpoint, attempt: int) -> float:
        """Log the timeout event and return the backoff delay for the next attempt."""
        delay = self.calculate_backoff_delay(attempt)
        logger.warning(
            "Connection timeout on endpoint %s (host=%s, attempt=%d/%d). "
            "Retrying in %.1fs.",
            endpoint.endpoint_id,
            endpoint.host,
            attempt + 1,
            self.max_attempts,
            delay,
        )
        return delay

    def handle_association_failure(self, endpoint: PACSEndpoint, reason: str) -> str:
        """Log the association failure and return the recommended recovery action.

        Returns "retry", "failover", or "abort" based on how many attempts have
        been consumed.  Switching to a backup endpoint is preferred once half the
        budget is spent but before we give up entirely.
        """
        counter_key = endpoint.endpoint_id
        self._attempt_counters[counter_key] = self._attempt_counters.get(counter_key, 0) + 1
        attempt = self._attempt_counters[counter_key]

        if attempt >= self.max_attempts:
            action = "abort"
        elif attempt >= self.max_attempts // 2:
            action = "failover"
        else:
            action = "retry"

        logger.error(
            "Association failure on endpoint %s (host=%s): %s. "
            "Attempt %d/%d → action=%s.",
            endpoint.endpoint_id,
            endpoint.host,
            reason,
            attempt,
            self.max_attempts,
            action,
        )
        return action


# ---------------------------------------------------------------------------
# DicomErrorHandler
# ---------------------------------------------------------------------------


class DicomErrorHandler:
    """Interprets DICOM status codes and recommends recovery actions."""

    # ------------------------------------------------------------------
    # C-FIND
    # ------------------------------------------------------------------

    def handle_c_find_status(self, status_code: int, error_comment: str = "") -> str:
        """Translate a C-FIND status code into an action string.

        Returns: "continue" (more results pending), "complete", "cancel", or "retry".
        Raises DicomCFindError for codes that cannot be recovered by retrying.
        """
        if status_code == CFIND_STATUS_PENDING:
            logger.debug("C-FIND pending (0x%04X), continuing to receive datasets.", status_code)
            return "continue"

        if status_code == CFIND_STATUS_SUCCESS:
            logger.info("C-FIND completed successfully (0x0000).")
            return "complete"

        if status_code == CFIND_STATUS_CANCEL:
            logger.info("C-FIND was cancelled by the remote (0xFE00).")
            return "cancel"

        description = CFIND_ERROR_CODES.get(status_code)
        if description:
            msg = f"C-FIND error 0x{status_code:04X}: {description}"
            if error_comment:
                msg += f" — {error_comment}"
            logger.error(msg)
            raise DicomCFindError(msg)

        # Unknown but non-fatal codes: attempt a retry and let the caller decide.
        logger.warning(
            "C-FIND unknown status 0x%04X (%s). Attempting retry.",
            status_code,
            error_comment,
        )
        return "retry"

    # ------------------------------------------------------------------
    # C-MOVE
    # ------------------------------------------------------------------

    def handle_c_move_status(
        self,
        status_code: int,
        failed_instance_uids: Optional[List[str]] = None,
    ) -> str:
        """Translate a C-MOVE status code into an action string.

        Returns: "pending", "complete", "partial_retry" (completed but some SOPs
        failed), "retry", or "abort".
        """
        failed_instance_uids = failed_instance_uids or []

        if status_code == CMOVE_STATUS_PENDING:
            logger.debug("C-MOVE sub-operation pending (0x%04X).", status_code)
            return "pending"

        if status_code == CMOVE_STATUS_SUCCESS:
            if failed_instance_uids:
                logger.warning(
                    "C-MOVE completed but %d instances failed: %s",
                    len(failed_instance_uids),
                    failed_instance_uids,
                )
                return "partial_retry"
            logger.info("C-MOVE completed successfully (0x0000).")
            return "complete"

        description = CMOVE_ERROR_CODES.get(status_code)
        if description:
            logger.error("C-MOVE error 0x%04X: %s", status_code, description)
            # Resource errors may resolve after backoff; destination errors will not.
            if status_code in (0xA801, 0xA900):
                return "abort"
            return "retry"

        logger.warning("C-MOVE unknown status 0x%04X. Treating as retriable.", status_code)
        return "retry"

    # ------------------------------------------------------------------
    # C-STORE
    # ------------------------------------------------------------------

    def handle_c_store_status(self, status_code: int, sop_instance_uid: str = "") -> str:
        """Translate a C-STORE status code into an action string.

        Returns: "success", "warning_coercion", "retry", or "abort".
        Raises DicomCStoreError for unrecoverable errors.
        """
        if status_code == CSTORE_STATUS_SUCCESS:
            logger.debug("C-STORE success for SOP %s (0x0000).", sop_instance_uid)
            return "success"

        if self.is_warning_status(status_code):
            logger.warning(
                "C-STORE warning 0x%04X for SOP %s — proceeding with coercion.",
                status_code,
                sop_instance_uid,
            )
            return "warning_coercion"

        description = CSTORE_ERROR_CODES.get(status_code, "Unknown error")
        msg = f"C-STORE error 0x{status_code:04X} for SOP {sop_instance_uid}: {description}"
        logger.error(msg)
        raise DicomCStoreError(msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_warning_status(self, status_code: int) -> bool:
        """Return True for DICOM warning status codes (0xB000–0xBFFF)."""
        return 0xB000 <= status_code <= 0xBFFF

    def is_error_status(self, status_code: int) -> bool:
        """Return True for DICOM error status codes (0xA000–0xAFFF or 0xC000–0xCFFF)."""
        return (0xA000 <= status_code <= 0xAFFF) or (0xC000 <= status_code <= 0xCFFF)


# ---------------------------------------------------------------------------
# FailedOperation dataclass
# ---------------------------------------------------------------------------


@dataclass
class FailedOperation:
    """Represents a PACS operation that could not be completed and awaits retry."""

    operation_id: str
    operation_type: str  # "c_find" | "c_move" | "c_store"
    study_instance_uid: str
    endpoint_id: str
    error_message: str
    error_code: Optional[int]
    retry_count: int
    max_retries: int
    first_failure_time: datetime
    last_failure_time: datetime
    queued_time: datetime
    operation_data: Optional[Dict[str, Any]] = field(default=None)

    def is_expired(self, max_age_hours: int = 72) -> bool:
        """Return True if the operation has been sitting in the queue past max_age_hours."""
        age = datetime.utcnow() - self.queued_time
        return age.total_seconds() > max_age_hours * 3600

    def can_retry(self) -> bool:
        """Return True if at least one retry attempt remains."""
        return self.retry_count < self.max_retries


# ---------------------------------------------------------------------------
# DeadLetterQueue
# ---------------------------------------------------------------------------


class DeadLetterQueue:
    """Thread-safe queue for failed PACS operations that require manual review or retry."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_queue_size: int = 1000,
    ) -> None:
        self._storage_path = storage_path
        self._max_queue_size = max_queue_size
        self._queue: List[FailedOperation] = []
        self._lock = threading.Lock()

        if self._storage_path and self._storage_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, operation: FailedOperation) -> None:
        """Add *operation* to the queue thread-safely.

        If the queue is at capacity, expired operations are pruned first to make
        room.  If still full after pruning, the oldest entry is dropped so that
        fresh failures are always preserved.
        """
        with self._lock:
            if len(self._queue) >= self._max_queue_size:
                logger.warning(
                    "Dead-letter queue at capacity (%d). Pruning expired operations.",
                    self._max_queue_size,
                )
                self._queue = [op for op in self._queue if not op.is_expired()]

            if len(self._queue) >= self._max_queue_size:
                dropped = self._queue.pop(0)
                logger.warning(
                    "Dead-letter queue still full after pruning. Dropping oldest entry: %s",
                    dropped.operation_id,
                )

            self._queue.append(operation)
            self._persist()

    def dequeue_for_retry(self, max_items: int = 10) -> List[FailedOperation]:
        """Remove and return up to *max_items* retryable, non-expired operations."""
        with self._lock:
            eligible = [op for op in self._queue if op.can_retry() and not op.is_expired()]
            to_return = eligible[:max_items]
            returned_ids = {op.operation_id for op in to_return}
            self._queue = [op for op in self._queue if op.operation_id not in returned_ids]
            self._persist()
        return to_return

    def get_failed_permanent(self) -> List[FailedOperation]:
        """Return operations that have exhausted all retry attempts."""
        with self._lock:
            return [op for op in self._queue if not op.can_retry()]

    def prune_expired(self, max_age_hours: int = 72) -> int:
        """Remove all expired operations; return the count removed."""
        with self._lock:
            before = len(self._queue)
            self._queue = [op for op in self._queue if not op.is_expired(max_age_hours)]
            removed = before - len(self._queue)
            if removed:
                self._persist()
        return removed

    def generate_failure_report(self) -> Dict[str, Any]:
        """Produce a summary of queue contents for operator review."""
        with self._lock:
            total = len(self._queue)
            permanent = [op for op in self._queue if not op.can_retry()]
            retryable = [op for op in self._queue if op.can_retry() and not op.is_expired()]

            by_type: Dict[str, int] = {}
            by_endpoint: Dict[str, int] = {}
            oldest: Optional[datetime] = None

            for op in self._queue:
                by_type[op.operation_type] = by_type.get(op.operation_type, 0) + 1
                by_endpoint[op.endpoint_id] = by_endpoint.get(op.endpoint_id, 0) + 1
                if oldest is None or op.first_failure_time < oldest:
                    oldest = op.first_failure_time

        return {
            "total_queued": total,
            "permanent_failures": len(permanent),
            "retryable": len(retryable),
            "by_operation_type": by_type,
            "by_endpoint": by_endpoint,
            "oldest_failure": oldest.isoformat() if oldest else None,
        }

    def size(self) -> int:
        """Return the current number of operations in the queue."""
        with self._lock:
            return len(self._queue)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Write the queue to *storage_path* as JSON (no-op when path is None)."""
        if not self._storage_path:
            return

        records = []
        for op in self._queue:
            records.append(
                {
                    "operation_id": op.operation_id,
                    "operation_type": op.operation_type,
                    "study_instance_uid": op.study_instance_uid,
                    "endpoint_id": op.endpoint_id,
                    "error_message": op.error_message,
                    "error_code": op.error_code,
                    "retry_count": op.retry_count,
                    "max_retries": op.max_retries,
                    "first_failure_time": op.first_failure_time.isoformat(),
                    "last_failure_time": op.last_failure_time.isoformat(),
                    "queued_time": op.queued_time.isoformat(),
                    "operation_data": op.operation_data,
                }
            )

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def _load(self) -> None:
        """Populate the queue from an existing *storage_path* JSON file."""
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load dead-letter queue from %s: %s", self._storage_path, exc)
            return

        for record in raw:
            try:
                op = FailedOperation(
                    operation_id=record["operation_id"],
                    operation_type=record["operation_type"],
                    study_instance_uid=record["study_instance_uid"],
                    endpoint_id=record["endpoint_id"],
                    error_message=record["error_message"],
                    error_code=record.get("error_code"),
                    retry_count=record["retry_count"],
                    max_retries=record["max_retries"],
                    first_failure_time=datetime.fromisoformat(record["first_failure_time"]),
                    last_failure_time=datetime.fromisoformat(record["last_failure_time"]),
                    queued_time=datetime.fromisoformat(record["queued_time"]),
                    operation_data=record.get("operation_data"),
                )
                self._queue.append(op)
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed dead-letter record: %s", exc)
