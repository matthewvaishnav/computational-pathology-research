"""Circuit-breaker and failover manager for PACS endpoint redundancy."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .data_models import PACSEndpoint
from .error_handling import PACSConnectionError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circuit-breaker
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    """States of a circuit-breaker protecting a single PACS endpoint."""

    CLOSED = "closed"  # Normal — requests flow through.
    OPEN = "open"  # Failing — requests are rejected immediately.
    HALF_OPEN = "half_open"  # Probing — limited requests allowed to test recovery.


class CircuitBreaker:
    """Per-endpoint circuit breaker that prevents thundering-herd on a failing PACS host.

    State machine:
        CLOSED → (failure_threshold consecutive failures) → OPEN
        OPEN   → (recovery_timeout elapsed)               → HALF_OPEN
        HALF_OPEN → (success_threshold successes)         → CLOSED
        HALF_OPEN → (any failure)                         → OPEN
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change: float = time.monotonic()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* if the circuit allows it; record the outcome.

        Raises PACSConnectionError immediately when the circuit is OPEN and the
        recovery timeout has not yet elapsed.
        """
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self._last_failure_time or 0)
            if elapsed < self._recovery_timeout:
                raise PACSConnectionError(
                    f"Circuit open — endpoint unavailable for {self._recovery_timeout - elapsed:.0f}s more."
                )
            # Recovery window has elapsed; try a probe call.
            self._transition(CircuitState.HALF_OPEN)

        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise

        self.record_success()
        return result

    def record_success(self) -> None:
        """Notify the breaker of a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._success_threshold:
                logger.info(
                    "Circuit breaker closing after %d consecutive successes.", self._success_count
                )
                self._transition(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED:
            # Reset running failure count so that isolated errors don't accumulate.
            self._failure_count = 0

    def record_failure(self) -> None:
        """Notify the breaker of a failed call."""
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                logger.warning(
                    "Circuit breaker opening after %d consecutive failures.",
                    self._failure_count,
                )
                self._transition(CircuitState.OPEN)

        elif self._state == CircuitState.HALF_OPEN:
            # Any failure during the probe resets the recovery clock.
            logger.warning("Circuit breaker re-opening after probe failure.")
            self._transition(CircuitState.OPEN)

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the breaker's internal state for health-check endpoints."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": (
                datetime.fromtimestamp(self._last_failure_time).isoformat()
                if self._last_failure_time
                else None
            ),
            "last_state_change": datetime.fromtimestamp(self._last_state_change).isoformat(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: CircuitState) -> None:
        self._state = new_state
        self._last_state_change = time.monotonic()
        self._failure_count = 0
        self._success_count = 0


# ---------------------------------------------------------------------------
# Failover manager
# ---------------------------------------------------------------------------


@dataclass
class EndpointHealth:
    """Mutable health record for a single PACS endpoint."""

    endpoint: PACSEndpoint
    is_available: bool = True
    failure_count: int = 0
    last_check_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)


class FailoverManager:
    """Manages an ordered list of PACS endpoints and enforces circuit-breaker-based failover.

    The primary endpoint (``is_primary=True``) is always preferred.  When it is
    unavailable (circuit OPEN), the manager transparently selects the next healthy
    backup endpoint so that callers never need to hard-code fallback logic.
    """

    def __init__(
        self,
        endpoints: List[PACSEndpoint],
        health_check_interval: float = 30.0,
    ) -> None:
        self._health_check_interval = health_check_interval
        self._health: Dict[str, EndpointHealth] = {}

        # Sort so that the primary endpoint is first; order among backups is
        # preserved from the input list (which callers may have ranked by preference).
        sorted_endpoints = sorted(endpoints, key=lambda ep: (not ep.is_primary,))
        for ep in sorted_endpoints:
            self._health[ep.endpoint_id] = EndpointHealth(endpoint=ep)

    # ------------------------------------------------------------------
    # Endpoint selection
    # ------------------------------------------------------------------

    def select_endpoint(self, operation_type: str = "query") -> PACSEndpoint:
        """Return the best available endpoint for *operation_type*.

        An endpoint is considered available when its circuit breaker is not in the
        OPEN state.  Raises PACSConnectionError if no endpoint is usable.
        """
        for health in self._health.values():
            cb_state = health.circuit_breaker.state
            if cb_state != CircuitState.OPEN:
                health.last_check_time = datetime.utcnow()
                return health.endpoint

        raise PACSConnectionError(
            f"No PACS endpoint available for operation '{operation_type}'. "
            "All circuit breakers are open."
        )

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def mark_endpoint_failed(self, endpoint: PACSEndpoint, error: Exception) -> None:
        """Record a failure against *endpoint* and update its circuit breaker."""
        health = self._health.get(endpoint.endpoint_id)
        if health is None:
            return

        health.failure_count += 1
        health.last_failure_time = datetime.utcnow()
        health.circuit_breaker.record_failure()

        if endpoint.is_primary:
            logger.warning(
                "Primary PACS endpoint %s failed (%s). Failing over to backup endpoints.",
                endpoint.endpoint_id,
                type(error).__name__,
            )
        else:
            logger.error(
                "Backup PACS endpoint %s failed: %s",
                endpoint.endpoint_id,
                error,
            )

    def mark_endpoint_success(self, endpoint: PACSEndpoint) -> None:
        """Record a successful operation against *endpoint*."""
        health = self._health.get(endpoint.endpoint_id)
        if health is None:
            return

        health.is_available = True
        health.last_check_time = datetime.utcnow()
        health.circuit_breaker.record_success()

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_available_endpoints(self) -> List[PACSEndpoint]:
        """Return endpoints whose circuit is not OPEN."""
        return [
            h.endpoint
            for h in self._health.values()
            if h.circuit_breaker.state != CircuitState.OPEN
        ]

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Return a per-endpoint health snapshot suitable for a monitoring dashboard."""
        return {
            ep_id: {
                "is_available": h.is_available,
                "failure_count": h.failure_count,
                "circuit_state": h.circuit_breaker.state.value,
                "last_check": h.last_check_time.isoformat() if h.last_check_time else None,
            }
            for ep_id, h in self._health.items()
        }

    def add_endpoint(self, endpoint: PACSEndpoint) -> None:
        """Register a new endpoint at runtime (e.g., after configuration reload)."""
        if endpoint.endpoint_id in self._health:
            logger.debug("Endpoint %s already registered; skipping.", endpoint.endpoint_id)
            return
        self._health[endpoint.endpoint_id] = EndpointHealth(endpoint=endpoint)
        logger.info("Added PACS endpoint %s.", endpoint.endpoint_id)

    def remove_endpoint(self, endpoint_id: str) -> None:
        """Deregister an endpoint (e.g., when decommissioning a PACS node)."""
        removed = self._health.pop(endpoint_id, None)
        if removed:
            logger.info("Removed PACS endpoint %s.", endpoint_id)
        else:
            logger.debug("Endpoint %s not found; nothing removed.", endpoint_id)
