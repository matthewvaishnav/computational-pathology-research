"""
Client failure detection and recovery system for federated learning.

Handles network disconnections, timeouts, and client dropouts while maintaining
training progress with remaining clients.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Callable
import threading

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of client failures."""
    NETWORK_TIMEOUT = "network_timeout"
    CONNECTION_LOST = "connection_lost"
    TRAINING_TIMEOUT = "training_timeout"
    INVALID_UPDATE = "invalid_update"
    AUTHENTICATION_FAILED = "authentication_failed"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNKNOWN_ERROR = "unknown_error"


class ClientStatus(Enum):
    """Client status states."""
    ACTIVE = "active"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    RECOVERING = "recovering"
    BLACKLISTED = "blacklisted"


@dataclass
class FailureEvent:
    """Records a client failure event."""
    client_id: str
    failure_type: FailureType
    timestamp: datetime
    round_id: int
    error_message: str
    retry_count: int = 0
    recovery_attempted: bool = False


@dataclass
class ClientHealthMetrics:
    """Tracks client health and performance metrics."""
    client_id: str
    last_seen: datetime
    status: ClientStatus = ClientStatus.ACTIVE
    consecutive_failures: int = 0
    total_failures: int = 0
    success_rate: float = 1.0
    average_response_time: float = 0.0
    last_failure: Optional[FailureEvent] = None
    failure_history: List[FailureEvent] = field(default_factory=list)


class ClientFailureHandler:
    """
    Handles client failures in federated learning.
    
    Responsibilities:
    - Detect client failures (timeouts, disconnections)
    - Implement recovery mechanisms
    - Maintain training progress with remaining clients
    - Provide failure notifications and logging
    - Manage client blacklisting and recovery
    """
    
    def __init__(
        self,
        client_timeout: float = 300.0,  # 5 minutes
        max_retry_attempts: int = 3,
        failure_threshold: int = 5,  # consecutive failures before blacklist
        recovery_timeout: float = 60.0,  # 1 minute recovery timeout
        min_clients_threshold: int = 2,  # minimum clients to continue training
        notification_callback: Optional[Callable] = None,
    ):
        """
        Initialize failure handler.
        
        Args:
            client_timeout: Timeout for client responses (seconds)
            max_retry_attempts: Maximum retry attempts per failure
            failure_threshold: Consecutive failures before blacklisting
            recovery_timeout: Timeout for recovery attempts
            min_clients_threshold: Minimum clients needed to continue training
            notification_callback: Callback for failure notifications
        """
        self.client_timeout = client_timeout
        self.max_retry_attempts = max_retry_attempts
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.min_clients_threshold = min_clients_threshold
        self.notification_callback = notification_callback
        
        # Client tracking
        self.client_metrics: Dict[str, ClientHealthMetrics] = {}
        self.active_clients: Set[str] = set()
        self.failed_clients: Set[str] = set()
        self.blacklisted_clients: Set[str] = set()
        
        # Round tracking
        self.current_round_id: int = 0
        self.round_start_time: Optional[datetime] = None
        self.expected_clients: Set[str] = set()
        self.received_updates: Set[str] = set()
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        logger.info("Client failure handler initialized")
    
    def register_client(self, client_id: str) -> None:
        """
        Register a new client for monitoring.
        
        Args:
            client_id: Unique client identifier
        """
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = ClientHealthMetrics(
                client_id=client_id,
                last_seen=datetime.now(),
            )
            self.active_clients.add(client_id)
            logger.info(f"Registered client {client_id} for failure monitoring")
    
    def start_round_monitoring(self, round_id: int, expected_clients: List[str]) -> None:
        """
        Start monitoring for a new training round.
        
        Args:
            round_id: Current round identifier
            expected_clients: List of clients expected to participate
        """
        self.current_round_id = round_id
        self.round_start_time = datetime.now()
        self.expected_clients = set(expected_clients)
        self.received_updates = set()
        
        # Start monitoring thread if not active
        if not self.monitoring_active:
            self.start_monitoring()
        
        logger.info(f"Started round {round_id} monitoring for {len(expected_clients)} clients")
    
    def record_client_activity(self, client_id: str, response_time: float = 0.0) -> None:
        """
        Record successful client activity.
        
        Args:
            client_id: Client identifier
            response_time: Response time in seconds
        """
        if client_id not in self.client_metrics:
            self.register_client(client_id)
        
        metrics = self.client_metrics[client_id]
        metrics.last_seen = datetime.now()
        metrics.status = ClientStatus.ACTIVE
        metrics.consecutive_failures = 0  # Reset on success
        
        # Update success rate and response time
        if metrics.total_failures > 0:
            total_attempts = metrics.total_failures + 1  # +1 for this success
            metrics.success_rate = 1.0 / total_attempts
        
        if response_time > 0:
            if metrics.average_response_time == 0:
                metrics.average_response_time = response_time
            else:
                # Exponential moving average
                metrics.average_response_time = 0.8 * metrics.average_response_time + 0.2 * response_time
        
        # Move to active clients if recovering
        if client_id in self.failed_clients:
            self.failed_clients.remove(client_id)
            self.active_clients.add(client_id)
            logger.info(f"Client {client_id} recovered and marked as active")
    
    def record_client_update(self, client_id: str, round_id: int) -> None:
        """
        Record that a client submitted an update for the current round.
        
        Args:
            client_id: Client identifier
            round_id: Round identifier
        """
        if round_id == self.current_round_id:
            self.received_updates.add(client_id)
            self.record_client_activity(client_id)
            logger.debug(f"Recorded update from client {client_id} for round {round_id}")
    
    def detect_client_failure(
        self,
        client_id: str,
        failure_type: FailureType,
        error_message: str = "",
        round_id: Optional[int] = None,
    ) -> FailureEvent:
        """
        Detect and record a client failure.
        
        Args:
            client_id: Failed client identifier
            failure_type: Type of failure
            error_message: Error description
            round_id: Round where failure occurred
            
        Returns:
            FailureEvent object
        """
        if client_id not in self.client_metrics:
            self.register_client(client_id)
        
        metrics = self.client_metrics[client_id]
        failure_event = FailureEvent(
            client_id=client_id,
            failure_type=failure_type,
            timestamp=datetime.now(),
            round_id=round_id or self.current_round_id,
            error_message=error_message,
        )
        
        # Update metrics
        metrics.consecutive_failures += 1
        metrics.total_failures += 1
        metrics.last_failure = failure_event
        metrics.failure_history.append(failure_event)
        metrics.status = ClientStatus.FAILED
        
        # Update success rate
        total_attempts = metrics.total_failures + max(1, len(self.received_updates))
        metrics.success_rate = len(self.received_updates) / total_attempts
        
        # Move to failed clients
        if client_id in self.active_clients:
            self.active_clients.remove(client_id)
        self.failed_clients.add(client_id)
        
        # Check if client should be blacklisted
        if metrics.consecutive_failures >= self.failure_threshold:
            self.blacklist_client(client_id, "Exceeded failure threshold")
        
        logger.warning(f"Client failure detected: {client_id} - {failure_type.value}: {error_message}")
        
        # Send notification if callback provided
        if self.notification_callback:
            try:
                self.notification_callback(failure_event)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")
        
        return failure_event
    
    def blacklist_client(self, client_id: str, reason: str) -> None:
        """
        Blacklist a client due to repeated failures.
        
        Args:
            client_id: Client to blacklist
            reason: Reason for blacklisting
        """
        if client_id in self.client_metrics:
            self.client_metrics[client_id].status = ClientStatus.BLACKLISTED
        
        # Remove from active sets
        self.active_clients.discard(client_id)
        self.failed_clients.discard(client_id)
        self.blacklisted_clients.add(client_id)
        
        logger.error(f"Client {client_id} blacklisted: {reason}")
    
    def attempt_client_recovery(self, client_id: str) -> bool:
        """
        Attempt to recover a failed client.
        
        Args:
            client_id: Client to recover
            
        Returns:
            True if recovery initiated successfully
        """
        if client_id not in self.client_metrics:
            return False
        
        metrics = self.client_metrics[client_id]
        
        # Don't attempt recovery for blacklisted clients
        if client_id in self.blacklisted_clients:
            logger.warning(f"Cannot recover blacklisted client {client_id}")
            return False
        
        # Check if already recovering
        if metrics.status == ClientStatus.RECOVERING:
            logger.debug(f"Client {client_id} already in recovery")
            return True
        
        # Mark as recovering
        metrics.status = ClientStatus.RECOVERING
        
        # Record recovery attempt
        if metrics.last_failure:
            metrics.last_failure.recovery_attempted = True
            metrics.last_failure.retry_count += 1
        
        logger.info(f"Attempting recovery for client {client_id}")
        
        # In a real implementation, this would trigger actual recovery actions
        # such as sending ping messages, reconnection attempts, etc.
        
        return True
    
    def check_round_timeouts(self) -> List[str]:
        """
        Check for clients that have timed out in the current round.
        
        Returns:
            List of timed out client IDs
        """
        if not self.round_start_time:
            return []
        
        timed_out_clients = []
        current_time = datetime.now()
        timeout_threshold = self.round_start_time + timedelta(seconds=self.client_timeout)
        
        if current_time > timeout_threshold:
            # Check which expected clients haven't submitted updates
            missing_clients = self.expected_clients - self.received_updates
            
            for client_id in missing_clients:
                if client_id not in self.blacklisted_clients:
                    self.detect_client_failure(
                        client_id=client_id,
                        failure_type=FailureType.TRAINING_TIMEOUT,
                        error_message=f"No update received within {self.client_timeout}s",
                        round_id=self.current_round_id,
                    )
                    timed_out_clients.append(client_id)
        
        return timed_out_clients
    
    def can_continue_training(self) -> bool:
        """
        Check if training can continue with remaining active clients.
        
        Returns:
            True if sufficient clients remain active
        """
        active_count = len(self.active_clients)
        return active_count >= self.min_clients_threshold
    
    def get_active_clients(self) -> List[str]:
        """
        Get list of currently active clients.
        
        Returns:
            List of active client IDs
        """
        return list(self.active_clients)
    
    def get_failed_clients(self) -> List[str]:
        """
        Get list of currently failed clients.
        
        Returns:
            List of failed client IDs
        """
        return list(self.failed_clients)
    
    def get_client_status(self, client_id: str) -> Optional[ClientHealthMetrics]:
        """
        Get health metrics for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            ClientHealthMetrics or None if not found
        """
        return self.client_metrics.get(client_id)
    
    def get_round_progress(self) -> Dict[str, any]:
        """
        Get current round progress information.
        
        Returns:
            Dictionary with round progress details
        """
        if not self.round_start_time:
            return {"status": "no_active_round"}
        
        elapsed_time = (datetime.now() - self.round_start_time).total_seconds()
        remaining_time = max(0, self.client_timeout - elapsed_time)
        
        return {
            "round_id": self.current_round_id,
            "expected_clients": len(self.expected_clients),
            "received_updates": len(self.received_updates),
            "active_clients": len(self.active_clients),
            "failed_clients": len(self.failed_clients),
            "elapsed_time": elapsed_time,
            "remaining_time": remaining_time,
            "completion_rate": len(self.received_updates) / len(self.expected_clients) if self.expected_clients else 0,
            "can_continue": self.can_continue_training(),
        }
    
    def start_monitoring(self) -> None:
        """Start the background monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started client failure monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop the background monitoring thread."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            self.monitor_thread = None
        
        logger.info("Stopped client failure monitoring")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active and not self.stop_event.is_set():
            try:
                # Check for timeouts
                timed_out_clients = self.check_round_timeouts()
                
                # Attempt recovery for failed clients
                for client_id in list(self.failed_clients):
                    if client_id not in self.blacklisted_clients:
                        metrics = self.client_metrics.get(client_id)
                        if metrics and metrics.last_failure:
                            if metrics.last_failure.retry_count < self.max_retry_attempts:
                                self.attempt_client_recovery(client_id)
                
                # Sleep before next check
                self.stop_event.wait(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.stop_event.wait(5.0)  # Wait before retrying
    
    def reset_round(self) -> None:
        """Reset state for a new round."""
        self.current_round_id = 0
        self.round_start_time = None
        self.expected_clients.clear()
        self.received_updates.clear()
        logger.debug("Reset round state")
    
    def get_failure_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive failure statistics.
        
        Returns:
            Dictionary with failure statistics
        """
        total_clients = len(self.client_metrics)
        if total_clients == 0:
            return {"total_clients": 0}
        
        total_failures = sum(metrics.total_failures for metrics in self.client_metrics.values())
        avg_success_rate = sum(metrics.success_rate for metrics in self.client_metrics.values()) / total_clients
        
        failure_types = {}
        for metrics in self.client_metrics.values():
            for failure in metrics.failure_history:
                failure_type = failure.failure_type.value
                failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        
        return {
            "total_clients": total_clients,
            "active_clients": len(self.active_clients),
            "failed_clients": len(self.failed_clients),
            "blacklisted_clients": len(self.blacklisted_clients),
            "total_failures": total_failures,
            "average_success_rate": avg_success_rate,
            "failure_types": failure_types,
            "can_continue_training": self.can_continue_training(),
        }