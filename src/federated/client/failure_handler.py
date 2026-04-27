"""
Client failure handling for federated learning.

Provides robust failure detection, recovery mechanisms, and graceful degradation
for federated learning clients in medical AI environments.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of client failures."""
    NETWORK_TIMEOUT = "network_timeout"
    CONNECTION_LOST = "connection_lost"
    COMPUTATION_ERROR = "computation_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    AUTHENTICATION_FAILURE = "authentication_failure"
    DATA_CORRUPTION = "data_corruption"
    UNKNOWN = "unknown"


@dataclass
class FailureEvent:
    """Represents a client failure event."""
    client_id: str
    failure_type: FailureType
    timestamp: float
    error_message: str
    round_number: int
    retry_count: int = 0
    recoverable: bool = True


@dataclass
class ClientStatus:
    """Tracks client status and health."""
    client_id: str
    is_active: bool = True
    last_seen: float = 0.0
    failure_count: int = 0
    consecutive_failures: int = 0
    total_rounds_participated: int = 0
    success_rate: float = 1.0
    current_round: Optional[int] = None
    last_failure: Optional[FailureEvent] = None


class ClientFailureHandler:
    """
    Handles client failures in federated learning.
    
    Features:
    - Failure detection and classification
    - Automatic retry mechanisms
    - Client health monitoring
    - Graceful degradation
    - Recovery notifications
    """
    
    def __init__(
        self,
        timeout_seconds: int = 300,
        max_retries: int = 3,
        min_clients_threshold: int = 2,
        failure_rate_threshold: float = 0.5,
        recovery_cooldown: int = 60
    ):
        """
        Initialize failure handler.
        
        Args:
            timeout_seconds: Client response timeout
            max_retries: Maximum retry attempts per client
            min_clients_threshold: Minimum clients needed to continue training
            failure_rate_threshold: Failure rate to trigger alerts
            recovery_cooldown: Cooldown period before retry (seconds)
        """
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.min_clients_threshold = min_clients_threshold
        self.failure_rate_threshold = failure_rate_threshold
        self.recovery_cooldown = recovery_cooldown
        
        # Client tracking
        self.client_status: Dict[str, ClientStatus] = {}
        self.failure_history: List[FailureEvent] = []
        self.active_clients: Set[str] = set()
        self.failed_clients: Set[str] = set()
        
        # Round tracking
        self.current_round = 0
        self.round_start_time = 0.0
        
        # Callbacks
        self.failure_callbacks = []
        self.recovery_callbacks = []
        
    def register_client(self, client_id: str) -> None:
        """Register a new client."""
        if client_id not in self.client_status:
            self.client_status[client_id] = ClientStatus(
                client_id=client_id,
                last_seen=time.time()
            )
            self.active_clients.add(client_id)
            logger.info(f"Registered client {client_id}")
    
    def start_round(self, round_number: int, participating_clients: List[str]) -> None:
        """Start a new federated learning round."""
        self.current_round = round_number
        self.round_start_time = time.time()
        
        # Update client status for participating clients
        for client_id in participating_clients:
            if client_id in self.client_status:
                self.client_status[client_id].current_round = round_number
                self.client_status[client_id].last_seen = time.time()
        
        logger.info(f"Started round {round_number} with {len(participating_clients)} clients")
    
    def report_client_response(self, client_id: str, success: bool, error_msg: str = "") -> None:
        """Report client response (success or failure)."""
        if client_id not in self.client_status:
            self.register_client(client_id)
        
        status = self.client_status[client_id]
        status.last_seen = time.time()
        
        if success:
            # Reset consecutive failures on success
            status.consecutive_failures = 0
            status.total_rounds_participated += 1
            
            # Update success rate
            total_attempts = status.total_rounds_participated + status.failure_count
            status.success_rate = status.total_rounds_participated / max(total_attempts, 1)
            
            # Move from failed to active if recovering
            if client_id in self.failed_clients:
                self.failed_clients.remove(client_id)
                self.active_clients.add(client_id)
                self._notify_recovery(client_id)
                logger.info(f"Client {client_id} recovered successfully")
        else:
            # Handle failure
            self._handle_client_failure(client_id, error_msg)
    
    def _handle_client_failure(self, client_id: str, error_msg: str) -> None:
        """Handle a client failure."""
        status = self.client_status[client_id]
        status.failure_count += 1
        status.consecutive_failures += 1
        
        # Classify failure type
        failure_type = self._classify_failure(error_msg)
        
        # Create failure event
        failure_event = FailureEvent(
            client_id=client_id,
            failure_type=failure_type,
            timestamp=time.time(),
            error_message=error_msg,
            round_number=self.current_round,
            retry_count=status.consecutive_failures
        )
        
        status.last_failure = failure_event
        self.failure_history.append(failure_event)
        
        # Update success rate
        total_attempts = status.total_rounds_participated + status.failure_count
        status.success_rate = status.total_rounds_participated / max(total_attempts, 1)
        
        # Determine if client should be marked as failed
        if (status.consecutive_failures >= self.max_retries or 
            not failure_event.recoverable):
            self._mark_client_failed(client_id)
        
        # Notify failure
        self._notify_failure(failure_event)
        
        logger.warning(
            f"Client {client_id} failed (attempt {status.consecutive_failures}): "
            f"{failure_type.value} - {error_msg}"
        )
    
    def _classify_failure(self, error_msg: str) -> FailureType:
        """Classify failure type based on error message."""
        error_lower = error_msg.lower()
        
        if "timeout" in error_lower or "timed out" in error_lower:
            return FailureType.NETWORK_TIMEOUT
        elif "connection" in error_lower and ("lost" in error_lower or "refused" in error_lower):
            return FailureType.CONNECTION_LOST
        elif "memory" in error_lower or "resource" in error_lower:
            return FailureType.RESOURCE_EXHAUSTION
        elif "auth" in error_lower or "permission" in error_lower:
            return FailureType.AUTHENTICATION_FAILURE
        elif "corrupt" in error_lower or "invalid" in error_lower:
            return FailureType.DATA_CORRUPTION
        elif "computation" in error_lower or "calculation" in error_lower:
            return FailureType.COMPUTATION_ERROR
        else:
            return FailureType.UNKNOWN
    
    def _mark_client_failed(self, client_id: str) -> None:
        """Mark client as failed and remove from active set."""
        if client_id in self.active_clients:
            self.active_clients.remove(client_id)
        self.failed_clients.add(client_id)
        
        status = self.client_status[client_id]
        status.is_active = False
        
        logger.error(f"Client {client_id} marked as failed after {status.consecutive_failures} attempts")
    
    def check_client_timeouts(self) -> List[str]:
        """Check for client timeouts and mark them as failed."""
        current_time = time.time()
        timed_out_clients = []
        
        for client_id in list(self.active_clients):
            status = self.client_status[client_id]
            
            # Check if client has timed out
            if (status.current_round == self.current_round and 
                current_time - status.last_seen > self.timeout_seconds):
                
                timed_out_clients.append(client_id)
                self._handle_client_failure(client_id, f"Client timeout after {self.timeout_seconds}s")
        
        return timed_out_clients
    
    def get_active_clients(self) -> List[str]:
        """Get list of currently active clients."""
        return list(self.active_clients)
    
    def get_failed_clients(self) -> List[str]:
        """Get list of failed clients."""
        return list(self.failed_clients)
    
    def can_continue_training(self) -> bool:
        """Check if training can continue with current active clients."""
        return len(self.active_clients) >= self.min_clients_threshold
    
    def get_client_health_summary(self) -> Dict[str, Dict]:
        """Get health summary for all clients."""
        summary = {}
        
        for client_id, status in self.client_status.items():
            summary[client_id] = {
                "is_active": status.is_active,
                "success_rate": status.success_rate,
                "total_rounds": status.total_rounds_participated,
                "failure_count": status.failure_count,
                "consecutive_failures": status.consecutive_failures,
                "last_seen": status.last_seen,
                "last_failure_type": status.last_failure.failure_type.value if status.last_failure else None
            }
        
        return summary
    
    def get_failure_statistics(self) -> Dict:
        """Get failure statistics."""
        if not self.failure_history:
            return {"total_failures": 0}
        
        # Count failures by type
        failure_counts = {}
        for failure in self.failure_history:
            failure_type = failure.failure_type.value
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
        
        # Recent failures (last hour)
        recent_threshold = time.time() - 3600
        recent_failures = [f for f in self.failure_history if f.timestamp > recent_threshold]
        
        return {
            "total_failures": len(self.failure_history),
            "recent_failures": len(recent_failures),
            "failure_types": failure_counts,
            "active_clients": len(self.active_clients),
            "failed_clients": len(self.failed_clients),
            "overall_failure_rate": len(self.failed_clients) / max(len(self.client_status), 1)
        }
    
    def attempt_client_recovery(self, client_id: str) -> bool:
        """Attempt to recover a failed client."""
        if client_id not in self.failed_clients:
            return True  # Already active
        
        status = self.client_status[client_id]
        
        # Check cooldown period
        if (status.last_failure and 
            time.time() - status.last_failure.timestamp < self.recovery_cooldown):
            logger.info(f"Client {client_id} still in recovery cooldown")
            return False
        
        # Reset failure counters for recovery attempt
        status.consecutive_failures = 0
        status.is_active = True
        
        # Move back to active clients
        self.failed_clients.remove(client_id)
        self.active_clients.add(client_id)
        
        logger.info(f"Attempting recovery for client {client_id}")
        return True
    
    def add_failure_callback(self, callback):
        """Add callback for failure events."""
        self.failure_callbacks.append(callback)
    
    def add_recovery_callback(self, callback):
        """Add callback for recovery events."""
        self.recovery_callbacks.append(callback)
    
    def _notify_failure(self, failure_event: FailureEvent) -> None:
        """Notify registered callbacks of failure."""
        for callback in self.failure_callbacks:
            try:
                callback(failure_event)
            except Exception as e:
                logger.error(f"Error in failure callback: {e}")
    
    def _notify_recovery(self, client_id: str) -> None:
        """Notify registered callbacks of recovery."""
        for callback in self.recovery_callbacks:
            try:
                callback(client_id)
            except Exception as e:
                logger.error(f"Error in recovery callback: {e}")
    
    def reset_client(self, client_id: str) -> None:
        """Reset client status (for testing or manual recovery)."""
        if client_id in self.client_status:
            status = self.client_status[client_id]
            status.failure_count = 0
            status.consecutive_failures = 0
            status.is_active = True
            status.last_seen = time.time()
            status.last_failure = None
            
            # Move to active clients
            self.failed_clients.discard(client_id)
            self.active_clients.add(client_id)
            
            logger.info(f"Reset client {client_id} status")


class FederatedRoundManager:
    """
    Manages federated learning rounds with failure handling.
    
    Coordinates between the failure handler and aggregation process.
    """
    
    def __init__(self, failure_handler: ClientFailureHandler):
        self.failure_handler = failure_handler
        self.round_results = {}
        
    async def execute_round(
        self, 
        round_number: int, 
        client_tasks: Dict[str, asyncio.Task],
        min_success_rate: float = 0.5
    ) -> Dict[str, any]:
        """
        Execute a federated learning round with failure handling.
        
        Args:
            round_number: Round number
            client_tasks: Dictionary of client_id -> asyncio.Task
            min_success_rate: Minimum success rate to continue
            
        Returns:
            Dictionary with round results and statistics
        """
        participating_clients = list(client_tasks.keys())
        self.failure_handler.start_round(round_number, participating_clients)
        
        # Wait for all tasks with timeout handling
        successful_results = {}
        failed_clients = []
        
        # Monitor tasks and handle timeouts
        timeout_task = asyncio.create_task(
            self._monitor_timeouts(client_tasks)
        )
        
        try:
            # Wait for all tasks to complete
            for client_id, task in client_tasks.items():
                try:
                    result = await asyncio.wait_for(
                        task, 
                        timeout=self.failure_handler.timeout_seconds
                    )
                    successful_results[client_id] = result
                    self.failure_handler.report_client_response(client_id, True)
                    
                except asyncio.TimeoutError:
                    failed_clients.append(client_id)
                    self.failure_handler.report_client_response(
                        client_id, False, "Task timeout"
                    )
                    
                except Exception as e:
                    failed_clients.append(client_id)
                    self.failure_handler.report_client_response(
                        client_id, False, str(e)
                    )
        
        finally:
            timeout_task.cancel()
        
        # Check if we can continue training
        success_rate = len(successful_results) / len(participating_clients)
        can_continue = (
            self.failure_handler.can_continue_training() and 
            success_rate >= min_success_rate
        )
        
        round_result = {
            "round_number": round_number,
            "participating_clients": participating_clients,
            "successful_clients": list(successful_results.keys()),
            "failed_clients": failed_clients,
            "success_rate": success_rate,
            "can_continue": can_continue,
            "results": successful_results,
            "failure_stats": self.failure_handler.get_failure_statistics()
        }
        
        self.round_results[round_number] = round_result
        
        logger.info(
            f"Round {round_number} completed: "
            f"{len(successful_results)}/{len(participating_clients)} clients succeeded "
            f"(success rate: {success_rate:.2%})"
        )
        
        return round_result
    
    async def _monitor_timeouts(self, client_tasks: Dict[str, asyncio.Task]) -> None:
        """Monitor for client timeouts during round execution."""
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            timed_out = self.failure_handler.check_client_timeouts()
            
            # Cancel tasks for timed out clients
            for client_id in timed_out:
                if client_id in client_tasks:
                    client_tasks[client_id].cancel()


if __name__ == "__main__":
    # Demo: Client failure handling
    
    print("=== Client Failure Handler Demo ===\n")
    
    # Create failure handler
    handler = ClientFailureHandler(
        timeout_seconds=30,
        max_retries=2,
        min_clients_threshold=2
    )
    
    # Register clients
    clients = ["hospital_a", "hospital_b", "hospital_c", "hospital_d"]
    for client in clients:
        handler.register_client(client)
    
    print(f"Registered {len(clients)} clients")
    print(f"Active clients: {handler.get_active_clients()}")
    
    # Simulate round with failures
    handler.start_round(1, clients)
    
    # Simulate responses
    handler.report_client_response("hospital_a", True)
    handler.report_client_response("hospital_b", False, "Network timeout occurred")
    handler.report_client_response("hospital_c", True)
    handler.report_client_response("hospital_d", False, "Connection lost to server")
    
    print(f"\nAfter round 1:")
    print(f"Active clients: {handler.get_active_clients()}")
    print(f"Failed clients: {handler.get_failed_clients()}")
    print(f"Can continue training: {handler.can_continue_training()}")
    
    # Show health summary
    print("\nClient health summary:")
    health = handler.get_client_health_summary()
    for client_id, stats in health.items():
        print(f"  {client_id}: success_rate={stats['success_rate']:.2%}, "
              f"failures={stats['failure_count']}")
    
    # Show failure statistics
    print("\nFailure statistics:")
    stats = handler.get_failure_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Demo Complete ===")