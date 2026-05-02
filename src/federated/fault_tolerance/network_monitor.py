"""
Network monitoring and partition detection for federated learning.

Detects network partitions, connectivity issues, and coordinator availability.
"""

import asyncio
import logging
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Callable

logger = logging.getLogger(__name__)


class NetworkStatus(Enum):
    """Network connectivity status."""
    
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    PARTITION_DETECTED = "partition_detected"
    UNKNOWN = "unknown"


@dataclass
class NetworkEvent:
    """Network status change event."""
    
    timestamp: datetime
    previous_status: NetworkStatus
    current_status: NetworkStatus
    coordinator_reachable: bool
    latency_ms: Optional[float]
    error_message: Optional[str] = None


class NetworkMonitor:
    """
    Monitors network connectivity to FL coordinator.
    
    **Validates: Requirements 9.3, 9.5**
    
    Features:
    - Heartbeat-based connectivity monitoring
    - Latency tracking
    - Connection quality assessment
    - Status change notifications
    """
    
    def __init__(
        self,
        coordinator_host: str,
        coordinator_port: int,
        heartbeat_interval: float = 10.0,  # seconds
        timeout: float = 5.0,  # seconds
        failure_threshold: int = 3,  # consecutive failures
        status_callback: Optional[Callable] = None,
    ):
        """
        Initialize network monitor.
        
        Args:
            coordinator_host: Coordinator hostname/IP
            coordinator_port: Coordinator port
            heartbeat_interval: Interval between heartbeats
            timeout: Timeout for connectivity checks
            failure_threshold: Consecutive failures before status change
            status_callback: Callback for status changes
        """
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self.status_callback = status_callback
        
        # State tracking
        self.current_status = NetworkStatus.UNKNOWN
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_successful_check = None
        self.last_latency_ms = None
        self.latency_history: List[float] = []
        
        # Event history
        self.events: List[NetworkEvent] = []
        
        # Monitoring control
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"Network monitor initialized for {coordinator_host}:{coordinator_port}"
        )
    
    async def start_monitoring(self) -> None:
        """Start background network monitoring."""
        if self.monitoring_active:
            logger.warning("Network monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Started network monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop background network monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped network monitoring")
    
    async def check_connectivity(self) -> bool:
        """
        Check connectivity to coordinator.
        
        Returns:
            True if coordinator is reachable
        
        **Validates: Requirements 9.3**
        """
        start_time = time.time()
        
        try:
            # Attempt TCP connection
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.coordinator_host, self.coordinator_port),
                timeout=self.timeout
            )
            
            # Close connection
            writer.close()
            await writer.wait_closed()
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            self.last_latency_ms = latency_ms
            self.latency_history.append(latency_ms)
            
            # Keep only recent history
            if len(self.latency_history) > 100:
                self.latency_history = self.latency_history[-100:]
            
            # Update success tracking
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            self.last_successful_check = datetime.now()
            
            # Update status if needed
            self._update_status(NetworkStatus.CONNECTED, True, latency_ms)
            
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Connection timeout to {self.coordinator_host}:{self.coordinator_port}")
            self._handle_connectivity_failure("Connection timeout")
            return False
            
        except Exception as e:
            logger.warning(f"Connection failed: {e}")
            self._handle_connectivity_failure(str(e))
            return False
    
    def _handle_connectivity_failure(self, error_message: str) -> None:
        """Handle connectivity check failure."""
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        
        # Determine new status based on failure count
        if self.consecutive_failures >= self.failure_threshold:
            new_status = NetworkStatus.DISCONNECTED
        elif self.consecutive_failures > 1:
            new_status = NetworkStatus.DEGRADED
        else:
            new_status = self.current_status
        
        self._update_status(new_status, False, None, error_message)
    
    def _update_status(
        self,
        new_status: NetworkStatus,
        coordinator_reachable: bool,
        latency_ms: Optional[float],
        error_message: Optional[str] = None,
    ) -> None:
        """Update network status and trigger notifications."""
        if new_status == self.current_status:
            return
        
        # Create event
        event = NetworkEvent(
            timestamp=datetime.now(),
            previous_status=self.current_status,
            current_status=new_status,
            coordinator_reachable=coordinator_reachable,
            latency_ms=latency_ms,
            error_message=error_message,
        )
        
        self.events.append(event)
        
        # Update status
        previous_status = self.current_status
        self.current_status = new_status
        
        logger.info(
            f"Network status changed: {previous_status.value} → {new_status.value}"
        )
        
        # Notify callback
        if self.status_callback:
            try:
                self.status_callback(event)
            except Exception as e:
                logger.error(f"Status callback failed: {e}")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                await self.check_connectivity()
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    def get_status(self) -> NetworkStatus:
        """Get current network status."""
        return self.current_status
    
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self.current_status == NetworkStatus.CONNECTED
    
    def get_average_latency(self) -> Optional[float]:
        """Get average latency from recent checks."""
        if not self.latency_history:
            return None
        
        return sum(self.latency_history) / len(self.latency_history)
    
    def get_connection_quality(self) -> str:
        """Get connection quality assessment."""
        if not self.is_connected():
            return "disconnected"
        
        avg_latency = self.get_average_latency()
        
        if avg_latency is None:
            return "unknown"
        elif avg_latency < 50:
            return "excellent"
        elif avg_latency < 100:
            return "good"
        elif avg_latency < 200:
            return "fair"
        else:
            return "poor"
    
    def get_statistics(self) -> dict:
        """Get network monitoring statistics."""
        return {
            'current_status': self.current_status.value,
            'is_connected': self.is_connected(),
            'consecutive_failures': self.consecutive_failures,
            'consecutive_successes': self.consecutive_successes,
            'last_successful_check': (
                self.last_successful_check.isoformat()
                if self.last_successful_check
                else None
            ),
            'last_latency_ms': self.last_latency_ms,
            'average_latency_ms': self.get_average_latency(),
            'connection_quality': self.get_connection_quality(),
            'total_events': len(self.events),
        }


class PartitionDetector:
    """
    Detects network partitions in federated learning.
    
    **Validates: Requirements 9.3, 9.5**
    
    Features:
    - Multi-client partition detection
    - Coordinator isolation detection
    - Partition recovery detection
    """
    
    def __init__(
        self,
        network_monitor: NetworkMonitor,
        partition_threshold: timedelta = timedelta(minutes=5),
    ):
        """
        Initialize partition detector.
        
        Args:
            network_monitor: NetworkMonitor instance
            partition_threshold: Time threshold for partition detection
        """
        self.network_monitor = network_monitor
        self.partition_threshold = partition_threshold
        
        self.partition_detected = False
        self.partition_start_time: Optional[datetime] = None
        self.partition_duration: Optional[timedelta] = None
    
    def check_partition(self) -> bool:
        """
        Check if network partition is detected.
        
        Returns:
            True if partition detected
        
        **Validates: Requirements 9.3**
        """
        # Check if disconnected for extended period
        if not self.network_monitor.is_connected():
            if self.network_monitor.last_successful_check:
                time_since_last_success = (
                    datetime.now() - self.network_monitor.last_successful_check
                )
                
                if time_since_last_success > self.partition_threshold:
                    if not self.partition_detected:
                        self._mark_partition_start()
                    return True
        
        # Check if partition recovered
        if self.partition_detected and self.network_monitor.is_connected():
            self._mark_partition_end()
        
        return self.partition_detected
    
    def _mark_partition_start(self) -> None:
        """Mark start of network partition."""
        self.partition_detected = True
        self.partition_start_time = datetime.now()
        
        logger.error(
            f"Network partition detected after {self.partition_threshold.total_seconds()}s "
            f"of disconnection"
        )
    
    def _mark_partition_end(self) -> None:
        """Mark end of network partition."""
        if self.partition_start_time:
            self.partition_duration = datetime.now() - self.partition_start_time
            
            logger.info(
                f"Network partition recovered after "
                f"{self.partition_duration.total_seconds():.1f}s"
            )
        
        self.partition_detected = False
        self.partition_start_time = None
    
    def get_partition_duration(self) -> Optional[timedelta]:
        """Get current or last partition duration."""
        if self.partition_detected and self.partition_start_time:
            return datetime.now() - self.partition_start_time
        
        return self.partition_duration
    
    def should_pause_training(self) -> bool:
        """
        Check if training should pause due to partition.
        
        Returns:
            True if training should pause
        
        **Validates: Requirements 9.5**
        """
        return self.partition_detected
