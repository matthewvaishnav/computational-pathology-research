"""
Dynamic timeout management for federated learning.

Tracks client latency and adjusts timeouts adaptively.
"""

import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ClientLatencyStats:
    """Latency statistics for a client."""
    
    client_id: str
    latency_history: deque = field(default_factory=lambda: deque(maxlen=10))
    """Recent latency measurements (seconds)."""
    
    last_update_time: Optional[float] = None
    """Timestamp of last update."""
    
    timeout_count: int = 0
    """Number of timeouts."""
    
    def add_latency(self, latency: float) -> None:
        """Add latency measurement."""
        self.latency_history.append(latency)
        self.last_update_time = time.time()
    
    def get_average_latency(self) -> float:
        """Get average latency."""
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history)
    
    def get_max_latency(self) -> float:
        """Get maximum latency."""
        if not self.latency_history:
            return 0.0
        return max(self.latency_history)
    
    def increment_timeout(self) -> None:
        """Increment timeout counter."""
        self.timeout_count += 1


class TimeoutManager:
    """
    Dynamic timeout management for federated learning.
    
    **Validates: Requirements 7.5, 7.6**
    
    Adjusts timeouts based on client latency patterns:
    - Tracks per-client latency history
    - Calculates adaptive timeouts (mean + k*std)
    - Handles client timeouts gracefully
    """
    
    def __init__(
        self,
        base_timeout: float = 600.0,
        min_timeout: float = 60.0,
        max_timeout: float = 1800.0,
        timeout_multiplier: float = 2.0,
        enable_dynamic: bool = True,
    ):
        """
        Initialize timeout manager.
        
        Args:
            base_timeout: Base timeout in seconds (default: 10 minutes)
            min_timeout: Minimum timeout in seconds
            max_timeout: Maximum timeout in seconds
            timeout_multiplier: Multiplier for adaptive timeout (mean + k*std)
            enable_dynamic: Enable dynamic timeout adjustment
        """
        if base_timeout <= 0:
            raise ValueError(f"base_timeout must be positive, got {base_timeout}")
        
        if not min_timeout <= base_timeout <= max_timeout:
            raise ValueError(
                f"Invalid timeout range: min={min_timeout}, "
                f"base={base_timeout}, max={max_timeout}"
            )
        
        self.base_timeout = base_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.timeout_multiplier = timeout_multiplier
        self.enable_dynamic = enable_dynamic
        
        # Client latency tracking
        self.client_stats: Dict[str, ClientLatencyStats] = {}
        
        logger.info(
            f"Timeout manager initialized: base={base_timeout}s, "
            f"range=[{min_timeout}, {max_timeout}]s, dynamic={enable_dynamic}"
        )
    
    def register_client(self, client_id: str) -> None:
        """Register a new client."""
        if client_id not in self.client_stats:
            self.client_stats[client_id] = ClientLatencyStats(client_id=client_id)
            logger.debug(f"Registered client: {client_id}")
    
    def record_latency(
        self,
        client_id: str,
        latency: float,
    ) -> None:
        """
        Record client latency measurement.
        
        Args:
            client_id: Client identifier
            latency: Latency in seconds
        
        **Validates: Requirement 7.5**
        """
        if client_id not in self.client_stats:
            self.register_client(client_id)
        
        self.client_stats[client_id].add_latency(latency)
        
        logger.debug(
            f"Recorded latency for {client_id}: {latency:.2f}s "
            f"(avg: {self.client_stats[client_id].get_average_latency():.2f}s)"
        )
    
    def record_timeout(self, client_id: str) -> None:
        """
        Record client timeout.
        
        Args:
            client_id: Client identifier
        
        **Validates: Requirement 7.6**
        """
        if client_id not in self.client_stats:
            self.register_client(client_id)
        
        self.client_stats[client_id].increment_timeout()
        
        logger.warning(
            f"Client {client_id} timed out "
            f"(total timeouts: {self.client_stats[client_id].timeout_count})"
        )
    
    def get_timeout(self, client_id: Optional[str] = None) -> float:
        """
        Get timeout for client.
        
        Args:
            client_id: Client identifier (None = global timeout)
        
        Returns:
            Timeout in seconds
        
        **Validates: Requirements 7.5, 7.6**
        """
        if not self.enable_dynamic or client_id is None:
            return self.base_timeout
        
        if client_id not in self.client_stats:
            return self.base_timeout
        
        stats = self.client_stats[client_id]
        
        if not stats.latency_history:
            return self.base_timeout
        
        # Calculate adaptive timeout: mean + k*std
        latencies = list(stats.latency_history)
        mean_latency = sum(latencies) / len(latencies)
        
        if len(latencies) > 1:
            variance = sum((x - mean_latency) ** 2 for x in latencies) / len(latencies)
            std_latency = variance ** 0.5
        else:
            std_latency = 0.0
        
        adaptive_timeout = mean_latency + self.timeout_multiplier * std_latency
        
        # Clamp to [min_timeout, max_timeout]
        timeout = max(self.min_timeout, min(adaptive_timeout, self.max_timeout))
        
        logger.debug(
            f"Adaptive timeout for {client_id}: {timeout:.2f}s "
            f"(mean={mean_latency:.2f}s, std={std_latency:.2f}s)"
        )
        
        return timeout
    
    def get_global_timeout(self) -> float:
        """
        Get global timeout based on all clients.
        
        Returns:
            Global timeout in seconds
        """
        if not self.enable_dynamic or not self.client_stats:
            return self.base_timeout
        
        # Use maximum of all client timeouts
        client_timeouts = [
            self.get_timeout(client_id)
            for client_id in self.client_stats.keys()
        ]
        
        if not client_timeouts:
            return self.base_timeout
        
        global_timeout = max(client_timeouts)
        
        logger.debug(f"Global timeout: {global_timeout:.2f}s")
        
        return global_timeout
    
    def is_client_timed_out(
        self,
        client_id: str,
        start_time: float,
    ) -> bool:
        """
        Check if client has timed out.
        
        Args:
            client_id: Client identifier
            start_time: Round start timestamp
        
        Returns:
            True if client timed out
        
        **Validates: Requirement 7.6**
        """
        elapsed = time.time() - start_time
        timeout = self.get_timeout(client_id)
        
        return elapsed > timeout
    
    def get_client_statistics(self, client_id: str) -> Dict:
        """Get statistics for client."""
        if client_id not in self.client_stats:
            return {
                'registered': False,
            }
        
        stats = self.client_stats[client_id]
        
        return {
            'registered': True,
            'avg_latency': stats.get_average_latency(),
            'max_latency': stats.get_max_latency(),
            'timeout_count': stats.timeout_count,
            'history_size': len(stats.latency_history),
            'adaptive_timeout': self.get_timeout(client_id),
        }
    
    def get_global_statistics(self) -> Dict:
        """Get global statistics."""
        if not self.client_stats:
            return {
                'num_clients': 0,
                'avg_latency': 0.0,
                'max_latency': 0.0,
                'total_timeouts': 0,
            }
        
        all_latencies = []
        total_timeouts = 0
        
        for stats in self.client_stats.values():
            all_latencies.extend(stats.latency_history)
            total_timeouts += stats.timeout_count
        
        return {
            'num_clients': len(self.client_stats),
            'avg_latency': sum(all_latencies) / len(all_latencies) if all_latencies else 0.0,
            'max_latency': max(all_latencies) if all_latencies else 0.0,
            'total_timeouts': total_timeouts,
            'global_timeout': self.get_global_timeout(),
        }
