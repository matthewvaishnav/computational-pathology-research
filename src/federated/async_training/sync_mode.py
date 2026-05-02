"""
Synchronization modes for federated learning.

Defines synchronous, semi-synchronous, and fully asynchronous training modes.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class SynchronizationMode(Enum):
    """
    Synchronization modes for federated learning.
    
    **Validates: Requirement 7.1**
    """
    
    SYNCHRONOUS = "synchronous"
    """Wait for all clients before aggregation."""
    
    SEMI_SYNCHRONOUS = "semi_synchronous"
    """Wait for minimum percentage of clients."""
    
    FULLY_ASYNCHRONOUS = "fully_asynchronous"
    """Aggregate updates as they arrive."""


@dataclass
class SyncConfig:
    """
    Configuration for synchronization mode.
    
    **Validates: Requirements 7.1, 7.2, 7.6**
    """
    
    mode: SynchronizationMode = SynchronizationMode.SYNCHRONOUS
    """Synchronization mode."""
    
    min_client_percentage: float = 0.8
    """Minimum percentage of clients for semi-sync (default: 80%)."""
    
    timeout_seconds: float = 600.0
    """Timeout threshold in seconds (default: 10 minutes)."""
    
    enable_staleness_weighting: bool = True
    """Enable staleness-aware weighting for async updates."""
    
    enable_dynamic_timeout: bool = True
    """Enable dynamic timeout adjustment based on client latency."""
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 < self.min_client_percentage <= 1.0:
            raise ValueError(
                f"min_client_percentage must be in (0, 1], got {self.min_client_percentage}"
            )
        
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )
    
    def get_min_clients(self, total_clients: int) -> int:
        """
        Calculate minimum number of clients required.
        
        Args:
            total_clients: Total number of registered clients
        
        Returns:
            Minimum number of clients required for aggregation
        """
        if self.mode == SynchronizationMode.SYNCHRONOUS:
            return total_clients
        elif self.mode == SynchronizationMode.SEMI_SYNCHRONOUS:
            return max(1, int(total_clients * self.min_client_percentage))
        else:  # FULLY_ASYNCHRONOUS
            return 1
    
    def should_wait_for_clients(
        self,
        received_updates: int,
        total_clients: int,
    ) -> bool:
        """
        Determine if coordinator should wait for more clients.
        
        Args:
            received_updates: Number of updates received
            total_clients: Total number of registered clients
        
        Returns:
            True if should wait, False if can proceed with aggregation
        
        **Validates: Requirements 7.2, 7.3**
        """
        min_clients = self.get_min_clients(total_clients)
        
        if self.mode == SynchronizationMode.FULLY_ASYNCHRONOUS:
            # Aggregate as soon as any update arrives
            return received_updates < 1
        else:
            # Wait for minimum threshold
            return received_updates < min_clients
