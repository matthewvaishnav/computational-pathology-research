"""
Asynchronous training support for federated learning.

Provides semi-synchronous and fully asynchronous training modes with
staleness-aware weighting and dynamic timeout adjustment.
"""

from .async_coordinator import AsyncCoordinator, ClientUpdate
from .staleness_weighting import StalenessWeighting, UpdateMetadata
from .sync_mode import SyncConfig, SynchronizationMode
from .timeout_manager import TimeoutManager

__all__ = [
    "SynchronizationMode",
    "SyncConfig",
    "StalenessWeighting",
    "UpdateMetadata",
    "TimeoutManager",
    "AsyncCoordinator",
    "ClientUpdate",
]
