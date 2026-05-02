"""
Asynchronous training support for federated learning.

Provides semi-synchronous and fully asynchronous training modes with
staleness-aware weighting and dynamic timeout adjustment.
"""

from .sync_mode import SynchronizationMode, SyncConfig
from .staleness_weighting import StalenessWeighting, UpdateMetadata
from .timeout_manager import TimeoutManager
from .async_coordinator import AsyncCoordinator, ClientUpdate

__all__ = [
    'SynchronizationMode',
    'SyncConfig',
    'StalenessWeighting',
    'UpdateMetadata',
    'TimeoutManager',
    'AsyncCoordinator',
    'ClientUpdate',
]
