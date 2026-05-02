"""
Asynchronous training coordinator.

Orchestrates async/semi-sync training with staleness weighting and timeouts.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import torch

from .sync_mode import SynchronizationMode, SyncConfig
from .staleness_weighting import StalenessWeighting, UpdateMetadata
from .timeout_manager import TimeoutManager

logger = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    """Client update with metadata."""
    
    client_id: str
    model_state: Dict[str, torch.Tensor]
    model_version: int
    dataset_size: int
    timestamp: float
    training_loss: float
    samples_processed: int


class AsyncCoordinator:
    """
    Asynchronous training coordinator.
    
    **Validates: Requirements 7.1-7.7**
    
    Orchestrates async/semi-sync federated learning:
    - Manages synchronization modes
    - Applies staleness-aware weighting
    - Handles client timeouts
    - Coordinates aggregation timing
    """
    
    def __init__(
        self,
        sync_config: SyncConfig,
        staleness_weighting: Optional[StalenessWeighting] = None,
        timeout_manager: Optional[TimeoutManager] = None,
    ):
        """
        Initialize async coordinator.
        
        Args:
            sync_config: Synchronization configuration
            staleness_weighting: Staleness weighting (None = create default)
            timeout_manager: Timeout manager (None = create default)
        """
        self.sync_config = sync_config
        
        # Initialize staleness weighting
        if staleness_weighting is None:
            self.staleness_weighting = StalenessWeighting(
                alpha=0.5,
                min_weight=0.1,
                max_staleness=10,
            )
        else:
            self.staleness_weighting = staleness_weighting
        
        # Initialize timeout manager
        if timeout_manager is None:
            self.timeout_manager = TimeoutManager(
                base_timeout=sync_config.timeout_seconds,
                enable_dynamic=sync_config.enable_dynamic_timeout,
            )
        else:
            self.timeout_manager = timeout_manager
        
        # State tracking
        self.current_version = 0
        self.registered_clients: List[str] = []
        self.pending_updates: Dict[str, ClientUpdate] = {}
        self.round_start_time: Optional[float] = None
        
        logger.info(
            f"Async coordinator initialized: mode={sync_config.mode.value}, "
            f"min_clients={sync_config.min_client_percentage*100:.0f}%"
        )
    
    def register_client(self, client_id: str) -> None:
        """Register a client."""
        if client_id not in self.registered_clients:
            self.registered_clients.append(client_id)
            self.timeout_manager.register_client(client_id)
            logger.info(f"Registered client: {client_id}")
    
    def start_round(self) -> None:
        """
        Start a new training round.
        
        **Validates: Requirement 7.1**
        """
        self.round_start_time = time.time()
        self.pending_updates.clear()
        
        logger.info(
            f"Started round {self.current_version + 1} "
            f"(mode={self.sync_config.mode.value})"
        )
    
    def submit_update(self, update: ClientUpdate) -> None:
        """
        Submit client update.
        
        Args:
            update: Client update with metadata
        
        **Validates: Requirements 7.3, 7.4**
        """
        # Check staleness
        if not self.staleness_weighting.is_update_acceptable(
            update.model_version,
            self.current_version,
        ):
            logger.warning(
                f"Rejected update from {update.client_id}: "
                f"staleness too high (version {update.model_version} vs {self.current_version})"
            )
            return
        
        # Record latency
        if self.round_start_time is not None:
            latency = update.timestamp - self.round_start_time
            self.timeout_manager.record_latency(update.client_id, latency)
        
        # Store update
        self.pending_updates[update.client_id] = update
        
        logger.info(
            f"Received update from {update.client_id} "
            f"(version={update.model_version}, "
            f"staleness={self.current_version - update.model_version})"
        )
    
    def should_aggregate(self) -> bool:
        """
        Check if should proceed with aggregation.
        
        Returns:
            True if ready to aggregate
        
        **Validates: Requirements 7.2, 7.3, 7.6**
        """
        num_updates = len(self.pending_updates)
        num_clients = len(self.registered_clients)
        
        # Check minimum client threshold
        if self.sync_config.should_wait_for_clients(num_updates, num_clients):
            # Check timeout
            if self.round_start_time is not None:
                elapsed = time.time() - self.round_start_time
                timeout = self.timeout_manager.get_global_timeout()
                
                if elapsed > timeout:
                    logger.warning(
                        f"Timeout reached ({elapsed:.1f}s > {timeout:.1f}s), "
                        f"proceeding with {num_updates}/{num_clients} clients"
                    )
                    
                    # Record timeouts for missing clients
                    for client_id in self.registered_clients:
                        if client_id not in self.pending_updates:
                            self.timeout_manager.record_timeout(client_id)
                    
                    return num_updates > 0
                else:
                    return False
            else:
                return False
        
        return True
    
    def aggregate_updates(
        self,
        aggregation_fn: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate pending updates.
        
        Args:
            aggregation_fn: Aggregation function (e.g., FedAvg)
        
        Returns:
            Aggregated model state
        
        **Validates: Requirements 7.4, 7.7**
        """
        if not self.pending_updates:
            raise ValueError("No updates to aggregate")
        
        # Create update metadata
        update_metadata = [
            UpdateMetadata(
                client_id=update.client_id,
                model_version=update.model_version,
                dataset_size=update.dataset_size,
                timestamp=update.timestamp,
            )
            for update in self.pending_updates.values()
        ]
        
        # Calculate staleness-aware weights
        if self.sync_config.enable_staleness_weighting:
            weights = self.staleness_weighting.calculate_weights(
                update_metadata,
                self.current_version,
                use_dataset_size=True,
            )
        else:
            # Uniform weights
            weights = {u.client_id: 1.0 / len(update_metadata) for u in update_metadata}
        
        # Prepare updates for aggregation
        model_states = [u.model_state for u in self.pending_updates.values()]
        weight_list = [weights[u.client_id] for u in self.pending_updates.values()]
        
        # Aggregate
        aggregated_state = aggregation_fn(model_states, weight_list)
        
        # Update version
        self.current_version += 1
        
        # Log statistics
        staleness_stats = self.staleness_weighting.get_statistics(
            update_metadata,
            self.current_version - 1,  # Use previous version for stats
        )
        
        logger.info(
            f"Aggregated {len(self.pending_updates)} updates "
            f"(version {self.current_version}): "
            f"avg_staleness={staleness_stats['avg_staleness']:.2f}, "
            f"max_staleness={staleness_stats['max_staleness']}"
        )
        
        return aggregated_state
    
    def get_round_statistics(self) -> Dict:
        """Get statistics for current round."""
        if self.round_start_time is None:
            return {'round_active': False}
        
        elapsed = time.time() - self.round_start_time
        
        return {
            'round_active': True,
            'current_version': self.current_version,
            'num_updates': len(self.pending_updates),
            'num_clients': len(self.registered_clients),
            'elapsed_time': elapsed,
            'timeout': self.timeout_manager.get_global_timeout(),
            'sync_mode': self.sync_config.mode.value,
        }
    
    def get_client_statistics(self) -> Dict[str, Dict]:
        """Get per-client statistics."""
        return {
            client_id: self.timeout_manager.get_client_statistics(client_id)
            for client_id in self.registered_clients
        }
