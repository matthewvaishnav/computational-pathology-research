"""
Secure aggregator using homomorphic encryption.

Implements secure multi-party computation for federated learning
where the central server cannot see individual hospital updates.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..common.data_models import ClientUpdate
from ..privacy.secure_aggregation import SecureAggregationProtocol
from .base import BaseAggregator

logger = logging.getLogger(__name__)


class SecureAggregator(BaseAggregator):
    """
    Secure aggregator using homomorphic encryption.
    
    Prevents central server from accessing individual hospital updates
    by using homomorphic encryption for secure multi-party computation.
    
    Properties:
        - Central server cannot decrypt individual updates
        - Only aggregated result is revealed
        - Handles hospital dropouts gracefully
        - Maintains accuracy equivalent to plaintext aggregation
    """

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        max_workers: int = 4,
        dropout_threshold: float = 0.5,
    ):
        """
        Initialize secure aggregator.

        Args:
            poly_modulus_degree: HE polynomial modulus degree (higher = more secure but slower)
            max_workers: Maximum worker threads for parallel operations
            dropout_threshold: Minimum fraction of clients required (0.5 = 50%)
        """
        super().__init__()
        self.algorithm_name = "SecureAggregation"
        
        self.poly_modulus_degree = poly_modulus_degree
        self.max_workers = max_workers
        self.dropout_threshold = dropout_threshold
        
        # Initialize secure aggregation protocol
        self.protocol = SecureAggregationProtocol(
            coordinator_id="central_server",
            poly_modulus_degree=poly_modulus_degree,
            max_workers=max_workers,
        )
        
        # Track expected clients for dropout handling
        self.expected_clients: Optional[List[str]] = None
        self.min_clients_required: int = 2
        
        logger.info(
            f"Secure aggregator initialized: poly_degree={poly_modulus_degree}, "
            f"dropout_threshold={dropout_threshold}"
        )

    def setup_round(self, expected_clients: List[str]) -> bytes:
        """
        Set up new aggregation round.
        
        Args:
            expected_clients: List of expected client IDs
            
        Returns:
            Serialized public context for clients
        """
        self.expected_clients = expected_clients
        self.min_clients_required = max(2, int(len(expected_clients) * self.dropout_threshold))
        
        logger.info(
            f"Setup round: expecting {len(expected_clients)} clients, "
            f"minimum required: {self.min_clients_required}"
        )
        
        return self.protocol.setup_round()

    def aggregate(
        self, client_updates: List[ClientUpdate], global_model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates securely using homomorphic encryption.

        Args:
            client_updates: List of client updates to aggregate
            global_model: Current global model (unused, for interface compatibility)

        Returns:
            Dictionary mapping parameter names to aggregated tensors
            
        Raises:
            ValueError: If insufficient clients or empty updates
        """
        if not client_updates:
            raise ValueError("Cannot aggregate empty list of client updates")
        
        # Check dropout threshold
        if self.expected_clients is not None:
            num_received = len(client_updates)
            if num_received < self.min_clients_required:
                logger.warning(
                    f"Dropout detected: received {num_received}/{len(self.expected_clients)} clients, "
                    f"minimum required: {self.min_clients_required}"
                )
                raise ValueError(
                    f"Insufficient clients: received {num_received}, "
                    f"minimum required {self.min_clients_required}"
                )
        
        logger.info(f"Aggregating {len(client_updates)} client updates securely")
        
        # Prepare client updates for secure aggregation
        # Format: Dict[client_id, (gradients, weight)]
        client_data = {}
        total_dataset_size = sum(update.dataset_size for update in client_updates)
        
        if total_dataset_size == 0:
            raise ValueError("Total dataset size is zero - cannot compute weights")
        
        for update in client_updates:
            # Compute weight based on dataset size
            weight = update.dataset_size / total_dataset_size
            client_data[update.client_id] = (update.gradients, weight)
        
        # Perform secure aggregation
        aggregated_gradients = self.protocol.aggregate_client_updates(client_data)
        
        logger.info(
            f"Secure aggregation completed: {len(aggregated_gradients)} parameters aggregated"
        )
        
        return aggregated_gradients

    def get_public_context(self) -> bytes:
        """
        Get public encryption context for clients.
        
        Returns:
            Serialized public context
        """
        return self.protocol.get_public_context()

    def handle_dropout(self, failed_clients: List[str]):
        """
        Handle client dropouts gracefully.
        
        Args:
            failed_clients: List of client IDs that failed to respond
        """
        if self.expected_clients is None:
            logger.warning("Cannot handle dropout: no expected clients set")
            return
        
        remaining_clients = [
            c for c in self.expected_clients if c not in failed_clients
        ]
        
        logger.info(
            f"Handling dropout: {len(failed_clients)} clients failed, "
            f"{len(remaining_clients)} remaining"
        )
        
        # Update minimum required based on remaining clients
        self.min_clients_required = max(
            2, int(len(remaining_clients) * self.dropout_threshold)
        )

    def __str__(self) -> str:
        """String representation."""
        return f"SecureAggregator(poly_degree={self.poly_modulus_degree})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"SecureAggregator(poly_modulus_degree={self.poly_modulus_degree}, "
            f"max_workers={self.max_workers}, dropout_threshold={self.dropout_threshold})"
        )
