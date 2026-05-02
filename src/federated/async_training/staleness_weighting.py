"""
Staleness-aware weighting for asynchronous federated learning.

Reduces impact of outdated gradients based on model version difference.
"""

import logging
import math
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UpdateMetadata:
    """Metadata for client update."""
    
    client_id: str
    model_version: int
    """Model version used for local training."""
    
    dataset_size: int
    """Number of samples in client dataset."""
    
    timestamp: float
    """Update submission timestamp."""


class StalenessWeighting:
    """
    Staleness-aware weighting for asynchronous updates.
    
    **Validates: Requirements 7.4, 7.7**
    
    Applies exponential decay to outdated gradients:
    weight = base_weight * exp(-alpha * staleness)
    
    where staleness = current_version - update_version
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        min_weight: float = 0.1,
        max_staleness: Optional[int] = None,
    ):
        """
        Initialize staleness weighting.
        
        Args:
            alpha: Decay rate for staleness (higher = faster decay)
            min_weight: Minimum weight for stale updates
            max_staleness: Maximum staleness to accept (None = no limit)
        """
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        
        if not 0 < min_weight <= 1.0:
            raise ValueError(f"min_weight must be in (0, 1], got {min_weight}")
        
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_staleness = max_staleness
        
        logger.info(
            f"Staleness weighting initialized: alpha={alpha}, "
            f"min_weight={min_weight}, max_staleness={max_staleness}"
        )
    
    def calculate_staleness(
        self,
        update_version: int,
        current_version: int,
    ) -> int:
        """
        Calculate staleness of update.
        
        Args:
            update_version: Model version used for update
            current_version: Current global model version
        
        Returns:
            Staleness (version difference)
        
        **Validates: Requirement 7.4**
        """
        staleness = current_version - update_version
        
        if staleness < 0:
            logger.warning(
                f"Update version {update_version} > current version {current_version}"
            )
            return 0
        
        return staleness
    
    def calculate_weight(
        self,
        staleness: int,
        base_weight: float = 1.0,
    ) -> float:
        """
        Calculate staleness-aware weight.
        
        Args:
            staleness: Model version difference
            base_weight: Base weight before staleness adjustment
        
        Returns:
            Adjusted weight
        
        **Validates: Requirements 7.4, 7.7**
        """
        if staleness == 0:
            # No staleness - use full weight
            return base_weight
        
        # Check max staleness threshold
        if self.max_staleness is not None and staleness > self.max_staleness:
            logger.warning(
                f"Update staleness {staleness} exceeds max {self.max_staleness}, "
                f"using min_weight"
            )
            return self.min_weight * base_weight
        
        # Exponential decay: exp(-alpha * staleness)
        decay_factor = math.exp(-self.alpha * staleness)
        
        # Apply minimum weight threshold
        decay_factor = max(decay_factor, self.min_weight)
        
        adjusted_weight = base_weight * decay_factor
        
        logger.debug(
            f"Staleness {staleness}: base_weight={base_weight:.4f}, "
            f"decay={decay_factor:.4f}, adjusted={adjusted_weight:.4f}"
        )
        
        return adjusted_weight
    
    def calculate_weights(
        self,
        updates: List[UpdateMetadata],
        current_version: int,
        use_dataset_size: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate weights for all updates.
        
        Args:
            updates: List of update metadata
            current_version: Current global model version
            use_dataset_size: Weight by dataset size (FedAvg style)
        
        Returns:
            Dict mapping client_id to weight
        
        **Validates: Requirements 7.4, 7.7**
        """
        if not updates:
            return {}
        
        # Calculate base weights (dataset size)
        if use_dataset_size:
            total_samples = sum(u.dataset_size for u in updates)
            base_weights = {
                u.client_id: u.dataset_size / total_samples
                for u in updates
            }
        else:
            # Uniform weights
            base_weights = {u.client_id: 1.0 / len(updates) for u in updates}
        
        # Apply staleness weighting
        staleness_weights = {}
        
        for update in updates:
            staleness = self.calculate_staleness(
                update.model_version,
                current_version,
            )
            
            base_weight = base_weights[update.client_id]
            adjusted_weight = self.calculate_weight(staleness, base_weight)
            
            staleness_weights[update.client_id] = adjusted_weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(staleness_weights.values())
        
        if total_weight > 0:
            normalized_weights = {
                client_id: weight / total_weight
                for client_id, weight in staleness_weights.items()
            }
        else:
            # Fallback to uniform weights
            logger.warning("Total weight is zero, using uniform weights")
            normalized_weights = {u.client_id: 1.0 / len(updates) for u in updates}
        
        logger.info(
            f"Calculated weights for {len(updates)} updates "
            f"(current_version={current_version})"
        )
        
        return normalized_weights
    
    def is_update_acceptable(
        self,
        update_version: int,
        current_version: int,
    ) -> bool:
        """
        Check if update is acceptable based on staleness.
        
        Args:
            update_version: Model version used for update
            current_version: Current global model version
        
        Returns:
            True if update should be accepted
        """
        staleness = self.calculate_staleness(update_version, current_version)
        
        if self.max_staleness is None:
            return True
        
        return staleness <= self.max_staleness
    
    def get_statistics(self, updates: List[UpdateMetadata], current_version: int) -> Dict:
        """Get staleness statistics for updates."""
        if not updates:
            return {
                'num_updates': 0,
                'avg_staleness': 0.0,
                'max_staleness': 0,
                'min_staleness': 0,
            }
        
        staleness_values = [
            self.calculate_staleness(u.model_version, current_version)
            for u in updates
        ]
        
        return {
            'num_updates': len(updates),
            'avg_staleness': sum(staleness_values) / len(staleness_values),
            'max_staleness': max(staleness_values),
            'min_staleness': min(staleness_values),
        }
