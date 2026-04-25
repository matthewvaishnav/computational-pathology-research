"""
Base aggregator class for federated learning.

Defines the interface that all aggregation algorithms must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..common.data_models import ClientUpdate


class BaseAggregator(ABC):
    """Base class for federated learning aggregators."""

    def __init__(self):
        """Initialize base aggregator."""
        self.algorithm_name = "BaseAggregator"

    @abstractmethod
    def aggregate(
        self, client_updates: List[ClientUpdate], global_model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates into a single update.

        Args:
            client_updates: List of client updates to aggregate
            global_model: Current global model (optional, used by some algorithms)

        Returns:
            Dictionary mapping parameter names to aggregated tensors
        """
        pass

    def __str__(self) -> str:
        """String representation of aggregator."""
        return f"{self.algorithm_name}Aggregator"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(algorithm='{self.algorithm_name}')"
