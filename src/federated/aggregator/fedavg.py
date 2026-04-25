"""FedAvg (Federated Averaging) aggregation algorithm."""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.federated.common.data_models import ClientUpdate

from .base import BaseAggregator

logger = logging.getLogger(__name__)


class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging aggregator.

    Computes weighted average of client model updates based on dataset sizes.

    Algorithm:
        aggregated_update = Σ(w_i * Δw_i) / Σ(w_i)
        where w_i = dataset_size_i

    Correctness Properties:
        - Invariant: aggregated_update = weighted average of client updates
        - Metamorphic: Order of client updates doesn't affect result
        - Model-Based: Compare against simple averaging baseline
    """

    def __init__(self):
        super().__init__()
        self.algorithm_name = "FedAvg"
        logger.info("FedAvg aggregator initialized")

    def aggregate(
        self, client_updates: List[ClientUpdate], global_model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using weighted averaging.

        Args:
            client_updates: List of updates from clients

        Returns:
            aggregated_update: Weighted average of client gradients

        Raises:
            ValueError: If client_updates is empty
        """
        if not client_updates:
            raise ValueError("Cannot aggregate empty list of client updates")

        # Extract weights (dataset sizes)
        weights = [update.dataset_size for update in client_updates]
        total_weight = sum(weights)

        if total_weight == 0:
            raise ValueError("Total dataset size is zero")

        # Initialize aggregated update with zeros
        aggregated_update = {}
        param_names = client_updates[0].gradients.keys()

        for param_name in param_names:
            # Weighted sum of gradients
            weighted_sum = sum(
                (weights[i] / total_weight) * client_updates[i].gradients[param_name]
                for i in range(len(client_updates))
            )
            aggregated_update[param_name] = weighted_sum

        return aggregated_update

    def aggregate_models(
        self, client_models: List[Dict[str, torch.Tensor]], client_weights: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate full model state dicts (alternative interface).

        Args:
            client_models: List of model state_dicts
            client_weights: List of weights (typically dataset sizes)

        Returns:
            aggregated_model: Weighted average model
        """
        if not client_models:
            raise ValueError("Cannot aggregate empty list of models")

        if len(client_models) != len(client_weights):
            raise ValueError("Number of models must match number of weights")

        total_weight = sum(client_weights)
        if total_weight == 0:
            raise ValueError("Total weight is zero")

        # Initialize aggregated model
        aggregated_model = {}
        param_names = client_models[0].keys()

        for param_name in param_names:
            # Weighted average of parameters
            weighted_sum = sum(
                (client_weights[i] / total_weight) * client_models[i][param_name]
                for i in range(len(client_models))
            )
            aggregated_model[param_name] = weighted_sum

        return aggregated_model
