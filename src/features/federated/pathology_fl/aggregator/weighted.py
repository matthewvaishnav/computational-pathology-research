"""Explicit weighted aggregation adapter.

This aggregator consumes externally computed institutional weights, such as
weights produced by the experimental FAIR-WEIGHTS-H engine. It does not compute
weights internally and does not replace FedAvg defaults.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.features.federated.pathology_fl.common.data_models import ClientUpdate

from .base import BaseAggregator


class ExplicitWeightedAggregator(BaseAggregator):
    """Aggregate client updates using an explicit client_id -> weight mapping."""

    def __init__(self, client_weights: Dict[str, float]):
        super().__init__()
        self.algorithm_name = "ExplicitWeighted"
        self.client_weights = dict(client_weights)
        self._validate_weights()

    def aggregate(
        self, client_updates: List[ClientUpdate], global_model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        if not client_updates:
            raise ValueError("Cannot aggregate empty list of client updates")

        update_ids = {update.client_id for update in client_updates}
        missing = sorted(update_ids - set(self.client_weights))
        if missing:
            raise ValueError(f"Missing explicit weights for clients: {missing}")

        selected_weights = {client_id: self.client_weights[client_id] for client_id in update_ids}
        total_weight = sum(selected_weights.values())
        if total_weight <= 0.0:
            raise ValueError("Total selected client weight must be positive")

        aggregated_update: Dict[str, torch.Tensor] = {}
        param_names = client_updates[0].gradients.keys()

        for param_name in param_names:
            weighted_sum = sum(
                (selected_weights[update.client_id] / total_weight) * update.gradients[param_name]
                for update in client_updates
            )
            aggregated_update[param_name] = weighted_sum

        return aggregated_update

    def _validate_weights(self) -> None:
        if not self.client_weights:
            raise ValueError("client_weights cannot be empty")
        for client_id, weight in self.client_weights.items():
            if not client_id:
                raise ValueError("client weight IDs must be non-empty")
            if weight < 0.0:
                raise ValueError("client weights must be non-negative")
