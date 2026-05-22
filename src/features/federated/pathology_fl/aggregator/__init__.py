"""Aggregation algorithms for federated learning."""

from src.features.federated.pathology_fl.aggregator.fedavg import FedAvgAggregator
from src.features.federated.pathology_fl.aggregator.pathology_fl import PathologyFLAggregator
from src.features.federated.pathology_fl.aggregator.secure import SecureAggregator
from src.features.federated.pathology_fl.aggregator.weighted import ExplicitWeightedAggregator

__all__ = [
    "FedAvgAggregator",
    "PathologyFLAggregator",
    "SecureAggregator",
    "ExplicitWeightedAggregator",
]
