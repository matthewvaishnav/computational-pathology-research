"""Aggregation algorithms for federated learning."""

from src.federated.aggregator.fedavg import FedAvgAggregator
from src.federated.aggregator.pathology_fl import PathologyFLAggregator
from src.federated.aggregator.secure import SecureAggregator

__all__ = [
    "FedAvgAggregator",
    "PathologyFLAggregator",
    "SecureAggregator",
]
