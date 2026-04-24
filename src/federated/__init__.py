"""
Federated learning for multi-institutional computational pathology.

Enables training across hospital silos without sharing patient data.
Uses Flower (flwr) framework with differential privacy support.
"""

from .client import PathologyFLClient
from .server import start_federated_server
from .aggregation import FedAvgPathology, FedProxPathology, ByzantineRobustAggregation
from .privacy import DifferentialPrivacyEngine, PrivacyAccountant

__all__ = [
    "PathologyFLClient",
    "start_federated_server",
    "FedAvgPathology",
    "FedProxPathology",
    "ByzantineRobustAggregation",
    "DifferentialPrivacyEngine",
    "PrivacyAccountant",
]
