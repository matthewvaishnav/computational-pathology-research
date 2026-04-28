"""
Federated learning for multi-institutional computational pathology.

Enables training across hospital silos without sharing patient data.
Uses Flower (flwr) framework with differential privacy support.
"""

from .aggregation import ByzantineRobustAggregation, FedAvgPathology, FedProxPathology
from .client import PathologyFLClient
from .privacy import DifferentialPrivacyEngine, PrivacyAccountant
from .server import start_federated_server

__all__ = [
    "PathologyFLClient",
    "start_federated_server",
    "FedAvgPathology",
    "FedProxPathology",
    "ByzantineRobustAggregation",
    "DifferentialPrivacyEngine",
    "PrivacyAccountant",
]
