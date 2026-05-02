"""
Federated learning for multi-institutional computational pathology.

Enables training across hospital silos without sharing patient data.
Uses Flower (flwr) framework with differential privacy support.
"""

# Lazy imports to avoid protobuf issues on Python 3.14
def __getattr__(name):
    if name == "HospitalClient":
        from .client import HospitalClient
        return HospitalClient
    elif name == "FederatedTrainer":
        from .client import FederatedTrainer
        return FederatedTrainer
    elif name == "start_federated_server":
        from .server import start_federated_server
        return start_federated_server
    elif name == "FedAvgPathology":
        from .aggregation import FedAvgPathology
        return FedAvgPathology
    elif name == "FedProxPathology":
        from .aggregation import FedProxPathology
        return FedProxPathology
    elif name == "ByzantineRobustAggregation":
        from .aggregation import ByzantineRobustAggregation
        return ByzantineRobustAggregation
    elif name == "DifferentialPrivacyEngine":
        from .privacy import DifferentialPrivacyEngine
        return DifferentialPrivacyEngine
    elif name == "PrivacyAccountant":
        from .privacy import PrivacyAccountant
        return PrivacyAccountant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HospitalClient",
    "FederatedTrainer",
    "start_federated_server",
    "FedAvgPathology",
    "FedProxPathology",
    "ByzantineRobustAggregation",
    "DifferentialPrivacyEngine",
    "PrivacyAccountant",
]
