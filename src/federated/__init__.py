"""
Federated learning for multi-institutional computational pathology.

Enables training across hospital silos without sharing patient data.
Uses production-hardened coordinator/client architecture with privacy guarantees.
"""


# Modern production components
def __getattr__(name):
    if name == "TrainingOrchestrator":
        from .coordinator.orchestrator import TrainingOrchestrator

        return TrainingOrchestrator
    elif name == "SecureAggregationProtocol":
        from .privacy.secure_aggregation import SecureAggregationProtocol

        return SecureAggregationProtocol
    elif name == "PrivacyAccountant":
        from .privacy.dp_sgd import PrivacyAccountant

        return PrivacyAccountant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TrainingOrchestrator",
    "SecureAggregationProtocol",
    "PrivacyAccountant",
]
