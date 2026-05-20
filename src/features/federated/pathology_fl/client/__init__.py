"""FL Client - Hospital-side component for local training."""

# Lazy imports to avoid protobuf issues
__all__ = [
    "FederatedTrainer",
    "HospitalClient",
    "PACSConnector",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies and protobuf issues."""
    if name == "HospitalClient":
        from src.features.federated.pathology_fl.client.hospital_client import HospitalClient

        return HospitalClient
    elif name == "FederatedTrainer":
        from src.features.federated.pathology_fl.client.trainer import FederatedTrainer

        return FederatedTrainer
    elif name == "PACSConnector":
        from src.features.federated.pathology_fl.client.pacs_connector import PACSConnector

        return PACSConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
