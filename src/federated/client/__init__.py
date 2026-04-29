"""FL Client - Hospital-side component for local training."""

from src.federated.client.hospital_client import HospitalClient
from src.federated.client.trainer import FederatedTrainer

__all__ = [
    "FederatedTrainer",
    "HospitalClient",
]
