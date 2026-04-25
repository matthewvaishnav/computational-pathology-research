"""FL Client - Hospital-side component for local training."""

from src.federated.client.trainer import LocalTrainer
from src.federated.client.pacs_connector import FLPACSConnector
from src.federated.client.resource_manager import ResourceManager

__all__ = [
    "LocalTrainer",
    "FLPACSConnector",
    "ResourceManager",
]
