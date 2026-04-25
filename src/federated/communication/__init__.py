"""gRPC communication layer for federated learning."""

from src.federated.communication.client import FLClientStub
from src.federated.communication.server import FLServer
from src.federated.communication.tls_setup import create_secure_channel, create_secure_server

__all__ = [
    "FLServer",
    "FLClientStub",
    "create_secure_server",
    "create_secure_channel",
]
