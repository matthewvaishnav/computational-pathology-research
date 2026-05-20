"""gRPC communication layer for federated learning."""

from .auth import AuthenticatedFLServer
from .grpc_client import FLClientTrainer
from .grpc_server import SecureFLServer
from .tls_utils import TLSManager

__all__ = [
    "FLClientTrainer",
    "SecureFLServer",
    "AuthenticatedFLServer",
    "TLSManager",
]
