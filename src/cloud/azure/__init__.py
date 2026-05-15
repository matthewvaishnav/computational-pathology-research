"""
Azure Cloud Integration Module

This module provides comprehensive Azure cloud services integration for HistoCore,
including Health Data Services, Blob Storage, Functions, and Monitor integration.
"""

from .health_data_services import AzureHealthDataServices
from .blob_storage import AzureBlobStorageConnector
from .functions import AzureFunctionsIntegration
from .monitor import AzureMonitorIntegration

__all__ = [
    "AzureHealthDataServices",
    "AzureBlobStorageConnector",
    "AzureFunctionsIntegration",
    "AzureMonitorIntegration",
]
