"""
Azure Blob Storage Connector

Enhanced Azure Blob Storage integration for HistoCore with advanced features:
- Hierarchical namespace support
- Lifecycle management
- Access tier optimization
- Batch operations
- Event-driven processing
- Metadata management
- Security and compliance features
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Iterator, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from azure.storage.blob import (
        BlobServiceClient, 
        BlobClient, 
        ContainerClient,
        BlobProperties,
        AccessTier,
        StandardBlobTier,
        PremiumPageBlobTier
    )
    from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
    from azure.identity import DefaultAzureCredential
    from azure.core.exceptions import AzureError, ResourceNotFoundError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class BlobAccessTier(Enum):
    """Azure Blob Storage access tiers."""
    HOT = "Hot"
    COOL = "Cool"
    ARCHIVE = "Archive"


class BlobType(Enum):
    """Azure Blob types."""
    BLOCK_BLOB = "BlockBlob"
    PAGE_BLOB = "PageBlob"
    APPEND_BLOB = "AppendBlob"


@dataclass
class BlobStorageConfig:
    """Configuration for Azure Blob Storage connector."""
    account_name: str
    container_name: str
    account_key: Optional[str] = None
    connection_string: Optional[str] = None
    use_managed_identity: bool = True
    default_access_tier: BlobAccessTier = BlobAccessTier.HOT
    enable_hierarchical_namespace: bool = True
    enable_versioning: bool = True
    enable_soft_delete: bool = True
    soft_delete_retention_days: int = 7
    max_concurrent_uploads: int = 4
    chunk_size: int = 4 * 1024 * 1024  # 4MB chunks
    timeout: int = 300  # 5 minutes


@dataclass
class BlobMetadata:
    """Blob metadata structure."""
    name: str
    size: int
    content_type: str
    access_tier: str
    last_modified: datetime
    etag: str
    md5_hash: Optional[str] = None
    custom_metadata: Optional[Dict[str, str]] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class UploadResult:
    """Result of blob upload operation."""
    blob_name: str
    success: bool
    etag: Optional[str] = None
    error_message: Optional[str] = None
    upload_time: Optional[float] = None
    bytes_uploaded: Optional[int] = None


class AzureBlobStorageConnector:
    """Enhanced Azure Blob Storage connector for HistoCore."""

    def __init__(self, config: BlobStorageConfig):
        """Initialize Azure Blob Storage connector."""
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure Storage SDK not available. Install with: "
                "pip install azure-storage-blob azure-identity"
            )

        self.config = config
        self.blob_service_client = None
        self.container_client = None
        self.async_blob_service_client = None
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_uploads)
        
        self._initialize_clients()
        logger.info(
            "Azure Blob Storage connector initialized: account=%s, container=%s",
            config.account_name,
            config.container_name
        )

    def _initialize_clients(self) -> None:
        """Initialize Azure Blob Storage clients."""
        try:
            if self.config.connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    self.config.connection_string
                )
            elif self.config.account_key:
                account_url = f"https://{self.config.account_name}.blob.core.windows.net"
                self.blob_service_client = BlobServiceClient(
                    account_url=account_url,
                    credential=self.config.account_key
                )
            elif self.config.use_managed_identity:
                account_url = f"https://{self.config.account_name}.blob.core.windows.net"
                credential = DefaultAzureCredential()
                self.blob_service_client = BlobServiceClient(
                    account_url=account_url,
                    credential=credential
                )
            else:
                raise ValueError("No valid authentication method provided")

            self.container_client = self.blob_service_client.get_container_client(
                self.config.container_name
            )
            
            # Ensure container exists
            self._ensure_container_exists()
            
            logger.info("Azure Blob Storage clients initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Azure Blob Storage clients: %s", e)
            raise

    def _ensure_container_exists(self) -> None:
        """Ensure the container exists, create if it doesn't."""
        try:
            self.container_client.get_container_properties()
            logger.debug("Container %s exists", self.config.container_name)
        except ResourceNotFoundError:
            logger.info("Creating container %s", self.config.container_name)
            self.container_client.create_container()
            
            # Set container properties if hierarchical namespace is enabled
            if self.config.enable_hierarchical_namespace:
                self._configure_container_properties()

    def _configure_container_properties(self) -> None:
        """Configure container properties for optimal performance."""
        try:
            # Set public access level to private
            self.container_client.set_container_access_policy(signed_identifiers={})
            
            # Configure lifecycle management if supported
            self._configure_lifecycle_management()
            
            logger.debug("Container properties configured")
        except Exception as e:
            logger.warning("Failed to configure container properties: %s", e)

    def _configure_lifecycle_management(self) -> None:
        """Configure blob lifecycle management policies."""
        try:
            # Define lifecycle policy to automatically move blobs to cooler tiers
            lifecycle_policy = {
                "rules": [
                    {
                        "name": "histocore-lifecycle",
                        "enabled": True,
                        "type": "Lifecycle",
                        "definition": {
                            "filters": {
                                "blobTypes": ["blockBlob"],
                                "prefixMatch": ["histocore/"]
                            },
                            "actions": {
                                "baseBlob": {
                                    "tierToCool": {"daysAfterModificationGreaterThan": 30},
                                    "tierToArchive": {"daysAfterModificationGreaterThan": 90},
                                    "delete": {"daysAfterModificationGreaterThan": 2555}  # 7 years
                                }
                            }
                        }
                    }
                ]
            }
            
            # Note: Lifecycle management is set at the account level, not container level
            # This would typically be configured via ARM templates or Azure CLI
            logger.debug("Lifecycle management policy defined")
            
        except Exception as e:
            logger.warning("Failed to configure lifecycle management: %s", e)

    def upload_file(
        self, 
        local_path: str, 
        blob_name: str,
        access_tier: Optional[BlobAccessTier] = None,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        overwrite: bool = True
    ) -> UploadResult:
        """Upload file to Azure Blob Storage."""
        start_time = datetime.now()
        
        try:
            if not os.path.exists(local_path):
                return UploadResult(
                    blob_name=blob_name,
                    success=False,
                    error_message=f"Local file not found: {local_path}"
                )

            file_size = os.path.getsize(local_path)
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Prepare upload parameters
            upload_kwargs = {
                "overwrite": overwrite,
                "standard_blob_tier": (access_tier or self.config.default_access_tier).value,
                "timeout": self.config.timeout
            }
            
            if metadata:
                upload_kwargs["metadata"] = metadata
                
            if tags:
                upload_kwargs["tags"] = tags

            # Upload file with progress tracking
            with open(local_path, "rb") as data:
                result = blob_client.upload_blob(data, **upload_kwargs)

            upload_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                "Uploaded %s to %s (%.2f MB in %.2f seconds)",
                local_path, blob_name, file_size / (1024 * 1024), upload_time
            )
            
            return UploadResult(
                blob_name=blob_name,
                success=True,
                etag=result["etag"],
                upload_time=upload_time,
                bytes_uploaded=file_size
            )
            
        except Exception as e:
            error_msg = f"Failed to upload {local_path}: {e}"
            logger.error(error_msg)
            return UploadResult(
                blob_name=blob_name,
                success=False,
                error_message=error_msg
            )

    def upload_files_batch(
        self, 
        file_mappings: List[Tuple[str, str]],  # (local_path, blob_name)
        access_tier: Optional[BlobAccessTier] = None,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[UploadResult]:
        """Upload multiple files concurrently."""
        logger.info("Starting batch upload of %d files", len(file_mappings))
        
        futures = []
        for local_path, blob_name in file_mappings:
            future = self.executor.submit(
                self.upload_file,
                local_path,
                blob_name,
                access_tier,
                metadata,
                tags
            )
            futures.append(future)

        results = []
        for future in futures:
            try:
                result = future.result(timeout=self.config.timeout)
                results.append(result)
            except Exception as e:
                logger.error("Batch upload task failed: %s", e)
                results.append(UploadResult(
                    blob_name="unknown",
                    success=False,
                    error_message=str(e)
                ))

        successful_uploads = sum(1 for r in results if r.success)
        logger.info(
            "Batch upload completed: %d/%d successful",
            successful_uploads, len(file_mappings)
        )
        
        return results

    def download_file(self, blob_name: str, local_path: str) -> bool:
        """Download file from Azure Blob Storage."""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())

            logger.info("Downloaded %s to %s", blob_name, local_path)
            return True
            
        except Exception as e:
            logger.error("Failed to download %s: %s", blob_name, e)
            return False

    def get_blob_metadata(self, blob_name: str) -> Optional[BlobMetadata]:
        """Get blob metadata and properties."""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            properties = blob_client.get_blob_properties()
            
            return BlobMetadata(
                name=blob_name,
                size=properties.size,
                content_type=properties.content_settings.content_type or "application/octet-stream",
                access_tier=properties.blob_tier,
                last_modified=properties.last_modified,
                etag=properties.etag,
                md5_hash=properties.content_settings.content_md5,
                custom_metadata=properties.metadata,
                tags=properties.tag_count and blob_client.get_blob_tags() or None
            )
            
        except ResourceNotFoundError:
            logger.warning("Blob not found: %s", blob_name)
            return None
        except Exception as e:
            logger.error("Failed to get blob metadata for %s: %s", blob_name, e)
            return None

    def list_blobs(
        self, 
        prefix: Optional[str] = None,
        include_metadata: bool = False,
        include_tags: bool = False
    ) -> Iterator[BlobMetadata]:
        """List blobs in container with optional filtering."""
        try:
            include_options = []
            if include_metadata:
                include_options.append("metadata")
            if include_tags:
                include_options.append("tags")

            blob_list = self.container_client.list_blobs(
                name_starts_with=prefix,
                include=include_options
            )

            for blob in blob_list:
                yield BlobMetadata(
                    name=blob.name,
                    size=blob.size,
                    content_type=blob.content_settings.content_type or "application/octet-stream",
                    access_tier=blob.blob_tier,
                    last_modified=blob.last_modified,
                    etag=blob.etag,
                    md5_hash=blob.content_settings.content_md5,
                    custom_metadata=blob.metadata if include_metadata else None,
                    tags=blob.tags if include_tags else None
                )
                
        except Exception as e:
            logger.error("Failed to list blobs: %s", e)

    def delete_blob(self, blob_name: str, delete_snapshots: bool = True) -> bool:
        """Delete blob from Azure Blob Storage."""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            
            delete_kwargs = {}
            if delete_snapshots:
                delete_kwargs["delete_snapshots"] = "include"
                
            blob_client.delete_blob(**delete_kwargs)
            
            logger.info("Deleted blob: %s", blob_name)
            return True
            
        except ResourceNotFoundError:
            logger.warning("Blob not found for deletion: %s", blob_name)
            return False
        except Exception as e:
            logger.error("Failed to delete blob %s: %s", blob_name, e)
            return False

    def set_blob_access_tier(self, blob_name: str, access_tier: BlobAccessTier) -> bool:
        """Change blob access tier for cost optimization."""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.set_standard_blob_tier(access_tier.value)
            
            logger.info("Set blob %s access tier to %s", blob_name, access_tier.value)
            return True
            
        except Exception as e:
            logger.error("Failed to set access tier for %s: %s", blob_name, e)
            return False

    def copy_blob(
        self, 
        source_blob_name: str, 
        destination_blob_name: str,
        source_container: Optional[str] = None
    ) -> bool:
        """Copy blob within or between containers."""
        try:
            if source_container:
                source_blob_client = self.blob_service_client.get_blob_client(
                    container=source_container,
                    blob=source_blob_name
                )
            else:
                source_blob_client = self.container_client.get_blob_client(source_blob_name)
                
            dest_blob_client = self.container_client.get_blob_client(destination_blob_name)
            
            # Start copy operation
            copy_props = dest_blob_client.start_copy_from_url(source_blob_client.url)
            
            # Wait for copy to complete (for small files this is usually immediate)
            copy_status = copy_props["copy_status"]
            if copy_status == "pending":
                # For large files, you might want to poll the copy status
                logger.info("Copy operation started for %s -> %s", source_blob_name, destination_blob_name)
            else:
                logger.info("Copied %s to %s", source_blob_name, destination_blob_name)
                
            return True
            
        except Exception as e:
            logger.error("Failed to copy blob %s to %s: %s", source_blob_name, destination_blob_name, e)
            return False

    def generate_sas_url(
        self, 
        blob_name: str, 
        expiry_hours: int = 24,
        permissions: str = "r"  # r=read, w=write, d=delete, l=list
    ) -> Optional[str]:
        """Generate SAS URL for secure blob access."""
        try:
            from azure.storage.blob import generate_blob_sas, BlobSasPermissions
            from datetime import datetime, timedelta
            
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=self.config.account_name,
                container_name=self.config.container_name,
                blob_name=blob_name,
                account_key=self.config.account_key,
                permission=BlobSasPermissions.from_string(permissions),
                expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
            )
            
            sas_url = f"{blob_client.url}?{sas_token}"
            logger.debug("Generated SAS URL for %s (expires in %d hours)", blob_name, expiry_hours)
            return sas_url
            
        except Exception as e:
            logger.error("Failed to generate SAS URL for %s: %s", blob_name, e)
            return None

    def get_container_stats(self) -> Dict[str, Any]:
        """Get container statistics and usage information."""
        try:
            blob_count = 0
            total_size = 0
            tier_distribution = {"Hot": 0, "Cool": 0, "Archive": 0}
            
            for blob in self.container_client.list_blobs():
                blob_count += 1
                total_size += blob.size
                if blob.blob_tier:
                    tier_distribution[blob.blob_tier] = tier_distribution.get(blob.blob_tier, 0) + 1

            stats = {
                "blob_count": blob_count,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "total_size_gb": total_size / (1024 * 1024 * 1024),
                "tier_distribution": tier_distribution,
                "container_name": self.config.container_name,
                "account_name": self.config.account_name
            }
            
            logger.debug("Container stats: %d blobs, %.2f GB", blob_count, stats["total_size_gb"])
            return stats
            
        except Exception as e:
            logger.error("Failed to get container stats: %s", e)
            return {}

    def health_check(self) -> bool:
        """Check Azure Blob Storage connectivity."""
        try:
            # Try to get container properties as health check
            self.container_client.get_container_properties()
            logger.info("Azure Blob Storage health check passed")
            return True
        except Exception as e:
            logger.error("Azure Blob Storage health check failed: %s", e)
            return False

    def cleanup_old_blobs(self, days_old: int = 30, dry_run: bool = True) -> List[str]:
        """Clean up blobs older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_blobs = []
        
        try:
            for blob in self.container_client.list_blobs():
                if blob.last_modified < cutoff_date:
                    if not dry_run:
                        if self.delete_blob(blob.name):
                            deleted_blobs.append(blob.name)
                    else:
                        deleted_blobs.append(blob.name)
                        logger.info("Would delete old blob: %s (last modified: %s)", 
                                  blob.name, blob.last_modified)

            if dry_run:
                logger.info("Dry run: %d blobs would be deleted", len(deleted_blobs))
            else:
                logger.info("Deleted %d old blobs", len(deleted_blobs))
                
            return deleted_blobs
            
        except Exception as e:
            logger.error("Failed to cleanup old blobs: %s", e)
            return []

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.executor:
            self.executor.shutdown(wait=True)


# Factory function for easy initialization
def create_blob_storage_connector(
    account_name: str,
    container_name: str,
    **kwargs
) -> AzureBlobStorageConnector:
    """Create Azure Blob Storage connector with configuration."""
    config = BlobStorageConfig(
        account_name=account_name,
        container_name=container_name,
        **kwargs
    )
    return AzureBlobStorageConnector(config)