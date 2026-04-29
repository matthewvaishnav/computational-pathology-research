"""
AWS S3 Storage Plugin

Provides integration with AWS S3 for scalable object storage of medical images,
DICOM files, analysis results, and other healthcare data with encryption and compliance.
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

import aioboto3
import boto3
import botocore
from botocore.exceptions import ClientError, NoCredentialsError

from ..plugin_interface import PluginCapability, StoragePlugin
from ..plugin_manager import PluginMetadata


class S3StorageClass(Enum):
    """S3 storage classes"""

    STANDARD = "STANDARD"
    REDUCED_REDUNDANCY = "REDUCED_REDUNDANCY"
    STANDARD_IA = "STANDARD_IA"
    ONEZONE_IA = "ONEZONE_IA"
    INTELLIGENT_TIERING = "INTELLIGENT_TIERING"
    GLACIER = "GLACIER"
    DEEP_ARCHIVE = "DEEP_ARCHIVE"


class S3EncryptionType(Enum):
    """S3 encryption types"""

    NONE = "none"
    AES256 = "AES256"
    KMS = "aws:kms"


@dataclass
class S3Object:
    """S3 object information"""

    key: str
    bucket: str
    size: int
    last_modified: datetime
    etag: str
    storage_class: S3StorageClass
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["last_modified"] = self.last_modified.isoformat()
        data["storage_class"] = self.storage_class.value
        return data


@dataclass
class S3UploadResult:
    """S3 upload result"""

    key: str
    bucket: str
    etag: str
    version_id: Optional[str] = None
    upload_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class S3MultipartUpload:
    """S3 multipart upload information"""

    upload_id: str
    key: str
    bucket: str
    initiated: datetime
    parts: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["initiated"] = self.initiated.isoformat()
        return data


class AWSS3StoragePlugin(StoragePlugin):
    """
    AWS S3 storage integration plugin

    Provides comprehensive S3 integration including:
    - Object storage and retrieval
    - Multipart uploads for large files
    - Lifecycle management
    - Encryption and compliance
    - Metadata management
    - Integration with other AWS services
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize AWS S3 storage plugin"""
        super().__init__(config)

        # AWS configuration
        self.aws_region = config.get("aws_region", "us-east-1")
        self.aws_access_key_id = config.get("aws_access_key_id")
        self.aws_secret_access_key = config.get("aws_secret_access_key")
        self.aws_session_token = config.get("aws_session_token")

        # S3 settings
        self.bucket_name = config.get("bucket_name")
        self.key_prefix = config.get("key_prefix", "medical-ai/")
        self.default_storage_class = S3StorageClass(config.get("default_storage_class", "STANDARD"))

        # Encryption settings
        self.encryption_type = S3EncryptionType(config.get("encryption_type", "AES256"))
        self.kms_key_id = config.get("kms_key_id")

        # Upload settings
        self.multipart_threshold = config.get("multipart_threshold", 64 * 1024 * 1024)  # 64MB
        self.multipart_chunksize = config.get("multipart_chunksize", 8 * 1024 * 1024)  # 8MB
        self.max_concurrency = config.get("max_concurrency", 10)

        # Connection settings
        self.timeout = config.get("timeout", 300)  # 5 minutes for large uploads
        self.max_retries = config.get("max_retries", 3)

        # AWS clients
        self.s3_client = None
        self.s3_resource = None
        self.session = None

        # Lifecycle management
        self.lifecycle_enabled = config.get("lifecycle_enabled", False)
        self.lifecycle_rules = config.get("lifecycle_rules", [])

        # Versioning
        self.versioning_enabled = config.get("versioning_enabled", False)

        self.logger = logging.getLogger(__name__)

    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        return PluginMetadata(
            name="aws-s3-storage",
            version="1.0.0",
            description="AWS S3 Object Storage Integration",
            vendor="Amazon Web Services",
            capabilities=[
                PluginCapability.OBJECT_STORAGE,
                PluginCapability.LARGE_FILE_SUPPORT,
                PluginCapability.ENCRYPTION,
                PluginCapability.LIFECYCLE_MANAGEMENT,
                PluginCapability.VERSIONING,
                PluginCapability.METADATA_MANAGEMENT,
            ],
            supported_formats=["All file types", "DICOM", "Images", "JSON", "Binary"],
            configuration_schema={
                "aws_region": {"type": "string", "required": True},
                "bucket_name": {"type": "string", "required": True},
                "key_prefix": {"type": "string", "default": "medical-ai/"},
                "encryption_type": {
                    "type": "string",
                    "enum": ["none", "AES256", "aws:kms"],
                    "default": "AES256",
                },
                "multipart_threshold": {"type": "integer", "default": 67108864},
                "aws_access_key_id": {"type": "string", "required": False, "sensitive": True},
                "aws_secret_access_key": {"type": "string", "required": False, "sensitive": True},
            },
        )

    async def initialize(self) -> bool:
        """Initialize AWS S3 connection"""
        try:
            # Create aioboto3 session
            session_kwargs = {"region_name": self.aws_region}

            if self.aws_access_key_id and self.aws_secret_access_key:
                session_kwargs.update(
                    {
                        "aws_access_key_id": self.aws_access_key_id,
                        "aws_secret_access_key": self.aws_secret_access_key,
                    }
                )

                if self.aws_session_token:
                    session_kwargs["aws_session_token"] = self.aws_session_token

            # Create aioboto3 session for async operations
            self.session = aioboto3.Session(**session_kwargs)

            # Create sync boto3 clients for some operations
            boto_session = boto3.Session(**session_kwargs)
            self.s3_client = boto_session.client("s3")
            self.s3_resource = boto_session.resource("s3")

            # Test connection
            if not await self._test_connection():
                self.logger.error("S3 connection test failed")
                return False

            # Setup bucket if needed
            if not await self._setup_bucket():
                self.logger.error("S3 bucket setup failed")
                return False

            self.logger.info("AWS S3 storage plugin initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"AWS S3 initialization failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup resources"""
        # aioboto3 sessions are automatically cleaned up
        pass

    async def _test_connection(self) -> bool:
        """Test S3 connection"""
        try:
            async with self.session.client("s3") as s3:
                # Test by listing buckets
                response = await s3.list_buckets()
                return "Buckets" in response

        except Exception as e:
            self.logger.error(f"S3 connection test failed: {e}")
            return False

    async def _setup_bucket(self) -> bool:
        """Setup S3 bucket with required configuration"""
        try:
            # Check if bucket exists
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                bucket_exists = True
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    bucket_exists = False
                else:
                    raise

            # Create bucket if it doesn't exist
            if not bucket_exists:
                self.logger.info(f"Creating S3 bucket: {self.bucket_name}")

                if self.aws_region == "us-east-1":
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": self.aws_region},
                    )

            # Configure bucket encryption
            if self.encryption_type != S3EncryptionType.NONE:
                await self._configure_bucket_encryption()

            # Configure versioning if enabled
            if self.versioning_enabled:
                await self._configure_bucket_versioning()

            # Configure lifecycle rules if enabled
            if self.lifecycle_enabled and self.lifecycle_rules:
                await self._configure_bucket_lifecycle()

            return True

        except Exception as e:
            self.logger.error(f"Error setting up S3 bucket: {e}")
            return False

    async def _configure_bucket_encryption(self):
        """Configure bucket encryption"""
        try:
            encryption_config = {
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": self.encryption_type.value
                        }
                    }
                ]
            }

            if self.encryption_type == S3EncryptionType.KMS and self.kms_key_id:
                encryption_config["Rules"][0]["ApplyServerSideEncryptionByDefault"][
                    "KMSMasterKeyID"
                ] = self.kms_key_id

            self.s3_client.put_bucket_encryption(
                Bucket=self.bucket_name, ServerSideEncryptionConfiguration=encryption_config
            )

            self.logger.info(f"Configured bucket encryption: {self.encryption_type.value}")

        except Exception as e:
            self.logger.error(f"Error configuring bucket encryption: {e}")

    async def _configure_bucket_versioning(self):
        """Configure bucket versioning"""
        try:
            self.s3_client.put_bucket_versioning(
                Bucket=self.bucket_name, VersioningConfiguration={"Status": "Enabled"}
            )

            self.logger.info("Enabled bucket versioning")

        except Exception as e:
            self.logger.error(f"Error configuring bucket versioning: {e}")

    async def _configure_bucket_lifecycle(self):
        """Configure bucket lifecycle rules"""
        try:
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket_name, LifecycleConfiguration={"Rules": self.lifecycle_rules}
            )

            self.logger.info("Configured bucket lifecycle rules")

        except Exception as e:
            self.logger.error(f"Error configuring bucket lifecycle: {e}")

    def _get_full_key(self, key: str) -> str:
        """Get full S3 key with prefix"""
        if key.startswith(self.key_prefix):
            return key
        return f"{self.key_prefix}{key}"

    def _get_content_type(self, key: str) -> str:
        """Get content type based on file extension"""
        content_type, _ = mimetypes.guess_type(key)
        return content_type or "application/octet-stream"

    async def upload_file(
        self,
        local_path: str,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        storage_class: Optional[S3StorageClass] = None,
    ) -> Optional[S3UploadResult]:
        """Upload file to S3"""
        try:
            full_key = self._get_full_key(key)
            content_type = self._get_content_type(key)
            storage_class = storage_class or self.default_storage_class

            # Prepare upload parameters
            upload_params = {
                "Bucket": self.bucket_name,
                "Key": full_key,
                "ContentType": content_type,
                "StorageClass": storage_class.value,
            }

            if metadata:
                upload_params["Metadata"] = metadata

            # Add encryption parameters
            if self.encryption_type == S3EncryptionType.AES256:
                upload_params["ServerSideEncryption"] = "AES256"
            elif self.encryption_type == S3EncryptionType.KMS:
                upload_params["ServerSideEncryption"] = "aws:kms"
                if self.kms_key_id:
                    upload_params["SSEKMSKeyId"] = self.kms_key_id

            # Check file size for multipart upload
            file_size = Path(local_path).stat().st_size

            if file_size >= self.multipart_threshold:
                return await self._upload_large_file(local_path, upload_params)
            else:
                return await self._upload_small_file(local_path, upload_params)

        except Exception as e:
            self.logger.error(f"Error uploading file {local_path}: {e}")
            return None

    async def _upload_small_file(
        self, local_path: str, upload_params: Dict[str, Any]
    ) -> Optional[S3UploadResult]:
        """Upload small file using single request"""
        try:
            async with self.session.client("s3") as s3:
                with open(local_path, "rb") as f:
                    upload_params["Body"] = f.read()

                response = await s3.put_object(**upload_params)

                return S3UploadResult(
                    key=upload_params["Key"],
                    bucket=upload_params["Bucket"],
                    etag=response["ETag"].strip('"'),
                    version_id=response.get("VersionId"),
                )

        except Exception as e:
            self.logger.error(f"Error uploading small file: {e}")
            return None

    async def _upload_large_file(
        self, local_path: str, upload_params: Dict[str, Any]
    ) -> Optional[S3UploadResult]:
        """Upload large file using multipart upload"""
        try:
            async with self.session.client("s3") as s3:
                # Initiate multipart upload
                create_params = {k: v for k, v in upload_params.items() if k != "Body"}

                response = await s3.create_multipart_upload(**create_params)
                upload_id = response["UploadId"]

                try:
                    # Upload parts
                    parts = []
                    part_number = 1

                    with open(local_path, "rb") as f:
                        while True:
                            chunk = f.read(self.multipart_chunksize)
                            if not chunk:
                                break

                            part_response = await s3.upload_part(
                                Bucket=upload_params["Bucket"],
                                Key=upload_params["Key"],
                                PartNumber=part_number,
                                UploadId=upload_id,
                                Body=chunk,
                            )

                            parts.append({"ETag": part_response["ETag"], "PartNumber": part_number})

                            part_number += 1

                    # Complete multipart upload
                    complete_response = await s3.complete_multipart_upload(
                        Bucket=upload_params["Bucket"],
                        Key=upload_params["Key"],
                        UploadId=upload_id,
                        MultipartUpload={"Parts": parts},
                    )

                    return S3UploadResult(
                        key=upload_params["Key"],
                        bucket=upload_params["Bucket"],
                        etag=complete_response["ETag"].strip('"'),
                        version_id=complete_response.get("VersionId"),
                        upload_id=upload_id,
                    )

                except Exception as e:
                    # Abort multipart upload on error
                    await s3.abort_multipart_upload(
                        Bucket=upload_params["Bucket"], Key=upload_params["Key"], UploadId=upload_id
                    )
                    raise

        except Exception as e:
            self.logger.error(f"Error uploading large file: {e}")
            return None

    async def download_file(self, key: str, local_path: str) -> bool:
        """Download file from S3"""
        try:
            full_key = self._get_full_key(key)

            async with self.session.client("s3") as s3:
                # Create directory if it doesn't exist
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)

                # Download file
                await s3.download_file(self.bucket_name, full_key, local_path)

                self.logger.info(f"Downloaded {full_key} to {local_path}")
                return True

        except Exception as e:
            self.logger.error(f"Error downloading file {key}: {e}")
            return False

    async def get_object_info(self, key: str) -> Optional[S3Object]:
        """Get object information"""
        try:
            full_key = self._get_full_key(key)

            async with self.session.client("s3") as s3:
                response = await s3.head_object(Bucket=self.bucket_name, Key=full_key)

                return S3Object(
                    key=full_key,
                    bucket=self.bucket_name,
                    size=response["ContentLength"],
                    last_modified=response["LastModified"],
                    etag=response["ETag"].strip('"'),
                    storage_class=S3StorageClass(response.get("StorageClass", "STANDARD")),
                    content_type=response.get("ContentType"),
                    metadata=response.get("Metadata", {}),
                )

        except Exception as e:
            self.logger.error(f"Error getting object info for {key}: {e}")
            return None

    async def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[S3Object]:
        """List objects in bucket"""
        try:
            full_prefix = self._get_full_key(prefix)

            async with self.session.client("s3") as s3:
                response = await s3.list_objects_v2(
                    Bucket=self.bucket_name, Prefix=full_prefix, MaxKeys=max_keys
                )

                objects = []
                for obj in response.get("Contents", []):
                    s3_object = S3Object(
                        key=obj["Key"],
                        bucket=self.bucket_name,
                        size=obj["Size"],
                        last_modified=obj["LastModified"],
                        etag=obj["ETag"].strip('"'),
                        storage_class=S3StorageClass(obj.get("StorageClass", "STANDARD")),
                    )
                    objects.append(s3_object)

                return objects

        except Exception as e:
            self.logger.error(f"Error listing objects: {e}")
            return []

    async def delete_object(self, key: str) -> bool:
        """Delete object from S3"""
        try:
            full_key = self._get_full_key(key)

            async with self.session.client("s3") as s3:
                await s3.delete_object(Bucket=self.bucket_name, Key=full_key)

                self.logger.info(f"Deleted object: {full_key}")
                return True

        except Exception as e:
            self.logger.error(f"Error deleting object {key}: {e}")
            return False

    async def copy_object(
        self, source_key: str, dest_key: str, dest_bucket: Optional[str] = None
    ) -> bool:
        """Copy object within S3"""
        try:
            source_full_key = self._get_full_key(source_key)
            dest_full_key = self._get_full_key(dest_key)
            dest_bucket = dest_bucket or self.bucket_name

            async with self.session.client("s3") as s3:
                copy_source = {"Bucket": self.bucket_name, "Key": source_full_key}

                await s3.copy_object(CopySource=copy_source, Bucket=dest_bucket, Key=dest_full_key)

                self.logger.info(f"Copied {source_full_key} to {dest_full_key}")
                return True

        except Exception as e:
            self.logger.error(f"Error copying object: {e}")
            return False

    async def generate_presigned_url(
        self, key: str, expiration: int = 3600, method: str = "get_object"
    ) -> Optional[str]:
        """Generate presigned URL for object access"""
        try:
            full_key = self._get_full_key(key)

            # Use sync client for presigned URL generation
            url = self.s3_client.generate_presigned_url(
                method, Params={"Bucket": self.bucket_name, "Key": full_key}, ExpiresIn=expiration
            )

            return url

        except Exception as e:
            self.logger.error(f"Error generating presigned URL: {e}")
            return None

    async def set_object_metadata(self, key: str, metadata: Dict[str, str]) -> bool:
        """Set object metadata"""
        try:
            full_key = self._get_full_key(key)

            # Get current object info
            obj_info = await self.get_object_info(key)
            if not obj_info:
                return False

            async with self.session.client("s3") as s3:
                # Copy object to itself with new metadata
                copy_source = {"Bucket": self.bucket_name, "Key": full_key}

                await s3.copy_object(
                    CopySource=copy_source,
                    Bucket=self.bucket_name,
                    Key=full_key,
                    Metadata=metadata,
                    MetadataDirective="REPLACE",
                )

                return True

        except Exception as e:
            self.logger.error(f"Error setting object metadata: {e}")
            return False

    async def validate_connection(self) -> Dict[str, Any]:
        """Validate S3 connection"""
        try:
            # Test bucket access
            bucket_accessible = False
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                bucket_accessible = True
            except:
                pass

            # Test object operations
            objects_accessible = False
            try:
                objects = await self.list_objects(max_keys=1)
                objects_accessible = True
            except:
                pass

            # Test upload permissions
            upload_accessible = False
            try:
                test_key = f"test/{uuid.uuid4()}.txt"
                test_content = b"test"

                async with self.session.client("s3") as s3:
                    await s3.put_object(
                        Bucket=self.bucket_name, Key=self._get_full_key(test_key), Body=test_content
                    )

                    # Clean up test object
                    await s3.delete_object(
                        Bucket=self.bucket_name, Key=self._get_full_key(test_key)
                    )

                upload_accessible = True
            except:
                pass

            return {
                "connected": bucket_accessible and objects_accessible,
                "bucket_accessible": bucket_accessible,
                "objects_accessible": objects_accessible,
                "upload_accessible": upload_accessible,
                "bucket_name": self.bucket_name,
                "aws_region": self.aws_region,
                "encryption_enabled": self.encryption_type != S3EncryptionType.NONE,
                "versioning_enabled": self.versioning_enabled,
                "last_check": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"connected": False, "error": str(e), "last_check": datetime.now().isoformat()}


# Plugin factory function
def create_plugin(config: Dict[str, Any]) -> AWSS3StoragePlugin:
    """Create AWS S3 storage plugin instance"""
    return AWSS3StoragePlugin(config)
