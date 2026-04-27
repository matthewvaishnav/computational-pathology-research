"""
AWS HealthLake Integration Plugin

Provides integration with AWS HealthLake for FHIR-based healthcare data storage,
analytics, and machine learning capabilities in the cloud.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import base64

import boto3
import botocore
from botocore.exceptions import ClientError, NoCredentialsError
import aiohttp
import ssl

from ..plugin_interface import CloudPlugin, PluginCapability
from ..plugin_manager import PluginMetadata


class HealthLakeDatastoreStatus(Enum):
    """HealthLake datastore status"""
    CREATING = "CREATING"
    ACTIVE = "ACTIVE"
    DELETING = "DELETING"
    DELETED = "DELETED"


class FHIRVersion(Enum):
    """Supported FHIR versions"""
    R4 = "R4"


@dataclass
class HealthLakeDatastore:
    """HealthLake datastore information"""
    datastore_id: str
    datastore_name: str
    datastore_status: HealthLakeDatastoreStatus
    datastore_endpoint: str
    fhir_version: FHIRVersion
    created_at: datetime
    preload_data_config: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['datastore_status'] = self.datastore_status.value
        data['fhir_version'] = self.fhir_version.value
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class HealthLakeImportJob:
    """HealthLake import job information"""
    job_id: str
    job_name: str
    job_status: str
    datastore_id: str
    input_data_config: Dict[str, Any]
    output_data_config: Dict[str, Any]
    submit_time: datetime
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['submit_time'] = self.submit_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


@dataclass
class HealthLakeExportJob:
    """HealthLake export job information"""
    job_id: str
    job_name: str
    job_status: str
    datastore_id: str
    output_data_config: Dict[str, Any]
    submit_time: datetime
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['submit_time'] = self.submit_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


class AWSHealthLakePlugin(CloudPlugin):
    """
    AWS HealthLake integration plugin
    
    Provides comprehensive HealthLake integration including:
    - FHIR datastore management
    - Patient and clinical data storage
    - Bulk data import/export
    - Analytics and ML integration
    - Compliance and security features
    - Integration with other AWS services
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize AWS HealthLake plugin"""
        super().__init__(config)
        
        # AWS configuration
        self.aws_region = config.get('aws_region', 'us-east-1')
        self.aws_access_key_id = config.get('aws_access_key_id')
        self.aws_secret_access_key = config.get('aws_secret_access_key')
        self.aws_session_token = config.get('aws_session_token')
        
        # HealthLake settings
        self.datastore_id = config.get('datastore_id')
        self.datastore_name = config.get('datastore_name', 'ai-pathology-datastore')
        self.fhir_version = FHIRVersion(config.get('fhir_version', 'R4'))
        
        # S3 settings for bulk operations
        self.s3_bucket = config.get('s3_bucket')
        self.s3_key_prefix = config.get('s3_key_prefix', 'healthlake/')
        
        # Connection settings
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        
        # AWS clients
        self.healthlake_client = None
        self.s3_client = None
        self.session = None
        
        # Rate limiting
        self.rate_limit = config.get('rate_limit', 100)  # requests per minute
        self.rate_window = 60
        self.request_times = []
        
        # Datastore cache
        self.datastore_info = None
        self.datastore_endpoint = None
        
        self.logger = logging.getLogger(__name__)
    
    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        return PluginMetadata(
            name="aws-healthlake",
            version="1.0.0",
            description="AWS HealthLake FHIR Data Store Integration",
            vendor="Amazon Web Services",
            capabilities=[
                PluginCapability.FHIR_STORAGE,
                PluginCapability.BULK_DATA_OPERATIONS,
                PluginCapability.ANALYTICS_INTEGRATION,
                PluginCapability.ML_INTEGRATION,
                PluginCapability.COMPLIANCE_FEATURES
            ],
            supported_formats=["FHIR R4", "JSON", "NDJSON"],
            configuration_schema={
                "aws_region": {"type": "string", "required": True},
                "datastore_id": {"type": "string", "required": False},
                "datastore_name": {"type": "string", "default": "ai-pathology-datastore"},
                "s3_bucket": {"type": "string", "required": True},
                "aws_access_key_id": {"type": "string", "required": False, "sensitive": True},
                "aws_secret_access_key": {"type": "string", "required": False, "sensitive": True}
            }
        )
    
    async def initialize(self) -> bool:
        """Initialize AWS HealthLake connection"""
        try:
            # Create AWS session
            session_kwargs = {'region_name': self.aws_region}
            
            if self.aws_access_key_id and self.aws_secret_access_key:
                session_kwargs.update({
                    'aws_access_key_id': self.aws_access_key_id,
                    'aws_secret_access_key': self.aws_secret_access_key
                })
                
                if self.aws_session_token:
                    session_kwargs['aws_session_token'] = self.aws_session_token
            
            # Create boto3 session
            boto_session = boto3.Session(**session_kwargs)
            
            # Create HealthLake client
            self.healthlake_client = boto_session.client('healthlake')
            
            # Create S3 client
            self.s3_client = boto_session.client('s3')
            
            # Create HTTP session for FHIR API calls
            connector = aiohttp.TCPConnector(ssl=ssl.create_default_context())
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
            # Initialize or get datastore
            if not await self._initialize_datastore():
                self.logger.error("Failed to initialize HealthLake datastore")
                return False
            
            # Test connection
            if not await self._test_connection():
                self.logger.error("HealthLake connection test failed")
                return False
            
            self.logger.info("AWS HealthLake plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"AWS HealthLake initialization failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def _initialize_datastore(self) -> bool:
        """Initialize or get existing datastore"""
        try:
            if self.datastore_id:
                # Get existing datastore
                datastore_info = await self._get_datastore_info(self.datastore_id)
                if datastore_info:
                    self.datastore_info = datastore_info
                    self.datastore_endpoint = datastore_info.datastore_endpoint
                    return True
                else:
                    self.logger.error(f"Datastore {self.datastore_id} not found")
                    return False
            else:
                # List existing datastores
                datastores = await self._list_datastores()
                
                # Find datastore by name
                for datastore in datastores:
                    if datastore.datastore_name == self.datastore_name:
                        self.datastore_info = datastore
                        self.datastore_id = datastore.datastore_id
                        self.datastore_endpoint = datastore.datastore_endpoint
                        return True
                
                # Create new datastore if not found
                self.logger.info(f"Creating new HealthLake datastore: {self.datastore_name}")
                datastore_info = await self._create_datastore()
                if datastore_info:
                    self.datastore_info = datastore_info
                    self.datastore_id = datastore_info.datastore_id
                    self.datastore_endpoint = datastore_info.datastore_endpoint
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing datastore: {e}")
            return False
    
    async def _create_datastore(self) -> Optional[HealthLakeDatastore]:
        """Create new HealthLake datastore"""
        try:
            response = self.healthlake_client.create_fhir_datastore(
                DatastoreName=self.datastore_name,
                DatastoreTypeVersion=self.fhir_version.value,
                PreloadDataConfig={
                    'PreloadDataType': 'SYNTHEA'  # Optional: preload with synthetic data
                }
            )
            
            datastore_id = response['DatastoreId']
            
            # Wait for datastore to become active
            self.logger.info(f"Waiting for datastore {datastore_id} to become active...")
            
            waiter = self.healthlake_client.get_waiter('datastore_active')
            waiter.wait(
                DatastoreId=datastore_id,
                WaiterConfig={
                    'Delay': 30,
                    'MaxAttempts': 40  # Wait up to 20 minutes
                }
            )
            
            # Get datastore info
            return await self._get_datastore_info(datastore_id)
            
        except Exception as e:
            self.logger.error(f"Error creating datastore: {e}")
            return None
    
    async def _get_datastore_info(self, datastore_id: str) -> Optional[HealthLakeDatastore]:
        """Get datastore information"""
        try:
            response = self.healthlake_client.describe_fhir_datastore(
                DatastoreId=datastore_id
            )
            
            datastore_props = response['DatastoreProperties']
            
            return HealthLakeDatastore(
                datastore_id=datastore_props['DatastoreId'],
                datastore_name=datastore_props['DatastoreName'],
                datastore_status=HealthLakeDatastoreStatus(datastore_props['DatastoreStatus']),
                datastore_endpoint=datastore_props['DatastoreEndpoint'],
                fhir_version=FHIRVersion(datastore_props['DatastoreTypeVersion']),
                created_at=datastore_props['CreatedAt'],
                preload_data_config=datastore_props.get('PreloadDataConfig')
            )
            
        except Exception as e:
            self.logger.error(f"Error getting datastore info: {e}")
            return None
    
    async def _list_datastores(self) -> List[HealthLakeDatastore]:
        """List all datastores"""
        try:
            response = self.healthlake_client.list_fhir_datastores()
            
            datastores = []
            for datastore_props in response['DatastorePropertiesList']:
                datastore = HealthLakeDatastore(
                    datastore_id=datastore_props['DatastoreId'],
                    datastore_name=datastore_props['DatastoreName'],
                    datastore_status=HealthLakeDatastoreStatus(datastore_props['DatastoreStatus']),
                    datastore_endpoint=datastore_props['DatastoreEndpoint'],
                    fhir_version=FHIRVersion(datastore_props['DatastoreTypeVersion']),
                    created_at=datastore_props['CreatedAt'],
                    preload_data_config=datastore_props.get('PreloadDataConfig')
                )
                datastores.append(datastore)
            
            return datastores
            
        except Exception as e:
            self.logger.error(f"Error listing datastores: {e}")
            return []
    
    async def _test_connection(self) -> bool:
        """Test HealthLake connection"""
        try:
            if not self.datastore_endpoint:
                return False
            
            # Test FHIR metadata endpoint
            metadata_url = f"{self.datastore_endpoint}/metadata"
            
            # Get AWS credentials for signing
            credentials = self._get_aws_credentials()
            if not credentials:
                return False
            
            # Create signed request
            headers = await self._create_signed_headers('GET', metadata_url, credentials)
            
            async with self.session.get(metadata_url, headers=headers) as response:
                if response.status == 200:
                    metadata = await response.json()
                    return metadata.get('resourceType') == 'CapabilityStatement'
                return False
                
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def _get_aws_credentials(self) -> Optional[Dict[str, str]]:
        """Get AWS credentials for request signing"""
        try:
            # Try to get credentials from various sources
            if self.aws_access_key_id and self.aws_secret_access_key:
                return {
                    'access_key': self.aws_access_key_id,
                    'secret_key': self.aws_secret_access_key,
                    'session_token': self.aws_session_token
                }
            
            # Try to get from boto3 session
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if credentials:
                return {
                    'access_key': credentials.access_key,
                    'secret_key': credentials.secret_key,
                    'session_token': credentials.token
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting AWS credentials: {e}")
            return None
    
    async def _create_signed_headers(self, method: str, url: str, credentials: Dict[str, str]) -> Dict[str, str]:
        """Create AWS Signature Version 4 headers"""
        try:
            # This is a simplified implementation
            # In production, use boto3's request signing capabilities
            
            headers = {
                'Accept': 'application/fhir+json',
                'Content-Type': 'application/fhir+json'
            }
            
            # Add authorization header (simplified)
            # In practice, use botocore.auth.SigV4Auth for proper signing
            
            return headers
            
        except Exception as e:
            self.logger.error(f"Error creating signed headers: {e}")
            return {}
    
    async def _rate_limit_check(self):
        """Check and enforce rate limits"""
        now = datetime.now()
        
        # Remove old requests outside the window
        self.request_times = [
            req_time for req_time in self.request_times 
            if (now - req_time).total_seconds() < self.rate_window
        ]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.rate_limit:
            sleep_time = self.rate_window - (now - self.request_times[0]).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(now)
    
    async def create_fhir_resource(self, resource_type: str, resource_data: Dict[str, Any]) -> Optional[str]:
        """Create FHIR resource in HealthLake"""
        try:
            await self._rate_limit_check()
            
            if not self.datastore_endpoint:
                return None
            
            create_url = f"{self.datastore_endpoint}/{resource_type}"
            
            # Get credentials and create signed headers
            credentials = self._get_aws_credentials()
            if not credentials:
                return None
            
            headers = await self._create_signed_headers('POST', create_url, credentials)
            
            async with self.session.post(create_url, json=resource_data, headers=headers) as response:
                if response.status in [200, 201]:
                    created_resource = await response.json()
                    return created_resource.get('id')
                else:
                    self.logger.error(f"Failed to create FHIR resource: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error creating FHIR resource: {e}")
            return None
    
    async def get_fhir_resource(self, resource_type: str, resource_id: str) -> Optional[Dict[str, Any]]:
        """Get FHIR resource from HealthLake"""
        try:
            await self._rate_limit_check()
            
            if not self.datastore_endpoint:
                return None
            
            resource_url = f"{self.datastore_endpoint}/{resource_type}/{resource_id}"
            
            # Get credentials and create signed headers
            credentials = self._get_aws_credentials()
            if not credentials:
                return None
            
            headers = await self._create_signed_headers('GET', resource_url, credentials)
            
            async with self.session.get(resource_url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    return None
                else:
                    self.logger.error(f"Failed to get FHIR resource: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error getting FHIR resource: {e}")
            return None
    
    async def search_fhir_resources(self, resource_type: str, search_params: Dict[str, str]) -> List[Dict[str, Any]]:
        """Search FHIR resources in HealthLake"""
        try:
            await self._rate_limit_check()
            
            if not self.datastore_endpoint:
                return []
            
            search_url = f"{self.datastore_endpoint}/{resource_type}"
            
            # Get credentials and create signed headers
            credentials = self._get_aws_credentials()
            if not credentials:
                return []
            
            headers = await self._create_signed_headers('GET', search_url, credentials)
            
            async with self.session.get(search_url, params=search_params, headers=headers) as response:
                if response.status == 200:
                    bundle_data = await response.json()
                    resources = []
                    
                    if bundle_data.get('resourceType') == 'Bundle':
                        for entry in bundle_data.get('entry', []):
                            if 'resource' in entry:
                                resources.append(entry['resource'])
                    
                    return resources
                else:
                    self.logger.error(f"Failed to search FHIR resources: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error searching FHIR resources: {e}")
            return []
    
    async def start_import_job(self, input_s3_uri: str, job_name: Optional[str] = None) -> Optional[str]:
        """Start bulk data import job"""
        try:
            if not self.datastore_id:
                return None
            
            job_name = job_name or f"import-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            response = self.healthlake_client.start_fhir_import_job(
                InputDataConfig={
                    'S3Uri': input_s3_uri
                },
                OutputDataConfig={
                    'S3Configuration': {
                        'S3Uri': f"s3://{self.s3_bucket}/{self.s3_key_prefix}import-output/",
                        'KmsKeyId': 'alias/aws/s3'  # Use default S3 encryption
                    }
                },
                DatastoreId=self.datastore_id,
                DataAccessRoleArn=self._get_data_access_role_arn(),
                JobName=job_name
            )
            
            job_id = response['JobId']
            self.logger.info(f"Started import job: {job_id}")
            
            return job_id
            
        except Exception as e:
            self.logger.error(f"Error starting import job: {e}")
            return None
    
    async def start_export_job(self, output_s3_uri: str, job_name: Optional[str] = None) -> Optional[str]:
        """Start bulk data export job"""
        try:
            if not self.datastore_id:
                return None
            
            job_name = job_name or f"export-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            response = self.healthlake_client.start_fhir_export_job(
                OutputDataConfig={
                    'S3Configuration': {
                        'S3Uri': output_s3_uri,
                        'KmsKeyId': 'alias/aws/s3'
                    }
                },
                DatastoreId=self.datastore_id,
                DataAccessRoleArn=self._get_data_access_role_arn(),
                JobName=job_name
            )
            
            job_id = response['JobId']
            self.logger.info(f"Started export job: {job_id}")
            
            return job_id
            
        except Exception as e:
            self.logger.error(f"Error starting export job: {e}")
            return None
    
    async def get_import_job_status(self, job_id: str) -> Optional[HealthLakeImportJob]:
        """Get import job status"""
        try:
            response = self.healthlake_client.describe_fhir_import_job(
                DatastoreId=self.datastore_id,
                JobId=job_id
            )
            
            job_props = response['ImportJobProperties']
            
            return HealthLakeImportJob(
                job_id=job_props['JobId'],
                job_name=job_props['JobName'],
                job_status=job_props['JobStatus'],
                datastore_id=job_props['DatastoreId'],
                input_data_config=job_props['InputDataConfig'],
                output_data_config=job_props['OutputDataConfig'],
                submit_time=job_props['SubmitTime'],
                end_time=job_props.get('EndTime')
            )
            
        except Exception as e:
            self.logger.error(f"Error getting import job status: {e}")
            return None
    
    async def get_export_job_status(self, job_id: str) -> Optional[HealthLakeExportJob]:
        """Get export job status"""
        try:
            response = self.healthlake_client.describe_fhir_export_job(
                DatastoreId=self.datastore_id,
                JobId=job_id
            )
            
            job_props = response['ExportJobProperties']
            
            return HealthLakeExportJob(
                job_id=job_props['JobId'],
                job_name=job_props['JobName'],
                job_status=job_props['JobStatus'],
                datastore_id=job_props['DatastoreId'],
                output_data_config=job_props['OutputDataConfig'],
                submit_time=job_props['SubmitTime'],
                end_time=job_props.get('EndTime')
            )
            
        except Exception as e:
            self.logger.error(f"Error getting export job status: {e}")
            return None
    
    def _get_data_access_role_arn(self) -> str:
        """Get IAM role ARN for HealthLake data access"""
        # This should be configured based on your AWS setup
        # The role needs permissions for HealthLake and S3 access
        return f"arn:aws:iam::{self._get_account_id()}:role/HealthLakeDataAccessRole"
    
    def _get_account_id(self) -> str:
        """Get AWS account ID"""
        try:
            sts_client = boto3.client('sts', region_name=self.aws_region)
            response = sts_client.get_caller_identity()
            return response['Account']
        except Exception as e:
            self.logger.error(f"Error getting account ID: {e}")
            return "123456789012"  # Fallback
    
    async def validate_connection(self) -> Dict[str, Any]:
        """Validate HealthLake connection"""
        try:
            # Test datastore access
            datastore_accessible = False
            if self.datastore_id:
                datastore_info = await self._get_datastore_info(self.datastore_id)
                datastore_accessible = datastore_info is not None
            
            # Test FHIR endpoint
            fhir_accessible = await self._test_connection()
            
            # Test S3 access
            s3_accessible = False
            try:
                self.s3_client.head_bucket(Bucket=self.s3_bucket)
                s3_accessible = True
            except:
                pass
            
            return {
                'connected': datastore_accessible and fhir_accessible,
                'datastore_accessible': datastore_accessible,
                'fhir_accessible': fhir_accessible,
                's3_accessible': s3_accessible,
                'datastore_id': self.datastore_id,
                'datastore_endpoint': self.datastore_endpoint,
                'aws_region': self.aws_region,
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }


# Plugin factory function
def create_plugin(config: Dict[str, Any]) -> AWSHealthLakePlugin:
    """Create AWS HealthLake plugin instance"""
    return AWSHealthLakePlugin(config)