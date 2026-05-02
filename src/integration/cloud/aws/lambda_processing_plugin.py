"""
AWS Lambda Processing Plugin

Provides integration with AWS Lambda for serverless processing of medical images,
AI model inference, data transformation, and event-driven workflows.
"""

import asyncio
import base64
import io
import json
import logging
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import aioboto3
import boto3
import botocore
from botocore.exceptions import ClientError, NoCredentialsError

from ..plugin_interface import PluginCapability, ProcessingPlugin
from ..plugin_manager import PluginMetadata


class LambdaRuntime(Enum):
    """Lambda runtime environments"""

    PYTHON39 = "python3.9"
    PYTHON310 = "python3.10"
    PYTHON311 = "python3.11"
    NODEJS18 = "nodejs18.x"
    NODEJS20 = "nodejs20.x"
    JAVA11 = "java11"
    JAVA17 = "java17"
    DOTNET6 = "dotnet6"


class LambdaInvocationType(Enum):
    """Lambda invocation types"""

    SYNCHRONOUS = "RequestResponse"
    ASYNCHRONOUS = "Event"
    DRY_RUN = "DryRun"


class LambdaLogType(Enum):
    """Lambda log types"""

    NONE = "None"
    TAIL = "Tail"


@dataclass
class LambdaFunction:
    """Lambda function information"""

    function_name: str
    function_arn: str
    runtime: LambdaRuntime
    handler: str
    code_size: int
    description: Optional[str]
    timeout: int
    memory_size: int
    last_modified: datetime
    version: str
    state: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["runtime"] = self.runtime.value
        data["last_modified"] = self.last_modified.isoformat()
        return data


@dataclass
class LambdaInvocationResult:
    """Lambda invocation result"""

    status_code: int
    payload: Optional[Dict[str, Any]]
    log_result: Optional[str]
    executed_version: str
    function_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class LambdaLayer:
    """Lambda layer information"""

    layer_name: str
    layer_arn: str
    version: int
    description: Optional[str]
    created_date: datetime
    compatible_runtimes: List[LambdaRuntime]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["created_date"] = self.created_date.isoformat()
        data["compatible_runtimes"] = [rt.value for rt in self.compatible_runtimes]
        return data


class AWSLambdaProcessingPlugin(ProcessingPlugin):
    """
    AWS Lambda processing integration plugin

    Provides comprehensive Lambda integration including:
    - Function deployment and management
    - Synchronous and asynchronous invocations
    - Event-driven processing
    - Layer management for dependencies
    - Integration with other AWS services
    - Monitoring and logging
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize AWS Lambda processing plugin"""
        super().__init__(config)

        # AWS configuration
        self.aws_region = config.get("aws_region", "us-east-1")
        self.aws_access_key_id = config.get("aws_access_key_id")
        self.aws_secret_access_key = config.get("aws_secret_access_key")
        self.aws_session_token = config.get("aws_session_token")

        # Lambda settings
        self.function_prefix = config.get("function_prefix", "medical-ai-")
        self.default_runtime = LambdaRuntime(config.get("default_runtime", "python3.11"))
        self.default_timeout = config.get("default_timeout", 300)  # 5 minutes
        self.default_memory = config.get("default_memory", 512)  # 512 MB

        # IAM settings
        self.execution_role_arn = config.get("execution_role_arn")
        self.vpc_config = config.get("vpc_config")  # For VPC access

        # S3 settings for code deployment
        self.code_bucket = config.get("code_bucket")
        self.code_key_prefix = config.get("code_key_prefix", "lambda-code/")

        # Connection settings
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)

        # AWS clients
        self.lambda_client = None
        self.s3_client = None
        self.session = None

        # Function registry
        self.registered_functions = {}
        self.function_handlers = {}

        # Layer management
        self.layers = {}

        self.logger = logging.getLogger(__name__)

    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        return PluginMetadata(
            name="aws-lambda-processing",
            version="1.0.0",
            description="AWS Lambda Serverless Processing Integration",
            vendor="Amazon Web Services",
            capabilities=[
                PluginCapability.SERVERLESS_PROCESSING,
                PluginCapability.EVENT_DRIVEN_PROCESSING,
                PluginCapability.SCALABLE_COMPUTE,
                PluginCapability.FUNCTION_DEPLOYMENT,
                PluginCapability.ASYNC_PROCESSING,
            ],
            supported_formats=["Python", "Node.js", "Java", ".NET", "Custom Runtime"],
            configuration_schema={
                "aws_region": {"type": "string", "required": True},
                "execution_role_arn": {"type": "string", "required": True},
                "code_bucket": {"type": "string", "required": True},
                "function_prefix": {"type": "string", "default": "medical-ai-"},
                "default_runtime": {
                    "type": "string",
                    "enum": ["python3.9", "python3.10", "python3.11"],
                    "default": "python3.11",
                },
                "aws_access_key_id": {"type": "string", "required": False, "sensitive": True},
                "aws_secret_access_key": {"type": "string", "required": False, "sensitive": True},
            },
        )

    async def initialize(self) -> bool:
        """Initialize AWS Lambda connection"""
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
            self.lambda_client = boto_session.client("lambda")
            self.s3_client = boto_session.client("s3")

            # Test connection
            if not await self._test_connection():
                self.logger.error("Lambda connection test failed")
                return False

            # Load existing functions
            await self._load_existing_functions()

            self.logger.info("AWS Lambda processing plugin initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"AWS Lambda initialization failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup resources"""
        # aioboto3 sessions are automatically cleaned up
        pass

    async def _test_connection(self) -> bool:
        """Test Lambda connection"""
        try:
            async with self.session.client("lambda") as lambda_client:
                # Test by listing functions
                response = await lambda_client.list_functions(MaxItems=1)
                return "Functions" in response

        except Exception as e:
            self.logger.error(f"Lambda connection test failed: {e}")
            return False

    async def _load_existing_functions(self):
        """Load existing Lambda functions"""
        try:
            async with self.session.client("lambda") as lambda_client:
                paginator = lambda_client.get_paginator("list_functions")

                async for page in paginator.paginate():
                    for func in page["Functions"]:
                        if func["FunctionName"].startswith(self.function_prefix):
                            function_info = LambdaFunction(
                                function_name=func["FunctionName"],
                                function_arn=func["FunctionArn"],
                                runtime=LambdaRuntime(func["Runtime"]),
                                handler=func["Handler"],
                                code_size=func["CodeSize"],
                                description=func.get("Description"),
                                timeout=func["Timeout"],
                                memory_size=func["MemorySize"],
                                last_modified=datetime.fromisoformat(
                                    func["LastModified"].replace("Z", "+00:00")
                                ),
                                version=func["Version"],
                                state=func["State"],
                            )

                            self.registered_functions[func["FunctionName"]] = function_info

            self.logger.info(f"Loaded {len(self.registered_functions)} existing functions")

        except Exception as e:
            self.logger.error(f"Error loading existing functions: {e}")

    def register_function_handler(self, function_name: str, handler: Callable):
        """Register local handler for function"""
        self.function_handlers[function_name] = handler

    async def deploy_function(
        self,
        function_name: str,
        code_path: str,
        handler: str,
        runtime: Optional[LambdaRuntime] = None,
        description: Optional[str] = None,
        timeout: Optional[int] = None,
        memory_size: Optional[int] = None,
        environment_variables: Optional[Dict[str, str]] = None,
        layers: Optional[List[str]] = None,
    ) -> Optional[LambdaFunction]:
        """Deploy Lambda function"""
        try:
            full_function_name = f"{self.function_prefix}{function_name}"
            runtime = runtime or self.default_runtime
            timeout = timeout or self.default_timeout
            memory_size = memory_size or self.default_memory

            # Package code
            code_package = await self._package_code(code_path)
            if not code_package:
                return None

            # Upload code to S3
            code_key = f"{self.code_key_prefix}{full_function_name}/{uuid.uuid4()}.zip"

            self.s3_client.put_object(Bucket=self.code_bucket, Key=code_key, Body=code_package)

            # Prepare function configuration
            function_config = {
                "FunctionName": full_function_name,
                "Runtime": runtime.value,
                "Role": self.execution_role_arn,
                "Handler": handler,
                "Code": {"S3Bucket": self.code_bucket, "S3Key": code_key},
                "Timeout": timeout,
                "MemorySize": memory_size,
                "Publish": True,
            }

            if description:
                function_config["Description"] = description

            if environment_variables:
                function_config["Environment"] = {"Variables": environment_variables}

            if layers:
                function_config["Layers"] = layers

            if self.vpc_config:
                function_config["VpcConfig"] = self.vpc_config

            # Create or update function
            async with self.session.client("lambda") as lambda_client:
                try:
                    # Try to update existing function
                    await lambda_client.get_function(FunctionName=full_function_name)

                    # Update function code
                    await lambda_client.update_function_code(
                        FunctionName=full_function_name,
                        S3Bucket=self.code_bucket,
                        S3Key=code_key,
                        Publish=True,
                    )

                    # Update function configuration
                    config_update = {
                        k: v
                        for k, v in function_config.items()
                        if k not in ["FunctionName", "Code"]
                    }

                    response = await lambda_client.update_function_configuration(
                        FunctionName=full_function_name, **config_update
                    )

                except lambda_client.exceptions.ResourceNotFoundException:
                    # Create new function
                    response = await lambda_client.create_function(**function_config)

                # Wait for function to be active
                waiter = lambda_client.get_waiter("function_active")
                await waiter.wait(FunctionName=full_function_name)

                # Create function info
                function_info = LambdaFunction(
                    function_name=response["FunctionName"],
                    function_arn=response["FunctionArn"],
                    runtime=LambdaRuntime(response["Runtime"]),
                    handler=response["Handler"],
                    code_size=response["CodeSize"],
                    description=response.get("Description"),
                    timeout=response["Timeout"],
                    memory_size=response["MemorySize"],
                    last_modified=datetime.fromisoformat(
                        response["LastModified"].replace("Z", "+00:00")
                    ),
                    version=response["Version"],
                    state=response["State"],
                )

                self.registered_functions[full_function_name] = function_info

                self.logger.info(f"Deployed function: {full_function_name}")
                return function_info

        except Exception as e:
            self.logger.error(f"Error deploying function {function_name}: {e}")
            return None

    async def _package_code(self, code_path: str) -> Optional[bytes]:
        """Package code into ZIP file"""
        try:
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                code_path_obj = Path(code_path)

                if code_path_obj.is_file():
                    # Single file
                    zip_file.write(code_path_obj, code_path_obj.name)
                elif code_path_obj.is_dir():
                    # Directory
                    for file_path in code_path_obj.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(code_path_obj)
                            zip_file.write(file_path, arcname)
                else:
                    raise ValueError(f"Code path does not exist: {code_path}")

            return zip_buffer.getvalue()

        except Exception as e:
            self.logger.error(f"Error packaging code: {e}")
            return None

    async def invoke_function(
        self,
        function_name: str,
        payload: Optional[Dict[str, Any]] = None,
        invocation_type: LambdaInvocationType = LambdaInvocationType.SYNCHRONOUS,
        log_type: LambdaLogType = LambdaLogType.NONE,
    ) -> Optional[LambdaInvocationResult]:
        """Invoke Lambda function"""
        try:
            full_function_name = f"{self.function_prefix}{function_name}"

            # Check if function exists locally
            if function_name in self.function_handlers:
                return await self._invoke_local_handler(function_name, payload)

            # Invoke remote Lambda function
            async with self.session.client("lambda") as lambda_client:
                invoke_params = {
                    "FunctionName": full_function_name,
                    "InvocationType": invocation_type.value,
                    "LogType": log_type.value,
                }

                if payload:
                    invoke_params["Payload"] = json.dumps(payload)

                response = await lambda_client.invoke(**invoke_params)

                # Parse response
                result_payload = None
                if "Payload" in response:
                    payload_data = await response["Payload"].read()
                    if payload_data:
                        try:
                            result_payload = json.loads(payload_data.decode("utf-8"))
                        except json.JSONDecodeError:
                            result_payload = {"raw_response": payload_data.decode("utf-8")}

                log_result = None
                if "LogResult" in response:
                    log_result = base64.b64decode(response["LogResult"]).decode("utf-8")

                return LambdaInvocationResult(
                    status_code=response["StatusCode"],
                    payload=result_payload,
                    log_result=log_result,
                    executed_version=response["ExecutedVersion"],
                    function_error=response.get("FunctionError"),
                )

        except Exception as e:
            self.logger.error(f"Error invoking function {function_name}: {e}")
            return None

    async def _invoke_local_handler(
        self, function_name: str, payload: Optional[Dict[str, Any]]
    ) -> LambdaInvocationResult:
        """Invoke local function handler"""
        try:
            handler = self.function_handlers[function_name]

            # Create Lambda-like event and context
            event = payload or {}
            context = type(
                "LambdaContext",
                (),
                {
                    "function_name": function_name,
                    "function_version": "$LATEST",
                    "invoked_function_arn": f"arn:aws:lambda:{self.aws_region}:123456789012:function:{function_name}",
                    "memory_limit_in_mb": "512",
                    "remaining_time_in_millis": lambda: 300000,
                    "log_group_name": f"/aws/lambda/{function_name}",
                    "log_stream_name": "2023/01/01/[$LATEST]abcdef123456",
                    "aws_request_id": str(uuid.uuid4()),
                },
            )()

            # Invoke handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(event, context)
            else:
                result = handler(event, context)

            return LambdaInvocationResult(
                status_code=200,
                payload=result,
                log_result=None,
                executed_version="$LATEST",
                function_error=None,
            )

        except Exception as e:
            self.logger.error(f"Error invoking local handler {function_name}: {e}")
            return LambdaInvocationResult(
                status_code=500,
                payload={"error": str(e)},
                log_result=None,
                executed_version="$LATEST",
                function_error="Unhandled",
            )

    async def create_layer(
        self,
        layer_name: str,
        code_path: str,
        compatible_runtimes: List[LambdaRuntime],
        description: Optional[str] = None,
    ) -> Optional[LambdaLayer]:
        """Create Lambda layer"""
        try:
            # Package layer code
            layer_package = await self._package_code(code_path)
            if not layer_package:
                return None

            async with self.session.client("lambda") as lambda_client:
                response = await lambda_client.publish_layer_version(
                    LayerName=layer_name,
                    Description=description or f"Layer for {layer_name}",
                    Content={"ZipFile": layer_package},
                    CompatibleRuntimes=[rt.value for rt in compatible_runtimes],
                )

                layer_info = LambdaLayer(
                    layer_name=response["LayerArn"].split(":")[-2],
                    layer_arn=response["LayerArn"],
                    version=response["Version"],
                    description=response.get("Description"),
                    created_date=datetime.fromisoformat(
                        response["CreatedDate"].replace("Z", "+00:00")
                    ),
                    compatible_runtimes=compatible_runtimes,
                )

                self.layers[layer_name] = layer_info

                self.logger.info(f"Created layer: {layer_name} version {response['Version']}")
                return layer_info

        except Exception as e:
            self.logger.error(f"Error creating layer {layer_name}: {e}")
            return None

    async def list_functions(self) -> List[LambdaFunction]:
        """List all registered functions"""
        return list(self.registered_functions.values())

    async def get_function_info(self, function_name: str) -> Optional[LambdaFunction]:
        """Get function information"""
        full_function_name = f"{self.function_prefix}{function_name}"
        return self.registered_functions.get(full_function_name)

    async def delete_function(self, function_name: str) -> bool:
        """Delete Lambda function"""
        try:
            full_function_name = f"{self.function_prefix}{function_name}"

            async with self.session.client("lambda") as lambda_client:
                await lambda_client.delete_function(FunctionName=full_function_name)

                # Remove from registry
                if full_function_name in self.registered_functions:
                    del self.registered_functions[full_function_name]

                self.logger.info(f"Deleted function: {full_function_name}")
                return True

        except Exception as e:
            self.logger.error(f"Error deleting function {function_name}: {e}")
            return False

    async def get_function_logs(
        self,
        function_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[str]:
        """Get function logs from CloudWatch"""
        try:
            full_function_name = f"{self.function_prefix}{function_name}"
            log_group_name = f"/aws/lambda/{full_function_name}"

            # This would require CloudWatch Logs integration
            # For now, return empty list
            self.logger.info(f"Log retrieval not implemented for {function_name}")
            return []

        except Exception as e:
            self.logger.error(f"Error getting function logs: {e}")
            return []

    async def validate_connection(self) -> Dict[str, Any]:
        """Validate Lambda connection"""
        try:
            # Test Lambda service access
            lambda_accessible = await self._test_connection()

            # Test function listing
            functions_accessible = False
            try:
                functions = await self.list_functions()
                functions_accessible = True
            except Exception as e:
                logger.warning(f"Lambda health check failed: {type(e).__name__}")
                pass

            # Test S3 code bucket access
            s3_accessible = False
            try:
                self.s3_client.head_bucket(Bucket=self.code_bucket)
                s3_accessible = True
            except Exception as e:
                pass

            # Test IAM role
            role_accessible = False
            try:
                iam_client = boto3.client("iam", region_name=self.aws_region)
                iam_client.get_role(RoleName=self.execution_role_arn.split("/")[-1])
                role_accessible = True
            except Exception as e:
                pass

            return {
                "connected": lambda_accessible and s3_accessible,
                "lambda_accessible": lambda_accessible,
                "functions_accessible": functions_accessible,
                "s3_accessible": s3_accessible,
                "role_accessible": role_accessible,
                "registered_functions": len(self.registered_functions),
                "local_handlers": len(self.function_handlers),
                "aws_region": self.aws_region,
                "last_check": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"connected": False, "error": str(e), "last_check": datetime.now().isoformat()}


# Example Lambda function handlers
async def image_processing_handler(event, context):
    """Example image processing Lambda handler"""
    try:
        # Extract image information from event
        bucket = event.get("bucket")
        key = event.get("key")

        # Process image (placeholder)
        result = {
            "processed": True,
            "bucket": bucket,
            "key": key,
            "timestamp": datetime.now().isoformat(),
            "function_name": context.function_name,
        }

        return result

    except Exception as e:
        return {"error": str(e)}


def dicom_analysis_handler(event, context):
    """Example DICOM analysis Lambda handler"""
    try:
        # Extract DICOM file information
        dicom_path = event.get("dicom_path")
        analysis_type = event.get("analysis_type", "basic")

        # Analyze DICOM (placeholder)
        result = {
            "analysis_complete": True,
            "dicom_path": dicom_path,
            "analysis_type": analysis_type,
            "findings": ["Normal tissue", "No abnormalities detected"],
            "confidence": 0.95,
            "timestamp": datetime.now().isoformat(),
        }

        return result

    except Exception as e:
        return {"error": str(e)}


# Plugin factory function
def create_plugin(config: Dict[str, Any]) -> AWSLambdaProcessingPlugin:
    """Create AWS Lambda processing plugin instance"""
    plugin = AWSLambdaProcessingPlugin(config)

    # Register example handlers
    plugin.register_function_handler("image-processing", image_processing_handler)
    plugin.register_function_handler("dicom-analysis", dicom_analysis_handler)

    return plugin
