"""
Azure Functions Integration

Provides integration with Azure Functions for serverless processing of HistoCore workloads:
- Slide preprocessing functions
- Model inference functions
- Result post-processing functions
- Event-driven processing
- Batch processing capabilities
- Function monitoring and management
"""

import asyncio
import base64
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

try:
    import aiohttp
    import requests
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.web import WebSiteManagementClient
    from azure.mgmt.web.models import AppServicePlan, Site, SiteConfig

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class FunctionRuntime(Enum):
    """Azure Functions runtime versions."""

    PYTHON_3_8 = "python|3.8"
    PYTHON_3_9 = "python|3.9"
    PYTHON_3_10 = "python|3.10"
    PYTHON_3_11 = "python|3.11"


class FunctionTriggerType(Enum):
    """Azure Functions trigger types."""

    HTTP = "httpTrigger"
    BLOB = "blobTrigger"
    QUEUE = "queueTrigger"
    TIMER = "timerTrigger"
    EVENT_HUB = "eventHubTrigger"
    SERVICE_BUS = "serviceBusTrigger"


@dataclass
class FunctionConfig:
    """Configuration for Azure Functions integration."""

    subscription_id: str
    resource_group: str
    function_app_name: str
    location: str = "East US"
    runtime: FunctionRuntime = FunctionRuntime.PYTHON_3_11
    use_managed_identity: bool = True
    timeout: int = 300  # 5 minutes
    max_concurrent_requests: int = 100


@dataclass
class FunctionDefinition:
    """Definition of an Azure Function."""

    name: str
    trigger_type: FunctionTriggerType
    code: str
    requirements: Optional[str] = None
    app_settings: Optional[Dict[str, str]] = None
    bindings: Optional[List[Dict]] = None
    timeout: Optional[int] = None


@dataclass
class FunctionInvocation:
    """Function invocation request."""

    function_name: str
    input_data: Dict[str, Any]
    headers: Optional[Dict[str, str]] = None
    query_params: Optional[Dict[str, str]] = None


@dataclass
class FunctionResult:
    """Function execution result."""

    success: bool
    status_code: int
    response_data: Optional[Dict] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    logs: Optional[List[str]] = None


class AzureFunctionsIntegration:
    """Azure Functions integration for serverless HistoCore processing."""

    def __init__(self, config: FunctionConfig):
        """Initialize Azure Functions integration."""
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure SDK not available. Install with: "
                "pip install azure-identity azure-mgmt-web requests aiohttp"
            )

        self.config = config
        self.credential = None
        self.web_client = None
        self.function_app_url = None
        self.access_key = None

        self._initialize_clients()
        logger.info(
            "Azure Functions integration initialized: app=%s, resource_group=%s",
            config.function_app_name,
            config.resource_group,
        )

    def _initialize_clients(self) -> None:
        """Initialize Azure management clients."""
        try:
            if self.config.use_managed_identity:
                self.credential = DefaultAzureCredential()

            self.web_client = WebSiteManagementClient(
                credential=self.credential, subscription_id=self.config.subscription_id
            )

            # Get function app URL
            self.function_app_url = f"https://{self.config.function_app_name}.azurewebsites.net"

            # Get function access key for authentication
            self._get_function_access_key()

            logger.info("Azure Functions clients initialized")

        except Exception as e:
            logger.error("Failed to initialize Azure Functions clients: %s", e)
            raise

    def _get_function_access_key(self) -> None:
        """Get function app master key for authentication."""
        try:
            # Get the master key for function authentication
            keys = self.web_client.web_apps.list_host_keys(
                resource_group_name=self.config.resource_group, name=self.config.function_app_name
            )

            if hasattr(keys, "master_key"):
                self.access_key = keys.master_key
            elif hasattr(keys, "function_keys") and keys.function_keys:
                # Use the first available function key
                self.access_key = list(keys.function_keys.values())[0]
            else:
                logger.warning("No access key found, using managed identity")

        except Exception as e:
            logger.warning("Failed to get function access key: %s", e)

    def create_function_app(self) -> bool:
        """Create Azure Function App if it doesn't exist."""
        try:
            # Check if function app already exists
            try:
                existing_app = self.web_client.web_apps.get(
                    resource_group_name=self.config.resource_group,
                    name=self.config.function_app_name,
                )
                logger.info("Function app %s already exists", self.config.function_app_name)
                return True
            except Exception:
                pass  # App doesn't exist, create it

            # Create App Service Plan for the function app
            plan_name = f"{self.config.function_app_name}-plan"

            app_service_plan = AppServicePlan(
                location=self.config.location,
                sku={"name": "Y1", "tier": "Dynamic"},  # Consumption plan
                kind="functionapp",
            )

            logger.info("Creating App Service Plan: %s", plan_name)
            plan_operation = self.web_client.app_service_plans.begin_create_or_update(
                resource_group_name=self.config.resource_group,
                name=plan_name,
                app_service_plan=app_service_plan,
            )
            plan_result = plan_operation.result()

            # Create Function App
            site_config = SiteConfig(
                app_settings=[
                    {
                        "name": "AzureWebJobsStorage",
                        "value": "DefaultEndpointsProtocol=https;AccountName=...",
                    },
                    {"name": "FUNCTIONS_EXTENSION_VERSION", "value": "~4"},
                    {"name": "FUNCTIONS_WORKER_RUNTIME", "value": "python"},
                    {"name": "WEBSITE_PYTHON_VERSION", "value": "3.11"},
                ],
                linux_fx_version=self.config.runtime.value,
            )

            site = Site(
                location=self.config.location,
                server_farm_id=plan_result.id,
                site_config=site_config,
                kind="functionapp,linux",
            )

            logger.info("Creating Function App: %s", self.config.function_app_name)
            create_operation = self.web_client.web_apps.begin_create_or_update(
                resource_group_name=self.config.resource_group,
                name=self.config.function_app_name,
                site_envelope=site,
            )

            result = create_operation.result()
            logger.info("Function app created successfully: %s", result.default_host_name)
            return True

        except Exception as e:
            logger.error("Failed to create function app: %s", e)
            return False

    def deploy_function(self, function_def: FunctionDefinition) -> bool:
        """Deploy a function to the Azure Function App."""
        try:
            # Create function.json configuration
            function_json = {
                "bindings": function_def.bindings
                or self._get_default_bindings(function_def.trigger_type),
                "scriptFile": "__init__.py",
            }

            if function_def.timeout:
                function_json["timeout"] = (
                    f"00:{function_def.timeout // 60:02d}:{function_def.timeout % 60:02d}"
                )

            # Prepare deployment package
            deployment_files = {
                f"{function_def.name}/function.json": json.dumps(function_json, indent=2),
                f"{function_def.name}/__init__.py": function_def.code,
            }

            if function_def.requirements:
                deployment_files["requirements.txt"] = function_def.requirements

            # Deploy using Kudu API
            success = self._deploy_via_kudu(deployment_files)

            if success:
                logger.info("Function %s deployed successfully", function_def.name)

                # Update app settings if provided
                if function_def.app_settings:
                    self._update_app_settings(function_def.app_settings)

            return success

        except Exception as e:
            logger.error("Failed to deploy function %s: %s", function_def.name, e)
            return False

    def _get_default_bindings(self, trigger_type: FunctionTriggerType) -> List[Dict]:
        """Get default bindings for trigger type."""
        if trigger_type == FunctionTriggerType.HTTP:
            return [
                {
                    "authLevel": "function",
                    "type": "httpTrigger",
                    "direction": "in",
                    "name": "req",
                    "methods": ["get", "post"],
                },
                {"type": "http", "direction": "out", "name": "$return"},
            ]
        elif trigger_type == FunctionTriggerType.BLOB:
            return [
                {
                    "type": "blobTrigger",
                    "direction": "in",
                    "name": "blob",
                    "path": "histocore/{name}",
                    "connection": "AzureWebJobsStorage",
                }
            ]
        elif trigger_type == FunctionTriggerType.QUEUE:
            return [
                {
                    "type": "queueTrigger",
                    "direction": "in",
                    "name": "msg",
                    "queueName": "histocore-queue",
                    "connection": "AzureWebJobsStorage",
                }
            ]
        elif trigger_type == FunctionTriggerType.TIMER:
            return [
                {
                    "type": "timerTrigger",
                    "direction": "in",
                    "name": "timer",
                    "schedule": "0 */5 * * * *",  # Every 5 minutes
                }
            ]
        else:
            return []

    def _deploy_via_kudu(self, files: Dict[str, str]) -> bool:
        """Deploy function files via Kudu API."""
        try:
            kudu_url = f"https://{self.config.function_app_name}.scm.azurewebsites.net"

            # Get publishing credentials
            publish_profile = self.web_client.web_apps.list_publishing_credentials(
                resource_group_name=self.config.resource_group, name=self.config.function_app_name
            )

            username = publish_profile.publishing_user_name
            password = publish_profile.publishing_password

            # Deploy each file
            for file_path, content in files.items():
                file_url = f"{kudu_url}/api/vfs/site/wwwroot/{file_path}"

                response = requests.put(
                    file_url,
                    data=content.encode("utf-8"),
                    auth=(username, password),
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=self.config.timeout,
                )
                response.raise_for_status()

            logger.debug("Files deployed via Kudu API")
            return True

        except Exception as e:
            logger.error("Kudu deployment failed: %s", e)
            return False

    def _update_app_settings(self, settings: Dict[str, str]) -> None:
        """Update function app settings."""
        try:
            # Get current settings
            current_settings = self.web_client.web_apps.list_application_settings(
                resource_group_name=self.config.resource_group, name=self.config.function_app_name
            )

            # Merge with new settings
            updated_settings = dict(current_settings.properties)
            updated_settings.update(settings)

            # Update settings
            self.web_client.web_apps.update_application_settings(
                resource_group_name=self.config.resource_group,
                name=self.config.function_app_name,
                app_settings={"properties": updated_settings},
            )

            logger.debug("App settings updated")

        except Exception as e:
            logger.error("Failed to update app settings: %s", e)

    def _validate_function_url(self, url: str) -> None:
        """
        Validate function URL to prevent SSRF attacks.

        Args:
            url: URL to validate

        Raises:
            ValueError: If URL is not in allowed domains
        """
        parsed = urlparse(url)

        # Define allowed domains for Azure Functions
        allowed_domains = [
            f"{self.config.function_app_name}.azurewebsites.net",
            "azure-api.net",
            "azurewebsites.net",
        ]

        # Check if hostname ends with any allowed domain
        hostname = parsed.netloc.lower()
        if not any(hostname.endswith(domain) for domain in allowed_domains):
            raise ValueError(
                f"Function URL not in allowed domains: {url}. "
                f"Hostname must end with one of: {allowed_domains}"
            )

        # Ensure HTTPS
        if parsed.scheme != "https":
            raise ValueError(f"Function URL must use HTTPS: {url}")

        logger.debug("Function URL validated: %s", url)

    def invoke_function(self, invocation: FunctionInvocation) -> FunctionResult:
        """Invoke an Azure Function synchronously."""
        start_time = datetime.now()

        try:
            function_url = f"{self.function_app_url}/api/{invocation.function_name}"

            # Validate URL to prevent SSRF
            self._validate_function_url(function_url)

            headers = {"Content-Type": "application/json", **(invocation.headers or {})}

            # Add authentication
            if self.access_key:
                headers["x-functions-key"] = self.access_key

            response = requests.post(
                function_url,
                json=invocation.input_data,
                headers=headers,
                params=invocation.query_params,
                timeout=self.config.timeout,
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            if response.status_code == 200:
                try:
                    response_data = response.json()
                except (ValueError, requests.exceptions.JSONDecodeError):
                    response_data = {"result": response.text}

                return FunctionResult(
                    success=True,
                    status_code=response.status_code,
                    response_data=response_data,
                    execution_time=execution_time,
                )
            else:
                return FunctionResult(
                    success=False,
                    status_code=response.status_code,
                    error_message=response.text,
                    execution_time=execution_time,
                )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Function invocation failed: {e}"
            logger.error(error_msg)

            return FunctionResult(
                success=False,
                status_code=500,
                error_message=error_msg,
                execution_time=execution_time,
            )

    async def invoke_function_async(self, invocation: FunctionInvocation) -> FunctionResult:
        """Invoke an Azure Function asynchronously."""
        start_time = datetime.now()

        try:
            function_url = f"{self.function_app_url}/api/{invocation.function_name}"

            headers = {"Content-Type": "application/json", **(invocation.headers or {})}

            # Add authentication
            if self.access_key:
                headers["x-functions-key"] = self.access_key

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    function_url,
                    json=invocation.input_data,
                    headers=headers,
                    params=invocation.query_params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:

                    execution_time = (datetime.now() - start_time).total_seconds()
                    response_text = await response.text()

                    if response.status == 200:
                        try:
                            response_data = await response.json()
                        except (ValueError, aiohttp.ContentTypeError):
                            response_data = {"result": response_text}

                        return FunctionResult(
                            success=True,
                            status_code=response.status,
                            response_data=response_data,
                            execution_time=execution_time,
                        )
                    else:
                        return FunctionResult(
                            success=False,
                            status_code=response.status,
                            error_message=response_text,
                            execution_time=execution_time,
                        )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Async function invocation failed: {e}"
            logger.error(error_msg)

            return FunctionResult(
                success=False,
                status_code=500,
                error_message=error_msg,
                execution_time=execution_time,
            )

    def invoke_functions_batch(self, invocations: List[FunctionInvocation]) -> List[FunctionResult]:
        """Invoke multiple functions concurrently."""

        async def run_batch():
            tasks = [self.invoke_function_async(inv) for inv in invocations]
            return await asyncio.gather(*tasks, return_exceptions=True)

        try:
            results = asyncio.run(run_batch())

            # Convert exceptions to error results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(
                        FunctionResult(success=False, status_code=500, error_message=str(result))
                    )
                else:
                    processed_results.append(result)

            successful_invocations = sum(1 for r in processed_results if r.success)
            logger.info(
                "Batch function invocation completed: %d/%d successful",
                successful_invocations,
                len(invocations),
            )

            return processed_results

        except Exception as e:
            logger.error("Batch function invocation failed: %s", e)
            return [
                FunctionResult(success=False, status_code=500, error_message=str(e))
                for _ in invocations
            ]

    def get_function_logs(self, function_name: str, hours: int = 1) -> List[str]:
        """Get function execution logs."""
        try:
            # Use Kudu API to get logs
            kudu_url = f"https://{self.config.function_app_name}.scm.azurewebsites.net"

            # Get publishing credentials
            publish_profile = self.web_client.web_apps.list_publishing_credentials(
                resource_group_name=self.config.resource_group, name=self.config.function_app_name
            )

            username = publish_profile.publishing_user_name
            password = publish_profile.publishing_password

            # Get logs from Kudu
            logs_url = f"{kudu_url}/api/logs/recent"

            # Validate URL to prevent SSRF
            self._validate_function_url(logs_url)

            response = requests.get(
                logs_url, auth=(username, password), timeout=self.config.timeout
            )
            response.raise_for_status()

            logs_data = response.json()

            # Filter logs for the specific function
            function_logs = []
            for log_entry in logs_data:
                if function_name in log_entry.get("message", ""):
                    function_logs.append(log_entry["message"])

            logger.debug(
                "Retrieved %d log entries for function %s", len(function_logs), function_name
            )
            return function_logs

        except Exception as e:
            logger.error("Failed to get function logs: %s", e)
            return []

    def get_function_metrics(self, function_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get function execution metrics."""
        try:
            # This would typically integrate with Azure Monitor
            # For now, return basic metrics structure
            metrics = {
                "function_name": function_name,
                "time_range_hours": hours,
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_duration_ms": 0,
                "total_duration_ms": 0,
                "error_rate": 0.0,
            }

            logger.debug("Retrieved metrics for function %s", function_name)
            return metrics

        except Exception as e:
            logger.error("Failed to get function metrics: %s", e)
            return {}

    def list_functions(self) -> List[str]:
        """List all functions in the function app."""
        try:
            functions = self.web_client.web_apps.list_functions(
                resource_group_name=self.config.resource_group, name=self.config.function_app_name
            )

            function_names = [func.name for func in functions]
            logger.debug(
                "Found %d functions in app %s", len(function_names), self.config.function_app_name
            )
            return function_names

        except Exception as e:
            logger.error("Failed to list functions: %s", e)
            return []

    def delete_function(self, function_name: str) -> bool:
        """Delete a function from the function app."""
        try:
            self.web_client.web_apps.delete_function(
                resource_group_name=self.config.resource_group,
                name=self.config.function_app_name,
                function_name=function_name,
            )

            logger.info("Deleted function: %s", function_name)
            return True

        except Exception as e:
            logger.error("Failed to delete function %s: %s", function_name, e)
            return False

    def health_check(self) -> bool:
        """Check Azure Functions connectivity and health."""
        try:
            # Try to get function app properties
            app = self.web_client.web_apps.get(
                resource_group_name=self.config.resource_group, name=self.config.function_app_name
            )

            if app.state == "Running":
                logger.info("Azure Functions health check passed")
                return True
            else:
                logger.warning("Function app is not running: %s", app.state)
                return False

        except Exception as e:
            logger.error("Azure Functions health check failed: %s", e)
            return False


# Predefined function templates for HistoCore
def get_slide_preprocessing_function() -> FunctionDefinition:
    """Get slide preprocessing function definition."""
    code = '''
import logging
import json
import azure.functions as func

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Preprocess WSI slide for analysis."""
    logging.info('Slide preprocessing function triggered.')
    
    try:
        req_body = req.get_json()
        slide_path = req_body.get('slide_path')
        preprocessing_params = req_body.get('params', {})
        
        # Slide preprocessing logic would go here
        result = {
            'status': 'success',
            'processed_slide_path': f'processed_{slide_path}',
            'preprocessing_time': 45.2,
            'tile_count': 1024
        }
        
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f'Preprocessing failed: {str(e)}')
        return func.HttpResponse(
            json.dumps({'error': str(e)}),
            status_code=500,
            mimetype="application/json"
        )
'''

    return FunctionDefinition(
        name="slide-preprocessing",
        trigger_type=FunctionTriggerType.HTTP,
        code=code,
        requirements="azure-functions\nnumpy\nPillow",
        timeout=300,
    )


def get_model_inference_function() -> FunctionDefinition:
    """Get model inference function definition."""
    code = '''
import logging
import json
import azure.functions as func

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Run AI model inference on slide tiles."""
    logging.info('Model inference function triggered.')
    
    try:
        req_body = req.get_json()
        slide_id = req_body.get('slide_id')
        model_name = req_body.get('model_name', 'foundation_model')
        
        # Model inference logic would go here
        result = {
            'status': 'success',
            'slide_id': slide_id,
            'predictions': {
                'disease_type': 'breast_cancer',
                'confidence': 0.92,
                'grade': 'Grade 2',
                'stage': 'T2N0M0'
            },
            'inference_time': 12.5,
            'explanation': 'High confidence prediction based on glandular morphology'
        }
        
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f'Inference failed: {str(e)}')
        return func.HttpResponse(
            json.dumps({'error': str(e)}),
            status_code=500,
            mimetype="application/json"
        )
'''

    return FunctionDefinition(
        name="model-inference",
        trigger_type=FunctionTriggerType.HTTP,
        code=code,
        requirements="azure-functions\ntorch\nnumpy",
        timeout=600,
    )


# Factory function for easy initialization
def create_functions_integration(
    subscription_id: str, resource_group: str, function_app_name: str, **kwargs
) -> AzureFunctionsIntegration:
    """Create Azure Functions integration with configuration."""
    config = FunctionConfig(
        subscription_id=subscription_id,
        resource_group=resource_group,
        function_app_name=function_app_name,
        **kwargs,
    )
    return AzureFunctionsIntegration(config)
