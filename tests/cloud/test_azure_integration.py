"""
Tests for Azure Cloud Integration

Comprehensive test suite for all Azure cloud services integration:
- Health Data Services
- Blob Storage Connector
- Functions Integration
- Monitor Integration
"""

from unittest.mock import Mock, patch

import pytest

from src.platform.cloud.azure.blob_storage import (
    AzureBlobStorageConnector,
    BlobStorageConfig,
    create_blob_storage_connector,
)
from src.platform.cloud.azure.functions import (
    AzureFunctionsIntegration,
    FunctionConfig,
    FunctionDefinition,
    FunctionInvocation,
    FunctionTriggerType,
    create_functions_integration,
    get_model_inference_function,
    get_slide_preprocessing_function,
)

# Import Azure integration modules
from src.platform.cloud.azure.health_data_services import (
    AzureHealthDataServices,
    DiagnosticReportData,
    HealthDataConfig,
    PatientData,
    create_health_data_services,
)
from src.platform.cloud.azure.monitor import (
    AzureMonitorIntegration,
)
from src.platform.cloud.azure.monitor import CustomMetric as MonitorMetric
from src.platform.cloud.azure.monitor import (
    MetricType,
    MonitorConfig,
    create_monitor_integration,
    setup_histocore_monitoring,
)


class TestAzureHealthDataServices:
    """Test Azure Health Data Services integration."""

    @pytest.fixture
    def health_config(self):
        """Health Data Services configuration fixture."""
        return HealthDataConfig(
            workspace_url="https://test-workspace.healthcareapis.azure.com",
            fhir_service_name="test-fhir",
            tenant_id="test-tenant-id",
            use_managed_identity=True,
        )

    @pytest.fixture
    def patient_data(self):
        """Patient data fixture."""
        return PatientData(
            patient_id="patient-123",
            given_name="John",
            family_name="Doe",
            birth_date="1980-01-01",
            gender="male",
            mrn="MRN123456",
        )

    @pytest.fixture
    def diagnostic_report_data(self):
        """Diagnostic report data fixture."""
        return DiagnosticReportData(
            report_id="report-123",
            patient_id="patient-123",
            specimen_id="specimen-123",
            practitioner_id="practitioner-123",
            status="final",
            category="pathology",
            code="33717-0",
            conclusion="Invasive ductal carcinoma, Grade 2",
            confidence_score=0.92,
        )

    @patch("src.cloud.azure.health_data_services.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.health_data_services.DefaultAzureCredential")
    @patch("src.cloud.azure.health_data_services.requests")
    def test_initialization(self, mock_requests, mock_credential, health_config):
        """Test Health Data Services initialization."""
        mock_credential.return_value = Mock()

        service = AzureHealthDataServices(health_config)

        assert service.config == health_config
        assert service.base_url == f"{health_config.workspace_url}/fhir"
        mock_credential.assert_called_once()

    @patch("src.cloud.azure.health_data_services.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.health_data_services.DefaultAzureCredential")
    @patch("src.cloud.azure.health_data_services.requests")
    def test_create_patient(self, mock_requests, mock_credential, health_config, patient_data):
        """Test FHIR Patient resource creation."""
        mock_credential.return_value = Mock()
        mock_token = Mock()
        mock_token.token = "test-token"
        mock_token.expires_on = 9999999999
        mock_credential.return_value.get_token.return_value = mock_token

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"resourceType": "Patient", "id": "patient-123"}
        mock_requests.request.return_value = mock_response

        service = AzureHealthDataServices(health_config)
        result = service.create_patient(patient_data)

        assert result["resourceType"] == "Patient"
        assert result["id"] == "patient-123"
        mock_requests.request.assert_called_once()

    @patch("src.cloud.azure.health_data_services.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.health_data_services.DefaultAzureCredential")
    @patch("src.cloud.azure.health_data_services.requests")
    def test_create_diagnostic_report(
        self, mock_requests, mock_credential, health_config, diagnostic_report_data
    ):
        """Test FHIR DiagnosticReport resource creation."""
        mock_credential.return_value = Mock()
        mock_token = Mock()
        mock_token.token = "test-token"
        mock_token.expires_on = 9999999999
        mock_credential.return_value.get_token.return_value = mock_token

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"resourceType": "DiagnosticReport", "id": "report-123"}
        mock_requests.request.return_value = mock_response

        service = AzureHealthDataServices(health_config)
        result = service.create_diagnostic_report(diagnostic_report_data)

        assert result["resourceType"] == "DiagnosticReport"
        assert result["id"] == "report-123"

    @patch("src.cloud.azure.health_data_services.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.health_data_services.DefaultAzureCredential")
    @patch("src.cloud.azure.health_data_services.requests")
    def test_health_check(self, mock_requests, mock_credential, health_config):
        """Test Health Data Services health check."""
        mock_credential.return_value = Mock()
        mock_token = Mock()
        mock_token.token = "test-token"
        mock_token.expires_on = 9999999999
        mock_credential.return_value.get_token.return_value = mock_token

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"resourceType": "CapabilityStatement"}
        mock_requests.request.return_value = mock_response

        service = AzureHealthDataServices(health_config)
        result = service.health_check()

        assert result is True

    def test_factory_function(self):
        """Test factory function for Health Data Services."""
        service = create_health_data_services(
            workspace_url="https://test.healthcareapis.azure.com",
            fhir_service_name="test-fhir",
            tenant_id="test-tenant",
        )

        assert isinstance(service, AzureHealthDataServices)
        assert service.config.workspace_url == "https://test.healthcareapis.azure.com"


class TestAzureBlobStorageConnector:
    """Test Azure Blob Storage connector."""

    @pytest.fixture
    def blob_config(self):
        """Blob Storage configuration fixture."""
        return BlobStorageConfig(
            account_name="testaccount", container_name="testcontainer", use_managed_identity=True
        )

    @patch("src.cloud.azure.blob_storage.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.blob_storage.DefaultAzureCredential")
    @patch("src.cloud.azure.blob_storage.BlobServiceClient")
    def test_initialization(self, mock_blob_client, mock_credential, blob_config):
        """Test Blob Storage connector initialization."""
        mock_credential.return_value = Mock()
        mock_service_client = Mock()
        mock_container_client = Mock()
        mock_blob_client.return_value = mock_service_client
        mock_service_client.get_container_client.return_value = mock_container_client

        connector = AzureBlobStorageConnector(blob_config)

        assert connector.config == blob_config
        mock_credential.assert_called_once()

    @patch("src.cloud.azure.blob_storage.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.blob_storage.DefaultAzureCredential")
    @patch("src.cloud.azure.blob_storage.BlobServiceClient")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("builtins.open")
    def test_upload_file(
        self, mock_open, mock_getsize, mock_exists, mock_blob_client, mock_credential, blob_config
    ):
        """Test file upload to Blob Storage."""
        mock_credential.return_value = Mock()
        mock_service_client = Mock()
        mock_container_client = Mock()
        mock_blob_client_instance = Mock()

        mock_blob_client.return_value = mock_service_client
        mock_service_client.get_container_client.return_value = mock_container_client
        mock_container_client.get_blob_client.return_value = mock_blob_client_instance

        mock_exists.return_value = True
        mock_getsize.return_value = 1024
        mock_open.return_value.__enter__.return_value = Mock()

        mock_upload_result = {"etag": "test-etag"}
        mock_blob_client_instance.upload_blob.return_value = mock_upload_result

        connector = AzureBlobStorageConnector(blob_config)
        result = connector.upload_file("test.txt", "test-blob.txt")

        assert result.success is True
        assert result.etag == "test-etag"
        assert result.bytes_uploaded == 1024

    @patch("src.cloud.azure.blob_storage.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.blob_storage.DefaultAzureCredential")
    @patch("src.cloud.azure.blob_storage.BlobServiceClient")
    def test_health_check(self, mock_blob_client, mock_credential, blob_config):
        """Test Blob Storage health check."""
        mock_credential.return_value = Mock()
        mock_service_client = Mock()
        mock_container_client = Mock()

        mock_blob_client.return_value = mock_service_client
        mock_service_client.get_container_client.return_value = mock_container_client
        mock_container_client.get_container_properties.return_value = {}

        connector = AzureBlobStorageConnector(blob_config)
        result = connector.health_check()

        assert result is True

    def test_factory_function(self):
        """Test factory function for Blob Storage connector."""
        connector = create_blob_storage_connector(
            account_name="testaccount", container_name="testcontainer"
        )

        assert isinstance(connector, AzureBlobStorageConnector)
        assert connector.config.account_name == "testaccount"


class TestAzureFunctionsIntegration:
    """Test Azure Functions integration."""

    @pytest.fixture
    def function_config(self):
        """Functions configuration fixture."""
        return FunctionConfig(
            subscription_id="test-subscription",
            resource_group="test-rg",
            function_app_name="test-function-app",
        )

    @pytest.fixture
    def function_definition(self):
        """Function definition fixture."""
        return FunctionDefinition(
            name="test-function",
            trigger_type=FunctionTriggerType.HTTP,
            code="def main(req): return 'Hello World'",
        )

    @patch("src.cloud.azure.functions.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.functions.DefaultAzureCredential")
    @patch("src.cloud.azure.functions.WebSiteManagementClient")
    def test_initialization(self, mock_web_client, mock_credential, function_config):
        """Test Functions integration initialization."""
        mock_credential.return_value = Mock()
        mock_web_client.return_value = Mock()

        integration = AzureFunctionsIntegration(function_config)

        assert integration.config == function_config
        assert (
            integration.function_app_url
            == f"https://{function_config.function_app_name}.azurewebsites.net"
        )

    @patch("src.cloud.azure.functions.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.functions.DefaultAzureCredential")
    @patch("src.cloud.azure.functions.WebSiteManagementClient")
    @patch("src.cloud.azure.functions.requests")
    def test_invoke_function(
        self, mock_requests, mock_web_client, mock_credential, function_config
    ):
        """Test function invocation."""
        mock_credential.return_value = Mock()
        mock_web_client.return_value = Mock()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_requests.post.return_value = mock_response

        integration = AzureFunctionsIntegration(function_config)

        invocation = FunctionInvocation(function_name="test-function", input_data={"test": "data"})

        result = integration.invoke_function(invocation)

        assert result.success is True
        assert result.status_code == 200
        assert result.response_data["result"] == "success"

    @patch("src.cloud.azure.functions.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.functions.DefaultAzureCredential")
    @patch("src.cloud.azure.functions.WebSiteManagementClient")
    def test_health_check(self, mock_web_client, mock_credential, function_config):
        """Test Functions health check."""
        mock_credential.return_value = Mock()
        mock_web_client_instance = Mock()
        mock_web_client.return_value = mock_web_client_instance

        mock_app = Mock()
        mock_app.state = "Running"
        mock_web_client_instance.web_apps.get.return_value = mock_app

        integration = AzureFunctionsIntegration(function_config)
        result = integration.health_check()

        assert result is True

    def test_predefined_functions(self):
        """Test predefined function templates."""
        preprocessing_func = get_slide_preprocessing_function()
        assert preprocessing_func.name == "slide-preprocessing"
        assert preprocessing_func.trigger_type == FunctionTriggerType.HTTP
        assert "def main" in preprocessing_func.code

        inference_func = get_model_inference_function()
        assert inference_func.name == "model-inference"
        assert inference_func.trigger_type == FunctionTriggerType.HTTP
        assert "def main" in inference_func.code

    def test_factory_function(self):
        """Test factory function for Functions integration."""
        integration = create_functions_integration(
            subscription_id="test-subscription",
            resource_group="test-rg",
            function_app_name="test-app",
        )

        assert isinstance(integration, AzureFunctionsIntegration)
        assert integration.config.subscription_id == "test-subscription"


class TestAzureMonitorIntegration:
    """Test Azure Monitor integration."""

    @pytest.fixture
    def monitor_config(self):
        """Monitor configuration fixture."""
        return MonitorConfig(
            subscription_id="test-subscription",
            resource_group="test-rg",
            workspace_name="test-workspace",
            application_insights_key="test-key",
        )

    @pytest.fixture
    def custom_metric(self):
        """Custom metric fixture."""
        return MonitorMetric(
            name="test.metric",
            value=42.0,
            metric_type=MetricType.GAUGE,
            dimensions={"test": "value"},
        )

    @patch("src.cloud.azure.monitor.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.monitor.DefaultAzureCredential")
    @patch("src.cloud.azure.monitor.MonitorManagementClient")
    @patch("src.cloud.azure.monitor.LogAnalyticsManagementClient")
    def test_initialization(
        self, mock_log_client, mock_monitor_client, mock_credential, monitor_config
    ):
        """Test Monitor integration initialization."""
        mock_credential.return_value = Mock()
        mock_monitor_client.return_value = Mock()
        mock_log_client.return_value = Mock()

        integration = AzureMonitorIntegration(monitor_config)

        assert integration.config == monitor_config
        mock_credential.assert_called_once()

    @patch("src.cloud.azure.monitor.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.monitor.DefaultAzureCredential")
    @patch("src.cloud.azure.monitor.MonitorManagementClient")
    @patch("src.cloud.azure.monitor.LogAnalyticsManagementClient")
    def test_send_custom_metric(
        self, mock_log_client, mock_monitor_client, mock_credential, monitor_config, custom_metric
    ):
        """Test sending custom metrics."""
        mock_credential.return_value = Mock()
        mock_monitor_client.return_value = Mock()
        mock_log_client.return_value = Mock()

        integration = AzureMonitorIntegration(monitor_config)
        integration.send_custom_metric(custom_metric)

        # Metric should be queued
        assert not integration.metrics_queue.empty()

    @patch("src.cloud.azure.monitor.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.monitor.DefaultAzureCredential")
    @patch("src.cloud.azure.monitor.MonitorManagementClient")
    @patch("src.cloud.azure.monitor.LogAnalyticsManagementClient")
    def test_track_slide_processing(
        self, mock_log_client, mock_monitor_client, mock_credential, monitor_config
    ):
        """Test slide processing tracking."""
        mock_credential.return_value = Mock()
        mock_monitor_client.return_value = Mock()
        mock_log_client.return_value = Mock()

        integration = AzureMonitorIntegration(monitor_config)
        integration.track_slide_processing(slide_id="slide-123", processing_time=25.5, success=True)

        # Should have queued metrics and logs
        assert not integration.metrics_queue.empty()
        assert not integration.logs_queue.empty()

    @patch("src.cloud.azure.monitor.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.monitor.DefaultAzureCredential")
    @patch("src.cloud.azure.monitor.MonitorManagementClient")
    @patch("src.cloud.azure.monitor.LogAnalyticsManagementClient")
    def test_health_check(
        self, mock_log_client, mock_monitor_client, mock_credential, monitor_config
    ):
        """Test Monitor health check."""
        mock_credential.return_value = Mock()
        mock_monitor_client.return_value = Mock()
        mock_log_client.return_value = Mock()

        integration = AzureMonitorIntegration(monitor_config)
        result = integration.health_check()

        # Should pass since we're not actually sending data in test
        assert result is True

    def test_factory_function(self):
        """Test factory function for Monitor integration."""
        integration = create_monitor_integration(
            subscription_id="test-subscription",
            resource_group="test-rg",
            workspace_name="test-workspace",
        )

        assert isinstance(integration, AzureMonitorIntegration)
        assert integration.config.subscription_id == "test-subscription"

    def test_setup_histocore_monitoring(self):
        """Test complete HistoCore monitoring setup."""
        integration = setup_histocore_monitoring(
            subscription_id="test-subscription",
            resource_group="test-rg",
            workspace_name="test-workspace",
            application_insights_key="test-key",
        )

        assert isinstance(integration, AzureMonitorIntegration)
        assert integration.config.enable_application_insights is True
        assert integration.config.enable_log_analytics is True


class TestAzureIntegrationEnd2End:
    """End-to-end integration tests for Azure services."""

    @patch("src.cloud.azure.health_data_services.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.blob_storage.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.functions.AZURE_AVAILABLE", True)
    @patch("src.cloud.azure.monitor.AZURE_AVAILABLE", True)
    def test_complete_azure_workflow(self):
        """Test complete Azure integration workflow."""
        # This would test a complete workflow using all Azure services
        # For now, just verify all services can be imported and initialized

        from src.platform.cloud.azure import (
            AzureBlobStorageConnector,
            AzureFunctionsIntegration,
            AzureHealthDataServices,
            AzureMonitorIntegration,
        )

        # Verify all classes are available
        assert AzureHealthDataServices is not None
        assert AzureBlobStorageConnector is not None
        assert AzureFunctionsIntegration is not None
        assert AzureMonitorIntegration is not None

    def test_azure_module_imports(self):
        """Test Azure module imports work correctly."""
        # Verify __all__ exports
        import src.cloud.azure as azure_module

        expected_exports = [
            "AzureHealthDataServices",
            "AzureBlobStorageConnector",
            "AzureFunctionsIntegration",
            "AzureMonitorIntegration",
        ]

        for export in expected_exports:
            assert hasattr(azure_module, export)


if __name__ == "__main__":
    pytest.main([__file__])
