"""
PACS Integration Service - Main orchestration service for HistoCore PACS integration.

This module provides the main PACSService class that orchestrates all PACS components
and integrates with existing HistoCore infrastructure.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..workflow import ClinicalWorkflowSystem
from .audit_logger import PACSAuditLogger
from .configuration_manager import ConfigurationManager
from .error_handling import DeadLetterQueue, DicomErrorHandler, NetworkErrorHandler
from .failover import FailoverManager
from .notification_system import NotificationSystem
from .pacs_adapter import PACSAdapter
from .query_engine import QueryEngine
from .retrieval_engine import RetrievalEngine
from .security_manager import SecurityManager
from .storage_engine import StorageEngine
from .workflow_orchestrator import WorkflowOrchestrator

logger = logging.getLogger(__name__)


class PACSService:
    """
    Main PACS Integration Service that orchestrates all components.

    This service provides:
    - Component initialization and lifecycle management
    - Integration with existing HistoCore components
    - Service startup, shutdown, and health checks
    - Configuration loading and validation
    - Centralized error handling and monitoring
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        profile: str = "default",
        clinical_workflow: Optional[ClinicalWorkflowSystem] = None,
    ):
        """
        Initialize PACS Integration Service.

        Args:
            config_path: Path to configuration file (default: .kiro/pacs/config.yaml)
            profile: Configuration profile to use (default, production, staging, dev)
            clinical_workflow: Existing clinical workflow system instance
        """
        self.config_path = config_path or Path(".kiro/pacs/config.yaml")
        self.profile = profile
        self._is_running = False

        # Initialize components
        logger.info("Initializing PACS Integration Service")

        # Configuration
        self.config_manager = ConfigurationManager(config_path=self.config_path)
        self.config = self.config_manager.load_configuration(profile=profile)

        # Security
        self.security_manager = SecurityManager(
            ca_bundle_path=self.config.security.ca_bundle_path,
            client_cert_path=self.config.security.client_cert_path,
            client_key_path=self.config.security.client_key_path,
        )

        # Error handling
        self.network_error_handler = NetworkErrorHandler(
            max_retries=self.config.error_handling.max_retries,
            initial_backoff=self.config.error_handling.initial_backoff_seconds,
        )
        self.dicom_error_handler = DicomErrorHandler()
        self.dead_letter_queue = DeadLetterQueue(
            storage_path=Path(self.config.error_handling.dead_letter_queue_path)
        )

        # Failover
        self.failover_manager = FailoverManager(
            endpoints=self.config.pacs_endpoints,
            health_check_interval=self.config.failover.health_check_interval_seconds,
        )

        # Audit logging
        self.audit_logger = PACSAuditLogger(
            storage_path=str(Path(self.config.audit.log_directory)),
            retention_years=self.config.audit.retention_days // 365,
        )

        # Notification system
        self.notification_system = NotificationSystem(
            email_config=self.config.notifications.email,
            sms_config=self.config.notifications.sms,
            hl7_config=self.config.notifications.hl7,
        )

        # Core PACS components
        self.query_engine = QueryEngine(
            security_manager=self.security_manager,
            audit_logger=self.audit_logger,
        )

        self.retrieval_engine = RetrievalEngine(
            security_manager=self.security_manager,
            audit_logger=self.audit_logger,
            storage_path=Path(self.config.storage.local_cache_path),
        )

        self.storage_engine = StorageEngine(
            security_manager=self.security_manager,
            audit_logger=self.audit_logger,
        )

        # PACS adapter
        self.pacs_adapter = PACSAdapter(
            query_engine=self.query_engine,
            retrieval_engine=self.retrieval_engine,
            storage_engine=self.storage_engine,
            security_manager=self.security_manager,
            configuration_manager=self.config_manager,
            failover_manager=self.failover_manager,
            error_handler=self.network_error_handler,
            audit_logger=self.audit_logger,
        )

        # Clinical workflow integration
        if clinical_workflow is None:
            # Create default clinical workflow system
            from ..workflow import ClinicalWorkflowSystem

            self.clinical_workflow = ClinicalWorkflowSystem()
        else:
            self.clinical_workflow = clinical_workflow

        # Workflow orchestrator
        self.workflow_orchestrator = WorkflowOrchestrator(
            pacs_adapter=self.pacs_adapter,
            clinical_workflow=self.clinical_workflow,
            poll_interval=self.config.workflow.poll_interval,
            max_concurrent_studies=self.config.performance.max_concurrent_studies,
        )

        logger.info("PACS Integration Service initialized successfully")

    def start(self) -> None:
        """
        Start the PACS Integration Service.

        This starts all background services including:
        - Automated PACS polling
        - Failover health checks
        - Dead letter queue processing
        - Audit log management
        """
        if self._is_running:
            logger.warning("PACS Service is already running")
            return

        logger.info("Starting PACS Integration Service")

        try:
            # Validate configuration
            validation_result = self.config_manager.validate_configuration(self.config)
            if not validation_result.is_valid:
                raise ValueError(f"Configuration validation failed: {validation_result.errors}")

            # Start failover manager
            self.failover_manager.start_health_checks()

            # Start workflow orchestrator
            self.workflow_orchestrator.start_automated_polling()

            # Start dead letter queue processor
            self.dead_letter_queue.start_processing()

            # Start audit log management
            self.audit_logger.start_log_management()

            self._is_running = True

            logger.info("PACS Integration Service started successfully")

            # Log startup event
            self.audit_logger.log_system_event(
                event_type="service_startup",
                details={
                    "profile": self.profile,
                    "endpoints": len(self.config.pacs_endpoints),
                    "poll_interval": str(self.config.workflow.poll_interval),
                },
            )

        except Exception as e:
            logger.error(f"Failed to start PACS Service: {str(e)}")
            self.shutdown()
            raise

    def shutdown(self) -> None:
        """
        Shutdown the PACS Integration Service gracefully.

        This stops all background services and ensures clean shutdown.
        """
        if not self._is_running:
            return

        logger.info("Shutting down PACS Integration Service")

        try:
            # Stop workflow orchestrator
            self.workflow_orchestrator.stop_automated_polling()

            # Stop failover manager
            self.failover_manager.stop_health_checks()

            # Stop dead letter queue processor
            self.dead_letter_queue.stop_processing()

            # Stop audit log management
            self.audit_logger.stop_log_management()

            # Log shutdown event
            self.audit_logger.log_system_event(
                event_type="service_shutdown",
                details={"profile": self.profile},
            )

            self._is_running = False

            logger.info("PACS Integration Service shut down successfully")

        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of all components.

        Returns:
            Health check results with component status
        """
        health_status = {
            "service_running": self._is_running,
            "timestamp": self.audit_logger._get_timestamp(),
            "components": {},
        }

        # Check PACS endpoints
        endpoint_health = self.failover_manager.get_endpoint_health()
        health_status["components"]["pacs_endpoints"] = {
            "status": (
                "healthy" if any(e["is_healthy"] for e in endpoint_health.values()) else "unhealthy"
            ),
            "endpoints": endpoint_health,
        }

        # Check workflow orchestrator
        workflow_status = self.workflow_orchestrator.get_processing_status()
        health_status["components"]["workflow_orchestrator"] = {
            "status": "healthy" if workflow_status["is_running"] else "stopped",
            "active_processing": workflow_status["active_processing"],
            "queued_studies": workflow_status["queued_studies"],
        }

        # Check dead letter queue
        dlq_stats = self.dead_letter_queue.get_statistics()
        health_status["components"]["dead_letter_queue"] = {
            "status": "healthy" if dlq_stats["queue_size"] < 100 else "warning",
            "queue_size": dlq_stats["queue_size"],
        }

        # Check audit logger
        audit_stats = self.audit_logger.get_statistics()
        health_status["components"]["audit_logger"] = {
            "status": "healthy",
            "total_logs": audit_stats["total_logs"],
            "log_directory": str(self.config.audit.log_directory),
        }

        # Overall status
        component_statuses = [c["status"] for c in health_status["components"].values()]
        if all(s == "healthy" for s in component_statuses):
            health_status["overall_status"] = "healthy"
        elif any(s == "unhealthy" for s in component_statuses):
            health_status["overall_status"] = "unhealthy"
        else:
            health_status["overall_status"] = "degraded"

        return health_status

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from all components.

        Returns:
            Statistics dictionary
        """
        return {
            "service": {
                "is_running": self._is_running,
                "profile": self.profile,
                "config_path": str(self.config_path),
            },
            "workflow": self.workflow_orchestrator.get_processing_status(),
            "failover": {
                "endpoint_health": self.failover_manager.get_endpoint_health(),
            },
            "dead_letter_queue": self.dead_letter_queue.get_statistics(),
            "audit": self.audit_logger.get_statistics(),
            "notifications": self.notification_system.get_statistics(),
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
