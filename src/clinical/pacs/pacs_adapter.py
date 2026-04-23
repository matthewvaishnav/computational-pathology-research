"""
Main PACS Adapter for HistoCore Integration System.

This module implements the main PACSAdapter class that orchestrates all PACS
integration components and provides a unified interface for DICOM operations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .query_engine import QueryEngine
from .retrieval_engine import RetrievalEngine
from .storage_engine import StorageEngine
from .security_manager import SecurityManager
from .configuration_manager import ConfigurationManager
from .data_models import (
    PACSConfiguration, PACSEndpoint, StudyInfo, SeriesInfo,
    AnalysisResults, OperationResult, ValidationResult
)

logger = logging.getLogger(__name__)


class PACSAdapter:
    """
    Main PACS Adapter that orchestrates all PACS integration components.
    
    This class provides a unified interface for:
    - DICOM query operations (C-FIND)
    - WSI retrieval operations (C-MOVE)
    - AI results storage (C-STORE)
    - Security management and authentication
    - Configuration management across environments
    - Integration with existing HistoCore DICOM adapter
    """
    
    def __init__(
        self,
        config_profile: str = "development",
        config_directory: Union[str, Path] = "configs/pacs",
        ae_title: str = "HISTOCORE"
    ):
        """
        Initialize PACS Adapter.
        
        Args:
            config_profile: Configuration profile to load
            config_directory: Directory containing configuration files
            ae_title: Base Application Entity title
        """
        self.config_profile = config_profile
        self.ae_title = ae_title
        
        logger.info(f"Initializing PACSAdapter with profile: {config_profile}")
        
        # Initialize configuration manager
        self.config_manager = ConfigurationManager(config_directory)
        
        # Load configuration
        try:
            self.configuration = self.config_manager.load_configuration(config_profile)
        except Exception as e:
            logger.warning(f"Failed to load configuration {config_profile}: {e}")
            logger.info("Creating default configuration")
            self.configuration = self.config_manager.create_default_configuration()
        
        # Initialize security manager
        self.security_manager = SecurityManager()
        
        # Initialize DICOM engines
        self.query_engine = QueryEngine(ae_title=f"{ae_title}_Q")
        self.retrieval_engine = RetrievalEngine(ae_title=f"{ae_title}_R")
        self.storage_engine = StorageEngine(ae_title=f"{ae_title}_S")
        
        # Track active operations
        self._active_operations: Dict[str, Dict[str, Any]] = {}
        
        logger.info("PACSAdapter initialized successfully")
    
    def query_studies(
        self,
        patient_id: Optional[str] = None,
        study_date_range: Optional[Tuple[datetime, datetime]] = None,
        modality: str = "SM",
        max_results: int = 1000,
        endpoint_id: Optional[str] = None
    ) -> Tuple[List[StudyInfo], OperationResult]:
        """
        Query PACS for WSI studies.
        
        Args:
            patient_id: Patient ID to search for
            study_date_range: Date range for study filtering
            modality: DICOM modality (default: SM for Slide Microscopy)
            max_results: Maximum number of results
            endpoint_id: Specific endpoint to query (uses primary if None)
            
        Returns:
            Tuple of (study list, operation result)
        """
        logger.info(f"Querying studies: patient_id={patient_id}, modality={modality}")
        
        operation_id = f"query_studies_{int(datetime.now().timestamp())}"
        
        try:
            # Select endpoint
            endpoint = self._select_endpoint(endpoint_id)
            if not endpoint:
                return [], OperationResult.error_result(
                    operation_id=operation_id,
                    message="No available PACS endpoint",
                    errors=["No endpoint configured"]
                )
            
            # Execute query
            studies = self.query_engine.query_studies(
                endpoint=endpoint,
                patient_id=patient_id,
                study_date_range=study_date_range,
                modality=modality,
                max_results=max_results
            )
            
            result = OperationResult.success_result(
                operation_id=operation_id,
                message=f"Found {len(studies)} studies",
                data={"study_count": len(studies), "endpoint": endpoint.endpoint_id}
            )
            
            return studies, result
            
        except Exception as e:
            logger.error(f"Query operation failed: {str(e)}")
            error_result = OperationResult.error_result(
                operation_id=operation_id,
                message=f"Query failed: {str(e)}",
                errors=[str(e)]
            )
            return [], error_result
    
    def retrieve_study(
        self,
        study_instance_uid: str,
        destination_path: Union[str, Path],
        endpoint_id: Optional[str] = None
    ) -> OperationResult:
        """
        Retrieve complete study from PACS.
        
        Args:
            study_instance_uid: Study Instance UID to retrieve
            destination_path: Local directory to store files
            endpoint_id: Specific endpoint to use (uses primary if None)
            
        Returns:
            OperationResult with retrieval status
        """
        logger.info(f"Retrieving study: {study_instance_uid}")
        
        try:
            # Select endpoint
            endpoint = self._select_endpoint(endpoint_id)
            if not endpoint:
                return OperationResult.error_result(
                    operation_id=f"retrieve_{study_instance_uid}",
                    message="No available PACS endpoint",
                    errors=["No endpoint configured"]
                )
            
            # Execute retrieval
            result = self.retrieval_engine.retrieve_study(
                endpoint=endpoint,
                study_instance_uid=study_instance_uid,
                destination_path=Path(destination_path)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Retrieval operation failed: {str(e)}")
            return OperationResult.error_result(
                operation_id=f"retrieve_{study_instance_uid}",
                message=f"Retrieval failed: {str(e)}",
                errors=[str(e)]
            )
    
    def store_analysis_results(
        self,
        analysis_results: AnalysisResults,
        original_study_uid: str,
        endpoint_id: Optional[str] = None
    ) -> OperationResult:
        """
        Store AI analysis results as DICOM Structured Report.
        
        Args:
            analysis_results: AI analysis results to store
            original_study_uid: Study UID of original WSI
            endpoint_id: Specific endpoint to use (uses primary if None)
            
        Returns:
            OperationResult with storage status
        """
        logger.info(f"Storing analysis results for study: {original_study_uid}")
        
        try:
            # Select endpoint
            endpoint = self._select_endpoint(endpoint_id)
            if not endpoint:
                return OperationResult.error_result(
                    operation_id=f"store_{original_study_uid}",
                    message="No available PACS endpoint",
                    errors=["No endpoint configured"]
                )
            
            # Execute storage
            result = self.storage_engine.store_analysis_results(
                endpoint=endpoint,
                analysis_results=analysis_results,
                original_study_uid=original_study_uid
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Storage operation failed: {str(e)}")
            return OperationResult.error_result(
                operation_id=f"store_{original_study_uid}",
                message=f"Storage failed: {str(e)}",
                errors=[str(e)]
            )
    
    def test_connection(self, endpoint_id: Optional[str] = None) -> OperationResult:
        """
        Test connection to PACS endpoint.
        
        Args:
            endpoint_id: Specific endpoint to test (tests primary if None)
            
        Returns:
            OperationResult with connection test status
        """
        logger.info(f"Testing PACS connection: {endpoint_id or 'primary'}")
        
        operation_id = f"test_connection_{endpoint_id or 'primary'}"
        
        try:
            # Select endpoint
            endpoint = self._select_endpoint(endpoint_id)
            if not endpoint:
                return OperationResult.error_result(
                    operation_id=operation_id,
                    message="No available PACS endpoint",
                    errors=["No endpoint configured"]
                )
            
            # Test secure connection if TLS enabled
            if endpoint.security_config.tls_enabled:
                try:
                    connection = self.security_manager.establish_secure_connection(endpoint)
                    connection.close()
                    
                    return OperationResult.success_result(
                        operation_id=operation_id,
                        message=f"Secure connection test successful: {endpoint.host}:{endpoint.port}",
                        data={"endpoint": endpoint.endpoint_id, "tls_enabled": True}
                    )
                    
                except Exception as e:
                    return OperationResult.error_result(
                        operation_id=operation_id,
                        message=f"Secure connection test failed: {str(e)}",
                        errors=[str(e)]
                    )
            else:
                # Test basic DICOM connection with echo
                try:
                    # Simple connection test (would use DICOM C-ECHO in full implementation)
                    import socket
                    sock = socket.create_connection(
                        (endpoint.host, endpoint.port),
                        timeout=10
                    )
                    sock.close()
                    
                    return OperationResult.success_result(
                        operation_id=operation_id,
                        message=f"Connection test successful: {endpoint.host}:{endpoint.port}",
                        data={"endpoint": endpoint.endpoint_id, "tls_enabled": False}
                    )
                    
                except Exception as e:
                    return OperationResult.error_result(
                        operation_id=operation_id,
                        message=f"Connection test failed: {str(e)}",
                        errors=[str(e)]
                    )
                    
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return OperationResult.error_result(
                operation_id=operation_id,
                message=f"Connection test error: {str(e)}",
                errors=[str(e)]
            )
    
    def get_endpoint_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all configured PACS endpoints.
        
        Returns:
            Dictionary mapping endpoint IDs to status information
        """
        status = {}
        
        for endpoint_id, endpoint in self.configuration.pacs_endpoints.items():
            try:
                # Test connection
                test_result = self.test_connection(endpoint_id)
                
                status[endpoint_id] = {
                    "endpoint_id": endpoint_id,
                    "host": endpoint.host,
                    "port": endpoint.port,
                    "vendor": endpoint.vendor.value,
                    "is_primary": endpoint.is_primary,
                    "tls_enabled": endpoint.security_config.tls_enabled,
                    "connection_status": "online" if test_result.success else "offline",
                    "last_tested": datetime.now().isoformat(),
                    "error": test_result.message if not test_result.success else None
                }
                
            except Exception as e:
                status[endpoint_id] = {
                    "endpoint_id": endpoint_id,
                    "host": endpoint.host,
                    "port": endpoint.port,
                    "connection_status": "error",
                    "error": str(e),
                    "last_tested": datetime.now().isoformat()
                }
        
        return status
    
    def reload_configuration(self, new_profile: Optional[str] = None) -> OperationResult:
        """
        Reload PACS configuration.
        
        Args:
            new_profile: Optional new profile to load (keeps current if None)
            
        Returns:
            OperationResult with reload status
        """
        profile = new_profile or self.config_profile
        logger.info(f"Reloading configuration: {profile}")
        
        operation_id = f"reload_config_{profile}"
        
        try:
            # Load new configuration
            new_config = self.config_manager.load_configuration(profile)
            
            # Validate configuration
            validation = self.config_manager.validate_configuration(new_config)
            if not validation.is_valid:
                return OperationResult.error_result(
                    operation_id=operation_id,
                    message="Invalid configuration",
                    errors=validation.errors
                )
            
            # Close existing connections
            self.security_manager.close_all_connections()
            
            # Update configuration
            self.configuration = new_config
            self.config_profile = profile
            
            return OperationResult.success_result(
                operation_id=operation_id,
                message=f"Configuration reloaded successfully: {profile}",
                data={"profile": profile, "endpoints": len(new_config.pacs_endpoints)}
            )
            
        except Exception as e:
            logger.error(f"Configuration reload failed: {str(e)}")
            return OperationResult.error_result(
                operation_id=operation_id,
                message=f"Reload failed: {str(e)}",
                errors=[str(e)]
            )
    
    def _select_endpoint(self, endpoint_id: Optional[str] = None) -> Optional[PACSEndpoint]:
        """Select PACS endpoint for operation."""
        if endpoint_id:
            return self.configuration.pacs_endpoints.get(endpoint_id)
        else:
            # Return primary endpoint
            primary = self.configuration.get_primary_endpoint()
            if primary:
                return primary
            
            # Fallback to first available endpoint
            if self.configuration.pacs_endpoints:
                return next(iter(self.configuration.pacs_endpoints.values()))
            
            return None
    
    def get_adapter_statistics(self) -> Dict[str, Any]:
        """Get comprehensive PACS adapter statistics."""
        return {
            "config_profile": self.config_profile,
            "ae_title": self.ae_title,
            "endpoints_configured": len(self.configuration.pacs_endpoints),
            "primary_endpoint": self.configuration.get_primary_endpoint().endpoint_id if self.configuration.get_primary_endpoint() else None,
            "active_operations": len(self._active_operations),
            "query_engine": self.query_engine.get_query_statistics(),
            "retrieval_engine": self.retrieval_engine.get_retrieval_statistics(),
            "storage_engine": self.storage_engine.get_storage_statistics(),
            "security_manager": self.security_manager.get_security_statistics(),
            "config_manager": self.config_manager.get_configuration_statistics()
        }
    
    def shutdown(self):
        """Shutdown PACS adapter and cleanup resources."""
        logger.info("Shutting down PACS adapter")
        
        try:
            # Close all secure connections
            self.security_manager.close_all_connections()
            
            # Clear active operations
            self._active_operations.clear()
            
            logger.info("PACS adapter shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()