#!/usr/bin/env python3
"""
PACS Integration Service

Main service class that orchestrates all PACS components for hospital integration.
Provides a unified interface for DICOM networking, clinical workflow, and HL7 integration.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

from .dicom_server import DicomServer, DicomStorageProvider, create_medical_ai_dicom_server
from .pacs_client import PACSClient, PACSConnection, create_hospital_pacs_connections
from .worklist_manager import WorklistManager, create_sample_pathology_worklist
from .clinical_workflow import ClinicalWorkflowOrchestrator, create_sample_ai_analysis_callback, create_sample_notification_callback
from .hl7_integration import HL7Server, setup_hl7_integration

logger = logging.getLogger(__name__)


class PACSIntegrationService:
    """Main PACS integration service."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize PACS integration service.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.dicom_server: Optional[DicomServer] = None
        self.pacs_client: Optional[PACSClient] = None
        self.worklist_manager: Optional[WorklistManager] = None
        self.workflow_orchestrator: Optional[ClinicalWorkflowOrchestrator] = None
        self.hl7_server: Optional[HL7Server] = None
        
        # Service state
        self.is_running = False
        self.workflow_task: Optional[asyncio.Task] = None
        
        logger.info("PACS integration service initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'dicom_server': {
                'ae_title': 'MEDICAL_AI',
                'port': 11112,
                'storage_dir': '/tmp/dicom_storage'
            },
            'pacs_client': {
                'ae_title': 'MEDICAL_AI_CLIENT'
            },
            'hl7_server': {
                'host': 'localhost',
                'port': 2575,
                'enabled': True
            },
            'workflow': {
                'polling_interval': 60,
                'max_concurrent_tasks': 10,
                'auto_start': True
            },
            'pacs_connections': {},
            'notifications': {
                'email_enabled': False,
                'sms_enabled': False,
                'hl7_enabled': True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(file_config)
                    logger.info(f"Loaded configuration from: {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
        
        return default_config
    
    async def start(self):
        """Start PACS integration service."""
        if self.is_running:
            logger.warning("PACS integration service already running")
            return
        
        try:
            logger.info("Starting PACS integration service...")
            
            # 1. Initialize DICOM server
            await self._start_dicom_server()
            
            # 2. Initialize PACS client
            self._initialize_pacs_client()
            
            # 3. Initialize worklist manager
            self._initialize_worklist_manager()
            
            # 4. Initialize workflow orchestrator
            self._initialize_workflow_orchestrator()
            
            # 5. Start HL7 server (if enabled)
            if self.config['hl7_server']['enabled']:
                self._start_hl7_server()
            
            # 6. Start workflow orchestration (if enabled)
            if self.config['workflow']['auto_start']:
                await self._start_workflow_orchestration()
            
            self.is_running = True
            logger.info("PACS integration service started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start PACS integration service: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop PACS integration service."""
        if not self.is_running:
            return
        
        logger.info("Stopping PACS integration service...")
        
        try:
            # Stop workflow orchestration
            if self.workflow_task and not self.workflow_task.done():
                self.workflow_task.cancel()
                try:
                    await self.workflow_task
                except asyncio.CancelledError:
                    pass
            
            if self.workflow_orchestrator:
                self.workflow_orchestrator.stop_workflow_orchestration()
            
            # Stop HL7 server
            if self.hl7_server:
                self.hl7_server.stop()
            
            # Stop DICOM server
            if self.dicom_server:
                self.dicom_server.stop()
            
            self.is_running = False
            logger.info("PACS integration service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping PACS integration service: {e}")
    
    async def _start_dicom_server(self):
        """Start DICOM server."""
        config = self.config['dicom_server']
        
        self.dicom_server = create_medical_ai_dicom_server(
            port=config['port'],
            storage_dir=config['storage_dir']
        )
        
        # Start server in background
        self.dicom_server.start(blocking=False)
        
        # Wait a moment for server to start
        await asyncio.sleep(1)
        
        logger.info(f"DICOM server started on port {config['port']}")
    
    def _initialize_pacs_client(self):
        """Initialize PACS client."""
        config = self.config['pacs_client']
        
        self.pacs_client = PACSClient(ae_title=config['ae_title'])
        
        # Add configured PACS connections
        pacs_connections = self.config.get('pacs_connections', {})
        if not pacs_connections:
            # Use default hospital connections for demo
            pacs_connections = create_hospital_pacs_connections()
        
        for connection in pacs_connections.values():
            if isinstance(connection, dict):
                # Convert dict to PACSConnection object
                connection = PACSConnection(**connection)
            self.pacs_client.add_pacs_connection(connection)
        
        logger.info(f"PACS client initialized with {len(pacs_connections)} connections")
    
    def _initialize_worklist_manager(self):
        """Initialize worklist manager."""
        self.worklist_manager = WorklistManager()
        
        # Add sample worklist entries for demo
        sample_manager = create_sample_pathology_worklist()
        self.worklist_manager.worklist_entries.update(sample_manager.worklist_entries)
        
        logger.info("Worklist manager initialized")
    
    def _initialize_workflow_orchestrator(self):
        """Initialize workflow orchestrator."""
        if not all([self.pacs_client, self.dicom_server, self.worklist_manager]):
            raise RuntimeError("Required components not initialized")
        
        self.workflow_orchestrator = ClinicalWorkflowOrchestrator(
            pacs_client=self.pacs_client,
            dicom_server=self.dicom_server,
            worklist_manager=self.worklist_manager
        )
        
        # Configure workflow settings
        workflow_config = self.config['workflow']
        self.workflow_orchestrator.polling_interval = workflow_config['polling_interval']
        self.workflow_orchestrator.max_concurrent_tasks = workflow_config['max_concurrent_tasks']
        
        # Add sample callbacks
        self.workflow_orchestrator.add_analysis_callback(create_sample_ai_analysis_callback)
        self.workflow_orchestrator.add_notification_callback(create_sample_notification_callback)
        
        logger.info("Workflow orchestrator initialized")
    
    def _start_hl7_server(self):
        """Start HL7 server."""
        if not self.worklist_manager:
            logger.warning("Cannot start HL7 server: worklist manager not initialized")
            return
        
        hl7_config = self.config['hl7_server']
        
        self.hl7_server = setup_hl7_integration(
            worklist_manager=self.worklist_manager,
            host=hl7_config['host'],
            port=hl7_config['port']
        )
        
        self.hl7_server.start()
        logger.info(f"HL7 server started on {hl7_config['host']}:{hl7_config['port']}")
    
    async def _start_workflow_orchestration(self):
        """Start workflow orchestration."""
        if not self.workflow_orchestrator:
            logger.warning("Cannot start workflow orchestration: orchestrator not initialized")
            return
        
        self.workflow_task = asyncio.create_task(
            self.workflow_orchestrator.start_workflow_orchestration()
        )
        
        logger.info("Workflow orchestration started")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status information.
        
        Returns:
            Dictionary with service status
        """
        status = {
            'is_running': self.is_running,
            'components': {
                'dicom_server': self.dicom_server.get_status() if self.dicom_server else None,
                'pacs_client': {
                    'ae_title': self.pacs_client.ae_title if self.pacs_client else None,
                    'connections': len(self.pacs_client.connections) if self.pacs_client else 0
                },
                'worklist_manager': self.worklist_manager.get_worklist_statistics() if self.worklist_manager else None,
                'workflow_orchestrator': self.workflow_orchestrator.get_workflow_status() if self.workflow_orchestrator else None,
                'hl7_server': {
                    'running': self.hl7_server.is_running if self.hl7_server else False,
                    'host': self.config['hl7_server']['host'],
                    'port': self.config['hl7_server']['port']
                }
            }
        }
        
        return status
    
    def test_pacs_connections(self) -> Dict[str, bool]:
        """Test all PACS connections.
        
        Returns:
            Dictionary with connection test results
        """
        if not self.pacs_client:
            return {}
        
        results = {}
        for pacs_name in self.pacs_client.connections.keys():
            results[pacs_name] = self.pacs_client.test_connection(pacs_name)
        
        return results
    
    async def process_emergency_study(self, accession_number: str, pacs_name: str) -> bool:
        """Process emergency study with high priority.
        
        Args:
            accession_number: Accession number of emergency study
            pacs_name: PACS system name
            
        Returns:
            True if processing initiated successfully
        """
        if not self.workflow_orchestrator:
            logger.error("Workflow orchestrator not available")
            return False
        
        try:
            # Find study on specified PACS
            studies = self.pacs_client.find_studies(
                pacs_name=pacs_name,
                accession_number=accession_number
            )
            
            if not studies:
                logger.error(f"Emergency study not found: {accession_number}")
                return False
            
            study = studies[0]
            
            # Create high-priority worklist entry
            entry = self.worklist_manager.create_pathology_worklist_entry(
                patient_id=study.patient_id,
                patient_name=study.patient_name,
                accession_number=accession_number,
                study_description="EMERGENCY: " + study.study_description,
                priority="URGENT"
            )
            
            logger.info(f"Emergency study processing initiated: {accession_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process emergency study: {e}")
            return False
    
    def get_workflow_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status dictionary or None if not found
        """
        if not self.workflow_orchestrator:
            return None
        
        return self.workflow_orchestrator.get_task_details(task_id)
    
    def retry_failed_workflow_task(self, task_id: str) -> bool:
        """Retry failed workflow task.
        
        Args:
            task_id: Task ID to retry
            
        Returns:
            True if retry initiated successfully
        """
        if not self.workflow_orchestrator:
            return False
        
        return self.workflow_orchestrator.retry_failed_task(task_id)
    
    def cancel_workflow_task(self, task_id: str) -> bool:
        """Cancel workflow task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        if not self.workflow_orchestrator:
            return False
        
        return self.workflow_orchestrator.cancel_task(task_id)


async def main():
    """Main function for running PACS integration service."""
    import argparse
    
    parser = argparse.ArgumentParser(description='PACS Integration Service')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start service
    service = PACSIntegrationService(config_path=args.config)
    
    try:
        await service.start()
        
        # Keep service running
        logger.info("PACS integration service running. Press Ctrl+C to stop.")
        while service.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Service error: {e}")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())