#!/usr/bin/env python3
"""
PACS Integration Test Script

Comprehensive testing script for PACS integration components.
Tests DICOM networking, worklist management, workflow orchestration, and HL7 integration.
"""

import os
import sys
import asyncio
import logging
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pacs.dicom_server import DicomServer, DicomStorageProvider
from src.pacs.pacs_client import PACSClient, PACSConnection
from src.pacs.worklist_manager import WorklistManager, create_sample_pathology_worklist
from src.pacs.clinical_workflow import ClinicalWorkflowOrchestrator
from src.pacs.hl7_integration import HL7Server, HL7MessageHandler, HL7Message
from src.pacs.pacs_service import PACSIntegrationService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PACSIntegrationTester:
    """PACS integration test suite."""
    
    def __init__(self):
        """Initialize tester."""
        self.test_results = {}
        self.temp_dir = None
        
    async def run_all_tests(self):
        """Run all PACS integration tests."""
        logger.info("Starting PACS integration tests...")
        
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp(prefix="pacs_test_")
        logger.info(f"Using temporary directory: {self.temp_dir}")
        
        try:
            # Run individual component tests
            await self.test_dicom_server()
            await self.test_pacs_client()
            await self.test_worklist_manager()
            await self.test_hl7_integration()
            await self.test_workflow_orchestration()
            await self.test_full_service_integration()
            
            # Print results
            self.print_test_results()
            
        finally:
            # Cleanup
            if self.temp_dir and Path(self.temp_dir).exists():
                import shutil
                shutil.rmtree(self.temp_dir)
    
    async def test_dicom_server(self):
        """Test DICOM server functionality."""
        logger.info("Testing DICOM server...")
        
        try:
            # Create DICOM server
            storage_provider = DicomStorageProvider(storage_dir=self.temp_dir)
            server = DicomServer(
                ae_title="TEST_SERVER",
                port=11113,  # Use different port for testing
                storage_provider=storage_provider
            )
            
            # Test server startup
            server.start(blocking=False)
            await asyncio.sleep(1)  # Wait for startup
            
            # Test server status
            status = server.get_status()
            assert status['is_running'] == True
            assert status['ae_title'] == "TEST_SERVER"
            assert status['port'] == 11113
            
            # Test connection test (should fail since no remote PACS)
            connection_result = server.test_connection("NONEXISTENT", "localhost", 11114)
            assert connection_result == False  # Expected to fail
            
            # Stop server
            server.stop()
            
            self.test_results['dicom_server'] = True
            logger.info("✓ DICOM server test passed")
            
        except Exception as e:
            self.test_results['dicom_server'] = False
            logger.error(f"✗ DICOM server test failed: {e}")
    
    async def test_pacs_client(self):
        """Test PACS client functionality."""
        logger.info("Testing PACS client...")
        
        try:
            # Create PACS client
            client = PACSClient(ae_title="TEST_CLIENT")
            
            # Add test connection
            test_connection = PACSConnection(
                name="test_pacs",
                ae_title="TEST_PACS",
                host="localhost",
                port=11114,
                vendor="test"
            )
            client.add_pacs_connection(test_connection)
            
            # Test connection (should fail since no server)
            connection_result = client.test_connection("test_pacs")
            assert connection_result == False  # Expected to fail
            
            # Test find studies (should return empty list)
            studies = client.find_studies("test_pacs", patient_id="TEST001")
            assert isinstance(studies, list)
            assert len(studies) == 0
            
            # Test connection status
            status = client.get_connection_status()
            assert "test_pacs" in status
            assert status["test_pacs"]["is_connected"] == False
            
            self.test_results['pacs_client'] = True
            logger.info("✓ PACS client test passed")
            
        except Exception as e:
            self.test_results['pacs_client'] = False
            logger.error(f"✗ PACS client test failed: {e}")
    
    async def test_worklist_manager(self):
        """Test worklist manager functionality."""
        logger.info("Testing worklist manager...")
        
        try:
            # Create worklist manager
            manager = create_sample_pathology_worklist()
            
            # Test worklist statistics
            stats = manager.get_worklist_statistics()
            assert stats['total_entries'] > 0
            assert 'status_distribution' in stats
            assert 'modality_distribution' in stats
            
            # Test creating new entry
            entry = manager.create_pathology_worklist_entry(
                patient_id="TEST001",
                patient_name="Test^Patient",
                accession_number="ACC_TEST001",
                study_description="Test Study"
            )
            
            assert entry.patient_id == "TEST001"
            assert entry.accession_number == "ACC_TEST001"
            
            # Test querying entries
            entries = manager.query_worklist(modality="SM")
            assert len(entries) > 0
            
            # Test getting scheduled studies for AI
            ai_studies = manager.get_scheduled_studies_for_ai()
            assert isinstance(ai_studies, list)
            
            self.test_results['worklist_manager'] = True
            logger.info("✓ Worklist manager test passed")
            
        except Exception as e:
            self.test_results['worklist_manager'] = False
            logger.error(f"✗ Worklist manager test failed: {e}")
    
    async def test_hl7_integration(self):
        """Test HL7 integration functionality."""
        logger.info("Testing HL7 integration...")
        
        try:
            # Create HL7 message handler
            handler = HL7MessageHandler()
            
            # Test message parsing
            sample_hl7_message = (
                "MSH|^~\\&|SENDING_APP|SENDING_FACILITY|RECEIVING_APP|RECEIVING_FACILITY|20260427120000||ORM^O01|12345|P|2.5\r"
                "PID|1||TEST001^^^MRN||Test^Patient^M||19800101|M|||123 Main St^^City^ST^12345\r"
                "OBR|1||ACC001|PATH^Pathology Analysis|||20260427120000\r"
            )
            
            # Process message
            ack_response = handler.process_message(sample_hl7_message)
            assert "MSA|AA|12345" in ack_response  # Should contain acceptance ACK
            
            # Test HL7 message parsing
            message = HL7Message.parse(sample_hl7_message)
            assert message.message_type == "ORM^O01"
            assert message.control_id == "12345"
            
            # Test patient info extraction
            patient_info = message.extract_patient_info()
            assert patient_info['patient_id'] == "TEST001"
            assert "Test^Patient" in patient_info['patient_name']
            
            # Test order info extraction
            order_info = message.extract_order_info()
            assert order_info['accession_number'] == "ACC001"
            
            self.test_results['hl7_integration'] = True
            logger.info("✓ HL7 integration test passed")
            
        except Exception as e:
            self.test_results['hl7_integration'] = False
            logger.error(f"✗ HL7 integration test failed: {e}")
    
    async def test_workflow_orchestration(self):
        """Test workflow orchestration functionality."""
        logger.info("Testing workflow orchestration...")
        
        try:
            # Create components
            storage_provider = DicomStorageProvider(storage_dir=self.temp_dir)
            dicom_server = DicomServer(
                ae_title="TEST_WORKFLOW_SERVER",
                port=11115,
                storage_provider=storage_provider
            )
            
            pacs_client = PACSClient(ae_title="TEST_WORKFLOW_CLIENT")
            worklist_manager = create_sample_pathology_worklist()
            
            # Create workflow orchestrator
            orchestrator = ClinicalWorkflowOrchestrator(
                pacs_client=pacs_client,
                dicom_server=dicom_server,
                worklist_manager=worklist_manager
            )
            
            # Test workflow status
            status = orchestrator.get_workflow_status()
            assert 'is_running' in status
            assert 'active_tasks' in status
            assert 'completed_tasks' in status
            
            # Test callback registration
            async def test_analysis_callback(study_path, study_uid):
                logger.info(f"Test analysis callback: {study_uid}")
            
            async def test_notification_callback(task, results):
                logger.info(f"Test notification callback: {task.accession_number}")
            
            orchestrator.add_analysis_callback(test_analysis_callback)
            orchestrator.add_notification_callback(test_notification_callback)
            
            self.test_results['workflow_orchestration'] = True
            logger.info("✓ Workflow orchestration test passed")
            
        except Exception as e:
            self.test_results['workflow_orchestration'] = False
            logger.error(f"✗ Workflow orchestration test failed: {e}")
    
    async def test_full_service_integration(self):
        """Test full PACS service integration."""
        logger.info("Testing full service integration...")
        
        try:
            # Create test configuration
            test_config = {
                'dicom_server': {
                    'ae_title': 'TEST_MEDICAL_AI',
                    'port': 11116,
                    'storage_dir': self.temp_dir
                },
                'pacs_client': {
                    'ae_title': 'TEST_MEDICAL_AI_CLIENT'
                },
                'hl7_server': {
                    'host': 'localhost',
                    'port': 2576,
                    'enabled': True
                },
                'workflow': {
                    'polling_interval': 5,  # Short interval for testing
                    'max_concurrent_tasks': 2,
                    'auto_start': False  # Don't auto-start for testing
                },
                'pacs_connections': {},
                'notifications': {
                    'email_enabled': False,
                    'sms_enabled': False,
                    'hl7_enabled': False
                }
            }
            
            # Create service with test config
            service = PACSIntegrationService()
            service.config = test_config
            
            # Start service
            await service.start()
            
            # Test service status
            status = service.get_service_status()
            assert status['is_running'] == True
            assert status['components']['dicom_server'] is not None
            assert status['components']['pacs_client']['ae_title'] == 'TEST_MEDICAL_AI_CLIENT'
            
            # Test PACS connection testing (should return empty dict)
            connection_results = service.test_pacs_connections()
            assert isinstance(connection_results, dict)
            
            # Stop service
            await service.stop()
            
            self.test_results['full_service_integration'] = True
            logger.info("✓ Full service integration test passed")
            
        except Exception as e:
            self.test_results['full_service_integration'] = False
            logger.error(f"✗ Full service integration test failed: {e}")
    
    def print_test_results(self):
        """Print test results summary."""
        print("\n" + "="*60)
        print("PACS INTEGRATION TEST RESULTS")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        for test_name, result in self.test_results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print("-" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print("="*60)
        
        if failed_tests == 0:
            print("🎉 All PACS integration tests passed!")
        else:
            print(f"⚠️  {failed_tests} test(s) failed. Check logs for details.")
        
        print()


async def main():
    """Main test function."""
    tester = PACSIntegrationTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())