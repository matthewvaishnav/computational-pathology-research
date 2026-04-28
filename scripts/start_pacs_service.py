#!/usr/bin/env python3
"""
PACS Integration Service Startup Script

Starts the complete PACS integration service with proper configuration
and error handling for production deployment.
"""

import os
import sys
import asyncio
import logging
import signal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pacs.pacs_service import PACSIntegrationService


class PACSServiceRunner:
    """PACS service runner with proper lifecycle management."""
    
    def __init__(self, config_path: str = None):
        """Initialize service runner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or str(project_root / "config" / "pacs_config.yaml")
        self.service: PACSIntegrationService = None
        self.shutdown_event = asyncio.Event()
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("PACS service runner initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir / "pacs_service.log")
            ]
        )
        
        # Set specific logger levels
        logging.getLogger('pynetdicom').setLevel(logging.WARNING)
        logging.getLogger('pydicom').setLevel(logging.WARNING)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Run the PACS integration service."""
        try:
            logger.info("Starting PACS integration service...")
            
            # Create and start service
            self.service = PACSIntegrationService(config_path=self.config_path)
            await self.service.start()
            
            # Print service status
            self._print_service_status()
            
            # Wait for shutdown signal
            logger.info("PACS service running. Press Ctrl+C to stop.")
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Failed to start PACS service: {e}")
            raise
        finally:
            await self._shutdown()
    
    async def _shutdown(self):
        """Shutdown the service gracefully."""
        if self.service:
            logger.info("Shutting down PACS integration service...")
            try:
                await self.service.stop()
                logger.info("PACS service shutdown complete")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
    
    def _print_service_status(self):
        """Print service status information."""
        if not self.service:
            return
        
        status = self.service.get_service_status()
        
        print("\n" + "="*60)
        print("PACS INTEGRATION SERVICE STATUS")
        print("="*60)
        print(f"Service Running: {status['is_running']}")
        
        # DICOM Server Status
        dicom_status = status['components']['dicom_server']
        if dicom_status:
            print(f"\nDICOM Server:")
            print(f"  AE Title: {dicom_status['ae_title']}")
            print(f"  Port: {dicom_status['port']}")
            print(f"  Running: {dicom_status['is_running']}")
            print(f"  Stored Studies: {dicom_status['stored_studies']}")
        
        # PACS Client Status
        pacs_status = status['components']['pacs_client']
        print(f"\nPACS Client:")
        print(f"  AE Title: {pacs_status['ae_title']}")
        print(f"  Connections: {pacs_status['connections']}")
        
        # Test PACS connections
        connection_results = self.service.test_pacs_connections()
        if connection_results:
            print(f"\nPACS Connection Tests:")
            for pacs_name, result in connection_results.items():
                status_str = "✓ Connected" if result else "✗ Failed"
                print(f"  {pacs_name}: {status_str}")
        
        # Worklist Status
        worklist_status = status['components']['worklist_manager']
        if worklist_status:
            print(f"\nWorklist Manager:")
            print(f"  Total Entries: {worklist_status['total_entries']}")
            print(f"  Scheduled Today: {worklist_status['scheduled_for_today']}")
        
        # Workflow Status
        workflow_status = status['components']['workflow_orchestrator']
        if workflow_status:
            print(f"\nWorkflow Orchestrator:")
            print(f"  Running: {workflow_status['is_running']}")
            print(f"  Active Tasks: {workflow_status['active_tasks']}")
            print(f"  Completed Tasks: {workflow_status['completed_tasks']}")
        
        # HL7 Server Status
        hl7_status = status['components']['hl7_server']
        print(f"\nHL7 Server:")
        print(f"  Running: {hl7_status['running']}")
        print(f"  Address: {hl7_status['host']}:{hl7_status['port']}")
        
        print("\n" + "="*60)
        print("Service endpoints:")
        print(f"  DICOM Server: {dicom_status['ae_title']}@localhost:{dicom_status['port']}")
        print(f"  HL7 Server: {hl7_status['host']}:{hl7_status['port']}")
        print("="*60 + "\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Start PACS Integration Service')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run service
    runner = PACSServiceRunner(config_path=args.config)
    
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()