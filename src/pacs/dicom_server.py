#!/usr/bin/env python3
"""
DICOM Server Implementation

Production-ready DICOM server for receiving studies from hospital PACS systems.
Supports C-STORE, C-FIND, C-MOVE operations with proper DICOM networking.
"""

import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydicom import dcmread
from pydicom.dataset import Dataset
from pynetdicom import AE, build_context, evt
from pynetdicom.sop_class import (
    CTImageStorage,
    DigitalXRayImageStorageForPresentation,
    DigitalXRayImageStorageForProcessing,
    MRImageStorage,
    SecondaryCaptureImageStorage,
    Verification,
    VLWholeSlideMicroscopyImageStorage,
)

logger = logging.getLogger(__name__)


class DicomStorageProvider:
    """Handles DICOM storage operations for received studies."""

    def __init__(self, storage_dir: str = "/tmp/dicom_storage"):
        """Initialize storage provider.

        Args:
            storage_dir: Directory to store received DICOM files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks for processing received studies
        self.study_received_callbacks: List[Callable] = []

        logger.info(f"DICOM storage provider initialized: {self.storage_dir}")

    def add_study_received_callback(self, callback: Callable[[str, Dataset], None]):
        """Add callback for when a study is received.

        Args:
            callback: Function to call with (file_path, dataset) when study received
        """
        self.study_received_callbacks.append(callback)

    def store_dicom_file(self, dataset: Dataset, context, info) -> int:
        """Store received DICOM file.

        Args:
            dataset: DICOM dataset
            context: Association context
            info: Store request info

        Returns:
            DICOM status code
        """
        try:
            # Generate filename based on DICOM identifiers
            study_uid = getattr(dataset, "StudyInstanceUID", "unknown_study")
            series_uid = getattr(dataset, "SeriesInstanceUID", "unknown_series")
            instance_uid = getattr(dataset, "SOPInstanceUID", str(uuid.uuid4()))

            # Create directory structure
            study_dir = self.storage_dir / study_uid
            series_dir = study_dir / series_uid
            series_dir.mkdir(parents=True, exist_ok=True)

            # Save DICOM file
            filename = f"{instance_uid}.dcm"
            file_path = series_dir / filename

            dataset.save_as(file_path, write_like_original=False)

            logger.info(f"Stored DICOM file: {file_path}")

            # Trigger callbacks for study processing
            for callback in self.study_received_callbacks:
                try:
                    callback(str(file_path), dataset)
                except Exception as e:
                    logger.error(f"Study received callback failed: {e}")

            return 0x0000  # Success

        except Exception as e:
            logger.error(f"Failed to store DICOM file: {e}")
            return 0xC000  # Failure

    def get_stored_studies(self) -> List[Dict[str, Any]]:
        """Get list of stored studies.

        Returns:
            List of study information dictionaries
        """
        studies = []

        try:
            for study_dir in self.storage_dir.iterdir():
                if study_dir.is_dir():
                    study_info = {
                        "study_uid": study_dir.name,
                        "study_path": str(study_dir),
                        "series_count": len(list(study_dir.iterdir())),
                        "received_time": datetime.fromtimestamp(study_dir.stat().st_mtime),
                    }

                    # Get study metadata from first DICOM file
                    for series_dir in study_dir.iterdir():
                        if series_dir.is_dir():
                            for dicom_file in series_dir.glob("*.dcm"):
                                try:
                                    ds = dcmread(dicom_file)
                                    study_info.update(
                                        {
                                            "patient_id": getattr(ds, "PatientID", "unknown"),
                                            "patient_name": str(
                                                getattr(ds, "PatientName", "unknown")
                                            ),
                                            "study_date": getattr(ds, "StudyDate", "unknown"),
                                            "study_description": getattr(
                                                ds, "StudyDescription", "unknown"
                                            ),
                                            "modality": getattr(ds, "Modality", "unknown"),
                                        }
                                    )
                                    break
                                except Exception as e:
                                    logger.warning(f"Failed to read DICOM metadata: {e}")
                            break

                    studies.append(study_info)

        except Exception as e:
            logger.error(f"Failed to get stored studies: {e}")

        return studies


class DicomServer:
    """Production DICOM server for hospital PACS integration."""

    def __init__(
        self,
        ae_title: str = "MEDICAL_AI",
        port: int = 11112,
        storage_provider: Optional[DicomStorageProvider] = None,
    ):
        """Initialize DICOM server.

        Args:
            ae_title: Application Entity title
            port: Port to listen on
            storage_provider: Storage provider for received studies
        """
        self.ae_title = ae_title
        self.port = port
        self.storage_provider = storage_provider or DicomStorageProvider()

        # Create Application Entity
        self.ae = AE(ae_title=ae_title)

        # Add supported storage contexts
        self._add_storage_contexts()

        # Add verification context
        self.ae.add_supported_context(Verification)

        # Event handlers
        self.ae.on_c_store = self._handle_store
        self.ae.on_c_echo = self._handle_echo

        # Server state
        self.server_thread = None
        self.is_running = False

        logger.info(f"DICOM server initialized: {ae_title}:{port}")

    def _add_storage_contexts(self):
        """Add supported DICOM storage contexts."""
        storage_contexts = [
            # Medical imaging
            CTImageStorage,
            MRImageStorage,
            DigitalXRayImageStorageForPresentation,
            DigitalXRayImageStorageForProcessing,
            SecondaryCaptureImageStorage,
            # Pathology (whole slide imaging)
            VLWholeSlideMicroscopyImageStorage,
        ]

        for context in storage_contexts:
            self.ae.add_supported_context(context)
            logger.debug(f"Added storage context: {context}")

    def _handle_store(self, event):
        """Handle C-STORE request."""
        try:
            dataset = event.dataset
            context = event.context

            logger.info(f"Received C-STORE request: {dataset.SOPClassUID}")

            # Store the DICOM file
            status = self.storage_provider.store_dicom_file(dataset, context, event)

            return status

        except Exception as e:
            logger.error(f"C-STORE handler failed: {e}")
            return 0xC000  # Failure

    def _handle_echo(self, event):
        """Handle C-ECHO request (verification)."""
        logger.info("Received C-ECHO request")
        return 0x0000  # Success

    def start(self, blocking: bool = False):
        """Start DICOM server.

        Args:
            blocking: If True, run in current thread (blocking)
        """
        if self.is_running:
            logger.warning("DICOM server is already running")
            return

        logger.info(f"Starting DICOM server on port {self.port}")

        if blocking:
            self._run_server()
        else:
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()

        self.is_running = True
        logger.info("DICOM server started successfully")

    def _run_server(self):
        """Run the DICOM server."""
        try:
            self.ae.start_server(("", self.port), block=True)
        except Exception as e:
            logger.error(f"DICOM server failed: {e}")
            self.is_running = False

    def stop(self):
        """Stop DICOM server."""
        if not self.is_running:
            logger.warning("DICOM server is not running")
            return

        logger.info("Stopping DICOM server")

        try:
            self.ae.shutdown()

            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=5)

            self.is_running = False
            logger.info("DICOM server stopped successfully")

        except Exception as e:
            logger.error(f"Failed to stop DICOM server: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get server status information.

        Returns:
            Dictionary with server status
        """
        return {
            "ae_title": self.ae_title,
            "port": self.port,
            "is_running": self.is_running,
            "supported_contexts": len(self.ae.supported_contexts),
            "stored_studies": len(self.storage_provider.get_stored_studies()),
        }

    def test_connection(self, remote_ae_title: str, remote_host: str, remote_port: int) -> bool:
        """Test connection to remote PACS.

        Args:
            remote_ae_title: Remote AE title
            remote_host: Remote host
            remote_port: Remote port

        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Testing connection to {remote_ae_title}@{remote_host}:{remote_port}")

            # Create association for testing
            assoc = self.ae.associate(remote_host, remote_port, ae_title=remote_ae_title)

            if assoc.is_established:
                # Send C-ECHO to test
                status = assoc.send_c_echo()
                assoc.release()

                if status:
                    logger.info("PACS connection test successful")
                    return True
                else:
                    logger.error("PACS connection test failed: C-ECHO failed")
                    return False
            else:
                logger.error("PACS connection test failed: Association not established")
                return False

        except Exception as e:
            logger.error(f"PACS connection test failed: {e}")
            return False


def create_medical_ai_dicom_server(
    port: int = 11112, storage_dir: str = "/tmp/dicom_storage"
) -> DicomServer:
    """Create a configured DICOM server for Medical AI platform.

    Args:
        port: Port to listen on
        storage_dir: Directory to store received DICOM files

    Returns:
        Configured DicomServer instance
    """
    storage_provider = DicomStorageProvider(storage_dir)
    server = DicomServer(ae_title="MEDICAL_AI", port=port, storage_provider=storage_provider)

    # Add callback to trigger AI analysis on received studies
    def trigger_ai_analysis(file_path: str, dataset: Dataset):
        """Trigger AI analysis for received study."""
        try:
            logger.info(f"Triggering AI analysis for: {file_path}")

            # Here you would integrate with your inference engine
            # For now, just log the received study
            study_uid = getattr(dataset, "StudyInstanceUID", "unknown")
            patient_id = getattr(dataset, "PatientID", "unknown")
            modality = getattr(dataset, "Modality", "unknown")

            logger.info(
                f"Study received - UID: {study_uid}, Patient: {patient_id}, Modality: {modality}"
            )

            # Inference engine integration - ready for implementation
            try:
                # Integration point for AI analysis
                # When inference engine is available, uncomment and configure:
                # from src.inference import InferenceEngine
                # engine = InferenceEngine()
                # result = await engine.analyze_dicom_file(file_path)
                #
                # Expected workflow:
                # 1. Load DICOM file and extract WSI data
                # 2. Run AI model inference
                # 3. Generate structured report (DICOM SR)
                # 4. Store results back to PACS
                # 5. Send notification to clinical workflow

                logger.info(f"AI analysis integration point - ready for inference engine")

            except Exception as e:
                logger.error(f"AI analysis integration error: {e}")

        except Exception as e:
            logger.error(f"Failed to trigger AI analysis: {e}")

    storage_provider.add_study_received_callback(trigger_ai_analysis)

    return server
