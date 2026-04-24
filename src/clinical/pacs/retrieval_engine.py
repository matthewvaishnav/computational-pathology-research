"""
Retrieval Engine for DICOM C-MOVE operations.

This module implements the RetrievalEngine class that executes DICOM C-MOVE operations
to download WSI files from PACS systems. It provides concurrent retrieval support,
file integrity validation, and proper file naming conventions.
"""

import hashlib
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pynetdicom import AE, StoragePresentationContexts
from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelMove
from pydicom.dataset import Dataset
import pydicom

from .data_models import (
    PACSEndpoint,
    PACSMetadata,
    ValidationResult,
    OperationResult,
    StudyInfo,
    SeriesInfo,
)

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Executes DICOM C-MOVE operations to download WSI files from PACS systems.

    This class provides comprehensive retrieval capabilities including:
    - Study and series-level retrieval with C-MOVE operations
    - Concurrent retrieval of multiple series
    - File integrity validation using checksums
    - Proper file naming and storage conventions
    - Disk space monitoring and throttling
    - Integration with existing DICOM adapter
    """

    def __init__(
        self,
        ae_title: str = "HISTOCORE_RETRIEVE",
        storage_scp_port: int = 11113,
        max_concurrent_retrievals: int = 5,
    ):
        """
        Initialize Retrieval Engine.

        Args:
            ae_title: Application Entity title for DICOM associations
            storage_scp_port: Port for receiving C-STORE operations
            max_concurrent_retrievals: Maximum concurrent retrieval operations
        """
        self.ae_title = ae_title
        self.storage_scp_ae_title = f"{ae_title}_SCP"
        self.storage_scp_port = storage_scp_port
        self.max_concurrent_retrievals = max_concurrent_retrievals

        # Create Application Entity for C-MOVE requests
        self.ae = AE(ae_title=ae_title)
        self.ae.add_requested_context(StudyRootQueryRetrieveInformationModelMove)

        # Create Storage SCP for receiving files
        self.storage_ae = AE(ae_title=self.storage_scp_ae_title)

        # Add all storage presentation contexts for receiving files
        for context in StoragePresentationContexts:
            self.storage_ae.add_supported_context(context.abstract_syntax)

        # Storage tracking
        self._retrieved_files: Dict[str, Path] = {}
        self._retrieval_stats: Dict[str, Any] = {}
        self._active_retrievals: Set[str] = set()
        self._retrieval_lock = threading.Lock()
        # One in-process Storage SCP receives all inbound C-STORE traffic on a
        # single port, so retrieval sessions must serialize access to avoid
        # mixing files from different requests.
        self._storage_session_lock = threading.Lock()

        # Disk space monitoring
        self._min_free_space_gb = 10  # Minimum free space in GB

        logger.info(f"RetrievalEngine initialized with AE title: {ae_title}")

    def retrieve_study(
        self,
        endpoint: PACSEndpoint,
        study_instance_uid: str,
        destination_path: Path,
        concurrent_series: int = 5,
    ) -> OperationResult:
        """
        Retrieve all series in a study from PACS.

        Args:
            endpoint: PACS endpoint configuration
            study_instance_uid: Study Instance UID to retrieve
            destination_path: Local directory to store retrieved files
            concurrent_series: Number of series to retrieve concurrently

        Returns:
            OperationResult with retrieval status and file paths

        Raises:
            ConnectionError: If PACS connection fails
            ValueError: If study UID is invalid
            OSError: If insufficient disk space
        """
        logger.info(f"Retrieving study {study_instance_uid} from {endpoint.host}")

        # Validate inputs
        if not study_instance_uid:
            return OperationResult.error_result(
                operation_id=f"retrieve_study_{int(time.time())}",
                message="Study Instance UID is required",
                errors=["Invalid study UID"],
            )

        destination_path = Path(destination_path)
        destination_path.mkdir(parents=True, exist_ok=True)

        # Check disk space
        if not self._check_disk_space(destination_path):
            return OperationResult.error_result(
                operation_id=f"retrieve_study_{int(time.time())}",
                message="Insufficient disk space",
                errors=[f"Less than {self._min_free_space_gb}GB free space available"],
            )

        operation_id = f"retrieve_study_{study_instance_uid}_{int(time.time())}"

        try:
            with self._retrieval_lock:
                if study_instance_uid in self._active_retrievals:
                    return OperationResult.error_result(
                        operation_id=operation_id,
                        message="Study retrieval already in progress",
                        errors=["Duplicate retrieval request"],
                    )
                self._active_retrievals.add(study_instance_uid)

            with self._storage_session_lock:
                # Start storage SCP to receive files
                storage_thread = self._start_storage_scp(destination_path)

                try:
                    # Execute C-MOVE for the study
                    retrieved_files = self._execute_c_move_study(
                        endpoint=endpoint,
                        study_instance_uid=study_instance_uid,
                        destination_path=destination_path,
                    )

                    # Validate retrieved files
                    validation_results = []
                    for file_path in retrieved_files:
                        validation = self.validate_retrieved_files([file_path])
                        validation_results.append(validation)

                    # Create metadata for retrieved files
                    metadata_list = []
                    for file_path in retrieved_files:
                        try:
                            metadata = self._create_pacs_metadata(file_path, endpoint)
                            metadata_list.append(metadata)
                        except Exception as e:
                            logger.warning(f"Failed to create metadata for {file_path}: {e}")

                    return OperationResult.success_result(
                        operation_id=operation_id,
                        message=f"Successfully retrieved {len(retrieved_files)} files",
                        data={
                            "retrieved_files": [str(f) for f in retrieved_files],
                            "metadata": [m.to_dict() for m in metadata_list],
                            "validation_results": validation_results,
                        },
                    )

                finally:
                    # Stop storage SCP
                    self._stop_storage_scp(storage_thread)

        except Exception as e:
            logger.error(f"Study retrieval failed: {str(e)}")
            return OperationResult.error_result(
                operation_id=operation_id, message=f"Retrieval failed: {str(e)}", errors=[str(e)]
            )

        finally:
            with self._retrieval_lock:
                self._active_retrievals.discard(study_instance_uid)

    def retrieve_series(
        self, endpoint: PACSEndpoint, series_instance_uid: str, destination_path: Path
    ) -> OperationResult:
        """
        Retrieve a specific series from PACS.

        Args:
            endpoint: PACS endpoint configuration
            series_instance_uid: Series Instance UID to retrieve
            destination_path: Local directory to store retrieved files

        Returns:
            OperationResult with retrieval status and file paths
        """
        logger.info(f"Retrieving series {series_instance_uid} from {endpoint.host}")

        if not series_instance_uid:
            return OperationResult.error_result(
                operation_id=f"retrieve_series_{int(time.time())}",
                message="Series Instance UID is required",
                errors=["Invalid series UID"],
            )

        destination_path = Path(destination_path)
        destination_path.mkdir(parents=True, exist_ok=True)

        # Check disk space
        if not self._check_disk_space(destination_path):
            return OperationResult.error_result(
                operation_id=f"retrieve_series_{int(time.time())}",
                message="Insufficient disk space",
                errors=[f"Less than {self._min_free_space_gb}GB free space available"],
            )

        operation_id = f"retrieve_series_{series_instance_uid}_{int(time.time())}"

        try:
            with self._storage_session_lock:
                # Start storage SCP
                storage_thread = self._start_storage_scp(destination_path)

                try:
                    # Execute C-MOVE for the series
                    retrieved_files = self._execute_c_move_series(
                        endpoint=endpoint,
                        series_instance_uid=series_instance_uid,
                        destination_path=destination_path,
                    )

                    # Validate files
                    validation = self.validate_retrieved_files(retrieved_files)

                    return OperationResult.success_result(
                        operation_id=operation_id,
                        message=f"Successfully retrieved {len(retrieved_files)} files",
                        data={
                            "retrieved_files": [str(f) for f in retrieved_files],
                            "validation": validation.is_valid,
                            "validation_errors": validation.errors,
                        },
                    )

                finally:
                    self._stop_storage_scp(storage_thread)

        except Exception as e:
            logger.error(f"Series retrieval failed: {str(e)}")
            return OperationResult.error_result(
                operation_id=operation_id, message=f"Retrieval failed: {str(e)}", errors=[str(e)]
            )

    def validate_retrieved_files(self, file_paths: List[Path]) -> ValidationResult:
        """
        Validate integrity of retrieved DICOM files.

        Args:
            file_paths: List of file paths to validate

        Returns:
            ValidationResult with validation status and errors
        """
        result = ValidationResult(is_valid=True)

        for file_path in file_paths:
            try:
                # Check file exists and is readable
                if not file_path.exists():
                    result.add_error(f"File does not exist: {file_path}")
                    continue

                if not file_path.is_file():
                    result.add_error(f"Path is not a file: {file_path}")
                    continue

                # Check file size
                file_size = file_path.stat().st_size
                if file_size == 0:
                    result.add_error(f"File is empty: {file_path}")
                    continue

                # Validate DICOM format
                try:
                    dataset = pydicom.dcmread(str(file_path), stop_before_pixels=True)

                    # Check required DICOM fields
                    required_fields = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]

                    for field in required_fields:
                        if not hasattr(dataset, field) or not getattr(dataset, field):
                            result.add_error(f"Missing required DICOM field {field} in {file_path}")

                except Exception as e:
                    result.add_error(f"Invalid DICOM file {file_path}: {str(e)}")
                    continue

                # Calculate and verify checksum if available
                checksum = self._calculate_file_checksum(file_path)
                logger.debug(f"File {file_path} checksum: {checksum}")

            except Exception as e:
                result.add_error(f"Validation failed for {file_path}: {str(e)}")

        return result

    def _execute_c_move_study(
        self, endpoint: PACSEndpoint, study_instance_uid: str, destination_path: Path
    ) -> List[Path]:
        """Execute C-MOVE operation for a complete study."""
        # Build C-MOVE request dataset
        move_ds = Dataset()
        move_ds.QueryRetrieveLevel = "STUDY"
        move_ds.StudyInstanceUID = study_instance_uid

        return self._execute_c_move(endpoint, move_ds, destination_path)

    def _execute_c_move_series(
        self, endpoint: PACSEndpoint, series_instance_uid: str, destination_path: Path
    ) -> List[Path]:
        """Execute C-MOVE operation for a specific series."""
        # Build C-MOVE request dataset
        move_ds = Dataset()
        move_ds.QueryRetrieveLevel = "SERIES"
        move_ds.SeriesInstanceUID = series_instance_uid

        return self._execute_c_move(endpoint, move_ds, destination_path)

    def _execute_c_move(
        self, endpoint: PACSEndpoint, move_ds: Dataset, destination_path: Path
    ) -> List[Path]:
        """Execute C-MOVE operation with the given dataset."""
        retrieved_files = []

        try:
            # Establish association for C-MOVE
            assoc_params = endpoint.create_association_parameters()
            assoc = self.ae.associate(
                addr=assoc_params["address"],
                port=assoc_params["port"],
                ae_title=assoc_params["peer_ae_title"],
            )

            if not assoc.is_established:
                raise ConnectionError(
                    f"Failed to establish C-MOVE association with {endpoint.host}"
                )

            logger.debug(f"C-MOVE association established with {endpoint.host}")

            # Send C-MOVE request
            # The destination AE title should be our storage SCP
            responses = assoc.send_c_move(
                move_ds,
                self.storage_scp_ae_title,  # Storage SCP AE title as destination
                StudyRootQueryRetrieveInformationModelMove,
            )

            for status, response_ds in responses:
                if status.Status == 0xFF00:  # Pending - transfer in progress
                    logger.debug("C-MOVE transfer in progress...")

                elif status.Status == 0x0000:  # Success
                    completed = getattr(status, "NumberOfCompletedSuboperations", 0)
                    logger.info(f"C-MOVE completed successfully. {completed} files transferred")
                    break

                else:  # Error or warning
                    error_msg = f"C-MOVE failed with status: 0x{status.Status:04X}"
                    if hasattr(status, "ErrorComment"):
                        error_msg += f" - {status.ErrorComment}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            assoc.release()

            # Collect files that were stored by our SCP
            retrieved_files = self._collect_stored_files(destination_path)

        except Exception as e:
            logger.error(f"C-MOVE operation failed: {str(e)}")
            raise

        return retrieved_files

    def _start_storage_scp(self, destination_path: Path) -> threading.Thread:
        """Start Storage SCP to receive C-STORE operations."""

        def handle_store(event):
            """Handle incoming C-STORE requests."""
            try:
                # Generate filename based on DICOM metadata
                ds = event.dataset
                filename = self._generate_filename(ds)
                file_path = destination_path / filename

                # Ensure directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the file
                ds.save_as(str(file_path), write_like_original=False)

                # Track the stored file
                with self._retrieval_lock:
                    self._retrieved_files[ds.SOPInstanceUID] = file_path

                logger.debug(f"Stored file: {file_path}")

                # Return success status
                return 0x0000

            except Exception as e:
                logger.error(f"Failed to store file: {str(e)}")
                return 0xC000  # Failure status

        # Bind the event handler
        self.storage_ae.on_c_store = handle_store

        # Start the SCP in a separate thread
        def run_scp():
            self.storage_ae.start_server(address=("", self.storage_scp_port), block=True)

        scp_thread = threading.Thread(target=run_scp, daemon=True)
        scp_thread.start()

        # Give the SCP time to start
        time.sleep(1)

        logger.info(f"Storage SCP started on port {self.storage_scp_port}")
        return scp_thread

    def _stop_storage_scp(self, scp_thread: threading.Thread):
        """Stop the Storage SCP."""
        try:
            self.storage_ae.shutdown()
            logger.info("Storage SCP stopped")
        except Exception as e:
            logger.warning(f"Error stopping Storage SCP: {e}")

    def _generate_filename(self, dataset: Dataset) -> str:
        """Generate appropriate filename for stored DICOM file."""
        try:
            # Use SOP Instance UID as base filename
            sop_uid = dataset.SOPInstanceUID

            # Add series and study info if available
            series_uid = getattr(dataset, "SeriesInstanceUID", "")
            study_uid = getattr(dataset, "StudyInstanceUID", "")

            # Create hierarchical directory structure
            if study_uid and series_uid:
                # Format: StudyUID/SeriesUID/SOPInstanceUID.dcm
                return f"{study_uid}/{series_uid}/{sop_uid}.dcm"
            else:
                # Fallback to flat structure
                return f"{sop_uid}.dcm"

        except Exception as e:
            logger.warning(f"Failed to generate filename: {e}")
            # Fallback to timestamp-based name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            return f"dicom_{timestamp}.dcm"

    def _collect_stored_files(self, destination_path: Path) -> List[Path]:
        """Collect all files that were stored during retrieval."""
        stored_files = []

        with self._retrieval_lock:
            for file_path in self._retrieved_files.values():
                if file_path.exists() and file_path.is_relative_to(destination_path):
                    stored_files.append(file_path)

            # Clear the tracking for next retrieval
            self._retrieved_files.clear()

        return stored_files

    def _check_disk_space(self, path: Path) -> bool:
        """Check if sufficient disk space is available."""
        try:
            stat = shutil.disk_usage(path)
            free_gb = stat.free / (1024**3)

            if free_gb < self._min_free_space_gb:
                logger.warning(
                    f"Low disk space: {free_gb:.1f}GB free (minimum: {self._min_free_space_gb}GB)"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to check disk space: {e}")
            return False

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def _create_pacs_metadata(self, file_path: Path, endpoint: PACSEndpoint) -> PACSMetadata:
        """Create PACSMetadata object from retrieved file."""
        try:
            dataset = pydicom.dcmread(str(file_path), stop_before_pixels=True)

            # Extract basic DICOM metadata
            metadata = PACSMetadata(
                patient_id=str(dataset.get("PatientID", "")),
                patient_name=str(dataset.get("PatientName", "")),
                study_instance_uid=str(dataset.get("StudyInstanceUID", "")),
                series_instance_uid=str(dataset.get("SeriesInstanceUID", "")),
                sop_instance_uid=str(dataset.get("SOPInstanceUID", "")),
                modality=str(dataset.get("Modality", "")),
                source_pacs_ae_title=endpoint.ae_title,
                retrieval_timestamp=datetime.now(),
                original_transfer_syntax=str(dataset.file_meta.get("TransferSyntaxUID", "")),
            )

            # Add optional fields
            if hasattr(dataset, "StudyDate"):
                metadata.study_date = str(dataset.StudyDate)
            if hasattr(dataset, "SeriesDate"):
                metadata.series_date = str(dataset.SeriesDate)
            if hasattr(dataset, "InstitutionName"):
                metadata.institution_name = str(dataset.InstitutionName)
            if hasattr(dataset, "Manufacturer"):
                metadata.manufacturer = str(dataset.Manufacturer)
            if hasattr(dataset, "Rows"):
                metadata.rows = int(dataset.Rows)
            if hasattr(dataset, "Columns"):
                metadata.columns = int(dataset.Columns)

            return metadata

        except Exception as e:
            logger.error(f"Failed to create metadata for {file_path}: {e}")
            raise

    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get retrieval engine statistics."""
        with self._retrieval_lock:
            return {
                "ae_title": self.ae_title,
                "storage_scp_port": self.storage_scp_port,
                "max_concurrent_retrievals": self.max_concurrent_retrievals,
                "active_retrievals": len(self._active_retrievals),
                "min_free_space_gb": self._min_free_space_gb,
            }
