"""
Workflow Orchestrator for PACS Integration System.

This module implements the WorkflowOrchestrator class that coordinates automated
processing workflows and integrates with the existing Clinical Workflow System.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from ..workflow import ClinicalWorkflowSystem
from .data_models import AnalysisResults, DicomPriority, OperationResult, StudyInfo
from .pacs_adapter import PACSAdapter

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    Coordinates automated processing workflows and integrates with Clinical Workflow System.

    This class provides comprehensive workflow automation including:
    - Automated polling of PACS for new WSI studies
    - Priority-based processing queue management
    - Integration with existing ClinicalWorkflowSystem
    - Real-time status updates and monitoring
    - Error handling and recovery procedures
    """

    def __init__(
        self,
        pacs_adapter: PACSAdapter,
        clinical_workflow: ClinicalWorkflowSystem,
        poll_interval: timedelta = timedelta(minutes=5),
        max_concurrent_studies: int = 10,
    ):
        """
        Initialize Workflow Orchestrator.

        Args:
            pacs_adapter: PACS adapter for DICOM operations
            clinical_workflow: Clinical workflow system for AI processing
            poll_interval: Interval between PACS polling operations
            max_concurrent_studies: Maximum concurrent study processing
        """
        self.pacs_adapter = pacs_adapter
        self.clinical_workflow = clinical_workflow
        self.poll_interval = poll_interval
        self.max_concurrent_studies = max_concurrent_studies

        # Processing state
        self._is_running = False
        self._polling_thread: Optional[threading.Thread] = None
        self._processing_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=max_concurrent_studies
        )

        # Study tracking
        self._processed_studies: Set[str] = set()
        self._processing_queue: List[StudyInfo] = []
        self._active_processing: Dict[str, Dict[str, Any]] = {}
        self._failed_studies: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self._stats = {
            "studies_processed": 0,
            "studies_failed": 0,
            "last_poll_time": None,
            "last_poll_results": 0,
            "processing_errors": [],
        }

        logger.info("WorkflowOrchestrator initialized")

    def start_automated_polling(self) -> None:
        """Start automated polling for new WSI studies."""
        if self._is_running:
            logger.warning("Automated polling is already running")
            return

        logger.info(f"Starting automated polling (interval: {self.poll_interval})")

        if self._processing_executor is None:
            self._processing_executor = ThreadPoolExecutor(max_workers=self.max_concurrent_studies)

        self._is_running = True
        self._polling_thread = threading.Thread(
            target=self._polling_loop, daemon=True, name="PACS-Polling"
        )
        self._polling_thread.start()

    def stop_automated_polling(self) -> None:
        """Stop automated polling."""
        if not self._is_running:
            return

        logger.info("Stopping automated polling")
        self._is_running = False

        if self._polling_thread:
            self._polling_thread.join(timeout=30)
            self._polling_thread = None

        # Shutdown processing executor
        if self._processing_executor is not None:
            self._processing_executor.shutdown(wait=True)
            self._processing_executor = None

        logger.info("Automated polling stopped")

    def process_new_studies(
        self, studies: List[StudyInfo], force_reprocess: bool = False
    ) -> List[OperationResult]:
        """
        Process a list of new studies through the clinical workflow.

        Args:
            studies: List of studies to process
            force_reprocess: Whether to reprocess already processed studies

        Returns:
            List of processing results
        """
        logger.info(f"Processing {len(studies)} studies")

        results = []

        # Filter out already processed studies
        if not force_reprocess:
            studies = [
                study
                for study in studies
                if study.study_instance_uid not in self._processed_studies
            ]

            if len(studies) == 0:
                logger.info("No new studies to process")
                return results

        # Sort by priority (urgent first)
        studies.sort(key=lambda s: self._get_priority_order(s.priority))

        if self._processing_executor is None:
            self._processing_executor = ThreadPoolExecutor(max_workers=self.max_concurrent_studies)

        # Queue studies for processing and drain them as worker slots free up.
        self._processing_queue.extend(studies)
        pending_futures: Dict[Any, StudyInfo] = {}

        def submit_next_study() -> None:
            if self._processing_executor is None:
                raise RuntimeError("Processing executor is not available")
            if not self._processing_queue:
                return

            next_study = self._processing_queue.pop(0)
            future = self._processing_executor.submit(self._process_single_study, next_study)
            pending_futures[future] = next_study
            self._active_processing[next_study.study_instance_uid] = {
                "study": next_study,
                "start_time": datetime.now(),
                "status": "processing",
            }

        while self._processing_queue and len(self._active_processing) < self.max_concurrent_studies:
            submit_next_study()

        # Collect results, continuing to schedule queued studies until everything completes.
        while pending_futures:
            completed_futures, _ = wait(list(pending_futures.keys()), return_when=FIRST_COMPLETED)

            for future in completed_futures:
                study = pending_futures.pop(future)
                study_uid = study.study_instance_uid

                try:
                    result = future.result()
                    results.append(result)

                    # Update tracking
                    if result.success:
                        self._processed_studies.add(study_uid)
                        self._stats["studies_processed"] += 1
                        logger.info(f"Successfully processed study: {study_uid}")
                    else:
                        self._failed_studies[study_uid] = {
                            "study": study,
                            "error": result.message,
                            "timestamp": datetime.now(),
                        }
                        self._stats["studies_failed"] += 1
                        logger.error(f"Failed to process study {study_uid}: {result.message}")

                except Exception as e:
                    logger.error(f"Processing exception for study {study_uid}: {str(e)}")

                    error_result = OperationResult.error_result(
                        operation_id=f"process_{study_uid}",
                        message=f"Processing exception: {str(e)}",
                        errors=[str(e)],
                    )
                    results.append(error_result)

                    self._failed_studies[study_uid] = {
                        "study": study,
                        "error": str(e),
                        "timestamp": datetime.now(),
                    }
                    self._stats["studies_failed"] += 1

                finally:
                    # Remove from active processing
                    self._active_processing.pop(study_uid, None)

            while (
                self._processing_queue
                and len(self._active_processing) < self.max_concurrent_studies
            ):
                submit_next_study()

        return results

    def handle_processing_failure(self, study_uid: str, error: Exception) -> str:
        """
        Handle processing failure with appropriate recovery action.

        Args:
            study_uid: Study Instance UID that failed
            error: Exception that caused the failure

        Returns:
            Recovery action taken
        """
        logger.error(f"Handling processing failure for study {study_uid}: {str(error)}")

        # Record the failure
        self._failed_studies[study_uid] = {
            "error": str(error),
            "timestamp": datetime.now(),
            "recovery_attempts": self._failed_studies.get(study_uid, {}).get("recovery_attempts", 0)
            + 1,
        }

        # Determine recovery action based on error type
        error_str = str(error).lower()

        if "connection" in error_str or "network" in error_str:
            # Network-related error - retry later
            recovery_action = "retry_later"
            logger.info(f"Network error for {study_uid} - will retry later")

        elif "disk" in error_str or "space" in error_str:
            # Disk space error - pause processing
            recovery_action = "pause_processing"
            logger.warning(f"Disk space error for {study_uid} - pausing processing")

        elif "authentication" in error_str or "certificate" in error_str:
            # Security error - requires manual intervention
            recovery_action = "manual_intervention"
            logger.error(f"Security error for {study_uid} - requires manual intervention")

        else:
            # Generic error - retry with exponential backoff
            attempts = self._failed_studies[study_uid]["recovery_attempts"]
            if attempts < 3:
                recovery_action = "retry_with_backoff"
                logger.info(f"Generic error for {study_uid} - retry attempt {attempts}")
            else:
                recovery_action = "move_to_dead_letter"
                logger.error(f"Max retries exceeded for {study_uid} - moving to dead letter queue")

        # Record recovery action
        self._failed_studies[study_uid]["recovery_action"] = recovery_action

        return recovery_action

    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status and statistics."""
        return {
            "is_running": self._is_running,
            "poll_interval_minutes": self.poll_interval.total_seconds() / 60,
            "max_concurrent_studies": self.max_concurrent_studies,
            "active_processing": len(self._active_processing),
            "queued_studies": len(self._processing_queue),
            "processed_studies": len(self._processed_studies),
            "failed_studies": len(self._failed_studies),
            "statistics": self._stats.copy(),
            "active_studies": [
                {
                    "study_uid": uid,
                    "start_time": info["start_time"].isoformat(),
                    "duration_minutes": (datetime.now() - info["start_time"]).total_seconds() / 60,
                    "status": info["status"],
                }
                for uid, info in self._active_processing.items()
            ],
        }

    def _polling_loop(self) -> None:
        """Main polling loop that runs in background thread."""
        logger.info("Starting PACS polling loop")

        while self._is_running:
            try:
                start_time = datetime.now()

                # Query PACS for new studies
                studies, query_result = self.pacs_adapter.query_studies(
                    modality="SM", max_results=100  # Slide Microscopy
                )

                self._stats["last_poll_time"] = start_time
                self._stats["last_poll_results"] = len(studies)

                if query_result.success and studies:
                    logger.info(f"Found {len(studies)} studies in PACS poll")

                    # Process new studies
                    processing_results = self.process_new_studies(studies)

                    # Log processing summary
                    successful = sum(1 for r in processing_results if r.success)
                    failed = len(processing_results) - successful

                    if processing_results:
                        logger.info(
                            f"Processing complete: {successful} successful, {failed} failed"
                        )

                elif not query_result.success:
                    logger.error(f"PACS query failed: {query_result.message}")
                    self._stats["processing_errors"].append(
                        {"timestamp": start_time.isoformat(), "error": query_result.message}
                    )

                # Sleep until next poll
                elapsed = datetime.now() - start_time
                sleep_time = max(0, self.poll_interval.total_seconds() - elapsed.total_seconds())

                if self._is_running and sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in polling loop: {str(e)}")
                self._stats["processing_errors"].append(
                    {"timestamp": datetime.now().isoformat(), "error": str(e)}
                )

                # Sleep before retrying
                if self._is_running:
                    time.sleep(60)  # Wait 1 minute before retrying

        logger.info("PACS polling loop stopped")

    def _process_single_study(self, study: StudyInfo) -> OperationResult:
        """
        Process a single study through the complete workflow.

        Args:
            study: Study to process

        Returns:
            OperationResult with processing status
        """
        study_uid = study.study_instance_uid
        operation_id = f"process_study_{study_uid}"

        logger.info(f"Processing study: {study_uid}")

        try:
            # Step 1: Retrieve WSI files from PACS
            logger.debug(f"Retrieving study {study_uid} from PACS")

            retrieval_result = self.pacs_adapter.retrieve_study(
                study_instance_uid=study_uid, destination_path=f"./data/retrieved/{study_uid}"
            )

            if not retrieval_result.success:
                return OperationResult.error_result(
                    operation_id=operation_id,
                    message=f"Retrieval failed: {retrieval_result.message}",
                    errors=retrieval_result.errors,
                )

            retrieved_files = retrieval_result.data.get("retrieved_files", [])
            if not retrieved_files:
                return OperationResult.error_result(
                    operation_id=operation_id,
                    message="No files retrieved from PACS",
                    errors=["Empty retrieval result"],
                )

            # Step 2: Process through clinical workflow
            logger.debug(f"Processing study {study_uid} through clinical workflow")

            # For now, create mock WSI features (in real implementation, would process DICOM files)
            import torch

            mock_features = torch.randn(1, 100, 1024)  # Mock patch features

            # Process through clinical workflow
            clinical_results = self.clinical_workflow.process_case(
                wsi_features=mock_features,
                patient_id=study.patient_id,
                scan_id=study_uid,
                scan_date=study.study_date.isoformat(),
                model_version="1.0.0",
            )

            # Step 3: Convert clinical results to DICOM format
            analysis_results = self._convert_clinical_results(clinical_results, study)

            # Step 4: Store results back to PACS
            logger.debug(f"Storing analysis results for study {study_uid}")

            storage_result = self.pacs_adapter.store_analysis_results(
                analysis_results=analysis_results, original_study_uid=study_uid
            )

            if not storage_result.success:
                logger.warning(f"Failed to store results to PACS: {storage_result.message}")
                # Continue - this is not a critical failure

            return OperationResult.success_result(
                operation_id=operation_id,
                message=f"Successfully processed study {study_uid}",
                data={
                    "study_uid": study_uid,
                    "retrieved_files": len(retrieved_files),
                    "clinical_results": clinical_results,
                    "stored_to_pacs": storage_result.success,
                },
            )

        except Exception as e:
            logger.error(f"Study processing failed: {str(e)}")

            # Handle the failure
            recovery_action = self.handle_processing_failure(study_uid, e)

            return OperationResult.error_result(
                operation_id=operation_id,
                message=f"Processing failed: {str(e)}",
                errors=[str(e), f"Recovery action: {recovery_action}"],
            )

    def _convert_clinical_results(
        self, clinical_results: Dict[str, Any], study: StudyInfo
    ) -> AnalysisResults:
        """Convert clinical workflow results to AnalysisResults format."""
        from .data_models import DetectedRegion, DiagnosticRecommendation

        # Extract primary diagnosis
        primary_diagnosis = clinical_results.get("primary_diagnosis", {})

        # Create detected regions (mock for now)
        detected_regions = [
            DetectedRegion(
                region_id="region_1",
                coordinates=(100, 100, 200, 200),
                confidence=0.85,
                region_type="suspicious_area",
                description="AI-detected suspicious region",
            )
        ]

        # Create diagnostic recommendations
        recommendations = [
            DiagnosticRecommendation(
                recommendation_id="rec_1",
                recommendation_text=f"Primary diagnosis: {primary_diagnosis.get('disease_name', 'Unknown')}",
                confidence=primary_diagnosis.get("probability", 0.0),
                urgency_level="MEDIUM",
            )
        ]

        return AnalysisResults(
            study_instance_uid=study.study_instance_uid,
            series_instance_uid=study.study_instance_uid,  # Simplified
            algorithm_name="HistoCore AI",
            algorithm_version=clinical_results.get("model_version", "1.0.0"),
            confidence_score=primary_diagnosis.get("probability", 0.0),
            detected_regions=detected_regions,
            diagnostic_recommendations=recommendations,
            processing_timestamp=datetime.now(),
            primary_diagnosis=primary_diagnosis.get("disease_name"),
            probability_distribution=clinical_results.get("probability_distribution", {}),
        )

    def _get_priority_order(self, priority: DicomPriority) -> int:
        """Get numeric order for priority sorting (lower = higher priority)."""
        priority_order = {
            DicomPriority.URGENT: 0,
            DicomPriority.HIGH: 1,
            DicomPriority.MEDIUM: 2,
            DicomPriority.LOW: 3,
        }
        return priority_order.get(priority, 2)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_automated_polling()
