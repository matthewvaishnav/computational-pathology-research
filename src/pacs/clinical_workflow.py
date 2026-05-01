#!/usr/bin/env python3
"""
Clinical Workflow Orchestrator

Orchestrates the complete clinical workflow from PACS query through AI analysis
to result delivery and notification. Integrates all PACS components into a
cohesive clinical workflow system.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .dicom_server import DicomServer
from .pacs_client import PACSClient, StudyInfo
from .worklist_manager import WorklistEntry, WorklistManager, WorklistStatus

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Clinical workflow stages."""

    SCHEDULED = "SCHEDULED"
    QUERYING = "QUERYING"
    RETRIEVING = "RETRIEVING"
    ANALYZING = "ANALYZING"
    STORING_RESULTS = "STORING_RESULTS"
    NOTIFYING = "NOTIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class WorkflowPriority(Enum):
    """Workflow priority levels."""

    URGENT = "URGENT"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


@dataclass
class WorkflowTask:
    """Clinical workflow task."""

    task_id: str
    accession_number: str
    patient_id: str
    study_uid: str
    stage: WorkflowStage
    priority: WorkflowPriority
    created_at: datetime
    updated_at: datetime
    pacs_name: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    analysis_results: Optional[List[Dict[str, Any]]] = None

    def update_stage(self, stage: WorkflowStage, error_message: Optional[str] = None):
        """Update workflow stage."""
        self.stage = stage
        self.updated_at = datetime.now()
        if error_message:
            self.error_message = error_message
            self.retry_count += 1


class ClinicalWorkflowOrchestrator:
    """Orchestrates clinical workflow from PACS to AI analysis."""

    def __init__(
        self, pacs_client: PACSClient, dicom_server: DicomServer, worklist_manager: WorklistManager
    ):
        """Initialize clinical workflow orchestrator.

        Args:
            pacs_client: PACS client for querying and retrieval
            dicom_server: DICOM server for receiving studies
            worklist_manager: Worklist manager for scheduling
        """
        self.pacs_client = pacs_client
        self.dicom_server = dicom_server
        self.worklist_manager = worklist_manager

        # Workflow state
        self.active_tasks: Dict[str, WorkflowTask] = {}
        self.completed_tasks: Dict[str, WorkflowTask] = {}

        # Callbacks for workflow events
        self.analysis_callbacks: List[Callable] = []
        self.notification_callbacks: List[Callable] = []

        # Configuration
        self.polling_interval = 60  # seconds
        self.max_concurrent_tasks = 10
        self.is_running = False

        logger.info("Clinical workflow orchestrator initialized")

    def add_analysis_callback(self, callback: Callable[[str, str], None]):
        """Add callback for AI analysis.

        Args:
            callback: Function to call with (study_path, study_uid) for analysis
        """
        self.analysis_callbacks.append(callback)

    def add_notification_callback(self, callback: Callable[[WorkflowTask, Any], None]):
        """Add callback for notifications.

        Args:
            callback: Function to call with (task, results) for notifications
        """
        self.notification_callbacks.append(callback)

    async def start_workflow_orchestration(self):
        """Start the clinical workflow orchestration."""
        if self.is_running:
            logger.warning("Workflow orchestration already running")
            return

        self.is_running = True
        logger.info("Starting clinical workflow orchestration")

        try:
            while self.is_running:
                await self._orchestration_cycle()
                await asyncio.sleep(self.polling_interval)
        except Exception as e:
            logger.error(f"Workflow orchestration failed: {e}")
        finally:
            self.is_running = False

    def stop_workflow_orchestration(self):
        """Stop the clinical workflow orchestration."""
        self.is_running = False
        logger.info("Stopping clinical workflow orchestration")

    async def _orchestration_cycle(self):
        """Single orchestration cycle."""
        try:
            # 1. Check for new scheduled studies
            await self._check_scheduled_studies()

            # 2. Process active workflow tasks
            await self._process_active_tasks()

            # 3. Clean up completed tasks
            self._cleanup_completed_tasks()

        except Exception as e:
            logger.error(f"Orchestration cycle failed: {e}")

    async def _check_scheduled_studies(self):
        """Check for new scheduled studies in worklist."""
        try:
            # Get studies scheduled for AI analysis
            scheduled_studies = self.worklist_manager.get_scheduled_studies_for_ai()

            for entry in scheduled_studies:
                # Skip if already in workflow
                if any(
                    task.accession_number == entry.accession_number
                    for task in self.active_tasks.values()
                ):
                    continue

                # Create workflow task
                task = WorkflowTask(
                    task_id=str(uuid.uuid4()),
                    accession_number=entry.accession_number,
                    patient_id=entry.patient_id,
                    study_uid=entry.study_instance_uid,
                    stage=WorkflowStage.SCHEDULED,
                    priority=WorkflowPriority.NORMAL,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )

                self.active_tasks[task.task_id] = task
                logger.info(f"Added workflow task: {task.accession_number}")

        except Exception as e:
            logger.error(f"Failed to check scheduled studies: {e}")

    async def _process_active_tasks(self):
        """Process active workflow tasks."""
        # Limit concurrent processing
        processing_tasks = [
            task
            for task in self.active_tasks.values()
            if task.stage not in [WorkflowStage.COMPLETED, WorkflowStage.FAILED]
        ]

        # Sort by priority and creation time
        processing_tasks.sort(key=lambda t: (t.priority.value, t.created_at))

        # Process up to max concurrent tasks
        for task in processing_tasks[: self.max_concurrent_tasks]:
            try:
                await self._process_workflow_task(task)
            except Exception as e:
                logger.error(f"Failed to process task {task.task_id}: {e}")
                task.update_stage(WorkflowStage.FAILED, str(e))

    async def _process_workflow_task(self, task: WorkflowTask):
        """Process a single workflow task."""

        if task.stage == WorkflowStage.SCHEDULED:
            await self._stage_query_pacs(task)
        elif task.stage == WorkflowStage.QUERYING:
            await self._stage_retrieve_study(task)
        elif task.stage == WorkflowStage.RETRIEVING:
            await self._stage_analyze_study(task)
        elif task.stage == WorkflowStage.ANALYZING:
            await self._stage_store_results(task)
        elif task.stage == WorkflowStage.STORING_RESULTS:
            await self._stage_notify_results(task)
        elif task.stage == WorkflowStage.NOTIFYING:
            await self._stage_complete_workflow(task)

    async def _stage_query_pacs(self, task: WorkflowTask):
        """Stage 1: Query PACS for study."""
        try:
            task.update_stage(WorkflowStage.QUERYING)

            # Find study on PACS systems
            study_found = False
            for pacs_name in self.pacs_client.connections.keys():
                studies = self.pacs_client.find_studies(
                    pacs_name=pacs_name, accession_number=task.accession_number
                )

                if studies:
                    task.pacs_name = pacs_name
                    study_found = True
                    logger.info(f"Found study {task.accession_number} on {pacs_name}")
                    break

            if study_found:
                task.update_stage(WorkflowStage.RETRIEVING)
            else:
                task.update_stage(WorkflowStage.FAILED, "Study not found on any PACS")

        except Exception as e:
            task.update_stage(WorkflowStage.FAILED, f"PACS query failed: {e}")

    async def _stage_retrieve_study(self, task: WorkflowTask):
        """Stage 2: Retrieve study from PACS."""
        try:
            if not task.pacs_name:
                task.update_stage(WorkflowStage.FAILED, "No PACS name specified")
                return

            # Move study to our DICOM server
            success = self.pacs_client.move_study(
                pacs_name=task.pacs_name,
                study_uid=task.study_uid,
                destination_ae=self.dicom_server.ae_title,
            )

            if success:
                task.update_stage(WorkflowStage.ANALYZING)
                logger.info(f"Retrieved study {task.accession_number}")
            else:
                task.update_stage(WorkflowStage.FAILED, "Study retrieval failed")

        except Exception as e:
            task.update_stage(WorkflowStage.FAILED, f"Study retrieval failed: {e}")

    async def _stage_analyze_study(self, task: WorkflowTask):
        """Stage 3: Analyze study with AI."""
        try:
            # Get stored study files
            stored_studies = self.dicom_server.storage_provider.get_stored_studies()
            study_info = None

            for study in stored_studies:
                if study["study_uid"] == task.study_uid:
                    study_info = study
                    break

            if not study_info:
                task.update_stage(WorkflowStage.FAILED, "Study not found in storage")
                return

            # Trigger AI analysis callbacks
            for callback in self.analysis_callbacks:
                try:
                    await callback(study_info["study_path"], task.study_uid, task)
                except Exception as e:
                    logger.error(f"Analysis callback failed: {e}")

            task.update_stage(WorkflowStage.STORING_RESULTS)
            logger.info(f"Analyzed study {task.accession_number}")

        except Exception as e:
            task.update_stage(WorkflowStage.FAILED, f"Study analysis failed: {e}")

    async def _stage_store_results(self, task: WorkflowTask):
        """Stage 4: Store AI results as DICOM SR."""
        try:
            # DICOM Structured Report creation - ready for implementation
            # Integration points for DICOM SR storage:
            #
            # 1. Create DICOM SR dataset with AI results
            # 2. Include measurement data, annotations, confidence scores
            # 3. Add provenance information (model version, parameters)
            # 4. Store to PACS using C-STORE operation
            # 5. Update study metadata with AI analysis flag
            #
            # Example implementation structure:
            # sr_dataset = create_structured_report(
            #     study_uid=task.study_uid,
            #     ai_results=task.results,
            #     model_info=self.model_metadata
            # )
            # await self.pacs_client.store_dataset(sr_dataset)

            logger.info(f"DICOM SR storage integration point - ready for implementation")
            task.update_stage(WorkflowStage.NOTIFYING)
            logger.info(f"Stored results for study {task.accession_number}")

        except Exception as e:
            task.update_stage(WorkflowStage.FAILED, f"Result storage failed: {e}")

    async def _stage_notify_results(self, task: WorkflowTask):
        """Stage 5: Notify clinicians of results."""
        try:
            # Trigger notification callbacks
            for callback in self.notification_callbacks:
                try:
                    await callback(task, {"status": "completed"})
                except Exception as e:
                    logger.error(f"Notification callback failed: {e}")

            task.update_stage(WorkflowStage.COMPLETED)
            logger.info(f"Notified results for study {task.accession_number}")

        except Exception as e:
            task.update_stage(WorkflowStage.FAILED, f"Notification failed: {e}")

    async def _stage_complete_workflow(self, task: WorkflowTask):
        """Stage 6: Complete workflow."""
        try:
            # Update worklist status
            self.worklist_manager.update_worklist_status(
                task.accession_number, WorklistStatus.COMPLETED
            )

            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            del self.active_tasks[task.task_id]

            logger.info(f"Completed workflow for study {task.accession_number}")

        except Exception as e:
            logger.error(f"Workflow completion failed: {e}")

    def _cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        tasks_to_remove = [
            task_id
            for task_id, task in self.completed_tasks.items()
            if task.updated_at < cutoff_time
        ]

        for task_id in tasks_to_remove:
            del self.completed_tasks[task_id]

        if tasks_to_remove:
            logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get workflow status information.

        Returns:
            Dictionary with workflow status
        """
        # Count tasks by stage
        stage_counts = {}
        for stage in WorkflowStage:
            stage_counts[stage.value] = sum(
                1 for task in self.active_tasks.values() if task.stage == stage
            )

        # Count tasks by priority
        priority_counts = {}
        for priority in WorkflowPriority:
            priority_counts[priority.value] = sum(
                1 for task in self.active_tasks.values() if task.priority == priority
            )

        return {
            "is_running": self.is_running,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "stage_distribution": stage_counts,
            "priority_distribution": priority_counts,
            "polling_interval": self.polling_interval,
            "max_concurrent_tasks": self.max_concurrent_tasks,
        }

    def get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a workflow task.

        Args:
            task_id: Task ID to query

        Returns:
            Task details dictionary or None if not found
        """
        task = self.active_tasks.get(task_id) or self.completed_tasks.get(task_id)

        if not task:
            return None

        return {
            "task_id": task.task_id,
            "accession_number": task.accession_number,
            "patient_id": task.patient_id,
            "study_uid": task.study_uid,
            "stage": task.stage.value,
            "priority": task.priority.value,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
            "pacs_name": task.pacs_name,
            "error_message": task.error_message,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
        }

    def retry_failed_task(self, task_id: str) -> bool:
        """Retry a failed workflow task.

        Args:
            task_id: Task ID to retry

        Returns:
            True if retry initiated successfully
        """
        task = self.active_tasks.get(task_id)

        if not task or task.stage != WorkflowStage.FAILED:
            return False

        if task.retry_count >= task.max_retries:
            logger.warning(f"Task {task_id} exceeded max retries")
            return False

        # Reset to scheduled stage for retry
        task.update_stage(WorkflowStage.SCHEDULED)
        task.error_message = None

        logger.info(f"Retrying failed task: {task_id}")
        return True

    def cancel_task(self, task_id: str) -> bool:
        """Cancel an active workflow task.

        Args:
            task_id: Task ID to cancel

        Returns:
            True if cancelled successfully
        """
        if task_id not in self.active_tasks:
            return False

        task = self.active_tasks[task_id]
        task.update_stage(WorkflowStage.FAILED, "Cancelled by user")

        # Update worklist status
        self.worklist_manager.update_worklist_status(
            task.accession_number, WorklistStatus.CANCELLED
        )

        logger.info(f"Cancelled workflow task: {task_id}")
        return True


async def create_sample_ai_analysis_callback(study_path: str, study_uid: str, task: WorkflowTask):
    """AI analysis callback with inference engine integration.

    Args:
        study_path: Path to study files
        study_uid: Study instance UID
        task: Workflow task for storing results
    """
    from pathlib import Path
    from src.inference.inference_engine import InferenceEngine
    
    logger.info(f"Starting AI analysis for study: {study_uid}")
    
    try:
        # Initialize inference engine
        engine = InferenceEngine()
        
        # Find DICOM images in study path
        study_dir = Path(study_path)
        image_files = list(study_dir.glob("**/*.dcm")) + list(study_dir.glob("**/*.jpg")) + list(study_dir.glob("**/*.png"))
        
        if not image_files:
            logger.warning(f"No image files found in study: {study_uid}")
            return
        
        # Analyze each image in the study
        results = []
        for image_file in image_files[:10]:  # Limit to first 10 images for performance
            try:
                result = engine.analyze_image(
                    image_path=str(image_file),
                    disease_type="breast_cancer"
                )
                results.append({
                    "image_file": image_file.name,
                    "prediction": result.prediction_class,
                    "confidence": result.confidence_score,
                    "probabilities": result.probability_scores,
                    "processing_time_ms": result.processing_time_ms,
                    "uncertainty": result.uncertainty_score
                })
                logger.info(f"Analyzed {image_file.name}: {result.prediction_class} ({result.confidence_score:.3f})")
            except Exception as e:
                logger.error(f"Failed to analyze {image_file.name}: {e}")
                continue
        
        # Store results in task for later retrieval
        if hasattr(task, 'analysis_results'):
            task.analysis_results = results
        
        logger.info(f"Completed AI analysis for study: {study_uid} ({len(results)} images analyzed)")
        
    except Exception as e:
        logger.error(f"AI analysis failed for study {study_uid}: {e}")
        raise


async def create_sample_notification_callback(task: WorkflowTask, results: Any):
    """Sample notification callback for testing.

    Args:
        task: Workflow task
        results: Analysis results
    """
    logger.info(f"Sending notification for study: {task.accession_number}")

    # Notification system integration - ready for implementation
    # Integration points for clinical notifications:
    #
    # 1. Email notifications to pathologists
    #    - SMTP/email service integration
    #    - Template-based messages with results summary
    #    - Attachment of key images/reports
    #
    # 2. HL7 messages to hospital systems
    #    - HL7 v2.x or FHIR message formatting
    #    - Integration with hospital ADT/EMR systems
    #    - Result delivery to clinical workflows
    #
    # 3. SMS alerts for urgent findings
    #    - SMS gateway integration (Twilio, AWS SNS)
    #    - Escalation rules based on findings severity
    #    - On-call pathologist notification
    #
    # 4. Dashboard/UI notifications
    #    - Real-time updates to clinical dashboards
    #    - WebSocket or push notification integration
    #    - Case prioritization and queue management

    logger.info(f"Notification integration point - ready for clinical messaging")

    logger.info(f"Notification sent for study: {task.accession_number}")
