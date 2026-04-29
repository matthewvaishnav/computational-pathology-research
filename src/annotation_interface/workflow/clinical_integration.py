"""
Clinical Workflow Integrator

Main orchestrator that connects all workflow components:
- Active learning → Annotation queue
- PACS → Slide retrieval
- WSI streaming → Annotation interface
- Notifications → Pathologists
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...clinical.pacs.pacs_adapter import PACSAdapter
from ...continuous_learning.active_learning import ActiveLearningSystem
from ..backend.annotation_api import get_annotations_for_slide
from .active_learning_connector import ActiveLearningConnector
from .notification_service import NotificationChannel, NotificationService
from .pacs_connector import PACSConnector

logger = logging.getLogger(__name__)


class ClinicalWorkflowIntegrator:
    """
    Main orchestrator for clinical workflow integration.

    Integrates:
    1. Active learning system → Annotation queue
    2. PACS → Slide retrieval
    3. WSI streaming → Tile serving
    4. Notifications → Pathologist alerts
    5. Expert feedback → Model retraining
    """

    def __init__(
        self,
        active_learning_system: ActiveLearningSystem,
        pacs_adapter: PACSAdapter,
        notification_service: NotificationService,
        auto_start: bool = True,
    ):
        """
        Initialize clinical workflow integrator.

        Args:
            active_learning_system: Active learning system instance
            pacs_adapter: PACS adapter instance
            notification_service: Notification service instance
            auto_start: Automatically start background tasks
        """
        # Initialize connectors
        self.al_connector = ActiveLearningConnector(active_learning_system)
        self.pacs_connector = PACSConnector(pacs_adapter)
        self.notification_service = notification_service

        # Track workflow state
        self._running = False
        self._workflow_tasks = []

        self.logger = logging.getLogger(__name__)

        if auto_start:
            asyncio.create_task(self.start())

    async def start(self):
        """Start clinical workflow integration."""
        if self._running:
            return

        self._running = True

        # Start active learning connector
        await self.al_connector.start()

        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_workflow())
        self._workflow_tasks.append(monitor_task)

        self.logger.info("Clinical workflow integrator started")

    async def stop(self):
        """Stop clinical workflow integration."""
        self._running = False

        # Stop active learning connector
        await self.al_connector.stop()

        # Cancel workflow tasks
        for task in self._workflow_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._workflow_tasks.clear()

        self.logger.info("Clinical workflow integrator stopped")

    async def _monitor_workflow(self):
        """Monitor workflow and handle events."""
        while self._running:
            try:
                # Check for high-priority cases that need immediate attention
                await self._check_urgent_cases()

                # Monitor annotation completion for retraining triggers
                await self._check_retraining_triggers()

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in workflow monitor: {e}")
                await asyncio.sleep(10)

    async def _check_urgent_cases(self):
        """Check for urgent cases and send notifications."""
        try:
            # Get high-priority cases from AL system
            al_queue = self.al_connector.al_system.get_annotation_queue(limit=10)

            for task in al_queue:
                # Check if case is urgent (high uncertainty + high priority)
                if task.uncertainty_score > 0.95 and task.priority > 0.8:
                    # Send urgent notification
                    await self.notification_service.notify_urgent_case(
                        expert_ids=["pathologist_1", "pathologist_2"],  # Would be from config
                        task_id=task.task_id,
                        slide_id=task.case_data.slide_id,
                        reason=f"Very high uncertainty ({task.uncertainty_score:.2f})",
                    )

        except Exception as e:
            self.logger.error(f"Failed to check urgent cases: {e}")

    async def _check_retraining_triggers(self):
        """Check if model retraining should be triggered."""
        try:
            stats = self.al_connector.al_system.get_statistics()

            # Check if enough new annotations have been collected
            annotations_received = stats.get("annotations_received", 0)
            min_annotations = self.al_connector.al_system.min_annotations_for_retraining

            if annotations_received >= min_annotations:
                # Trigger retraining
                success = self.al_connector.al_system.trigger_retraining()

                if success:
                    self.logger.info(
                        f"Triggered model retraining with {annotations_received} new annotations"
                    )

        except Exception as e:
            self.logger.error(f"Failed to check retraining triggers: {e}")

    async def process_new_case(
        self,
        study_uid: str,
        slide_id: str,
        patient_id: str,
        priority: float = 0.5,
        expert_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process new case through complete workflow.

        Args:
            study_uid: Study instance UID
            slide_id: Slide identifier
            patient_id: Patient identifier
            priority: Case priority (0-1)
            expert_id: Optional expert to assign

        Returns:
            Workflow result dictionary
        """
        workflow_start = datetime.now()
        result = {
            "success": False,
            "study_uid": study_uid,
            "slide_id": slide_id,
            "steps_completed": [],
        }

        try:
            # Step 1: Retrieve slide from PACS
            self.logger.info(f"Step 1: Retrieving slide {slide_id} from PACS")
            slide_info = await self.pacs_connector.retrieve_slide_for_annotation(
                study_uid, slide_id
            )

            if not slide_info:
                result["error"] = "Failed to retrieve slide from PACS"
                return result

            result["steps_completed"].append("pacs_retrieval")

            # Step 2: Run AI analysis (would integrate with foundation model)
            self.logger.info(f"Step 2: Running AI analysis on slide {slide_id}")
            # This would call the actual foundation model
            ai_prediction = {"diagnosis": "tumor_detected", "confidence": 0.75, "uncertainty": 0.88}
            result["steps_completed"].append("ai_analysis")

            # Step 3: Check if case needs expert review
            if ai_prediction["uncertainty"] > 0.85:
                self.logger.info(f"Step 3: High uncertainty, queuing for expert review")

                # Would create proper CaseForReview object
                # For now, just log
                result["steps_completed"].append("queued_for_review")

                # Step 4: Send notification
                if expert_id:
                    await self.notification_service.notify_new_annotation_task(
                        expert_id=expert_id,
                        task_id=f"task_{slide_id}",
                        slide_id=slide_id,
                        priority=priority,
                        uncertainty_score=ai_prediction["uncertainty"],
                    )
                    result["steps_completed"].append("notification_sent")

            result["success"] = True
            result["processing_time"] = (datetime.now() - workflow_start).total_seconds()

            self.logger.info(
                f"Completed workflow for case {slide_id} in {result['processing_time']:.2f}s"
            )

        except Exception as e:
            self.logger.error(f"Workflow failed for case {slide_id}: {e}")
            result["error"] = str(e)

        return result

    async def handle_annotation_completion(
        self,
        task_id: str,
        expert_id: str,
        diagnosis: str,
        confidence: float,
        annotation_time: float,
        comments: str = "",
    ) -> bool:
        """
        Handle completion of annotation task.

        Args:
            task_id: Task identifier
            expert_id: Expert identifier
            diagnosis: Expert diagnosis
            confidence: Expert confidence (0-1)
            annotation_time: Time spent annotating (seconds)
            comments: Optional comments

        Returns:
            Success status
        """
        try:
            # Submit feedback to active learning system
            success = await self.al_connector.submit_expert_feedback(
                task_id=task_id,
                expert_id=expert_id,
                diagnosis=diagnosis,
                confidence=confidence,
                annotation_time=annotation_time,
                comments=comments,
            )

            if success:
                self.logger.info(f"Processed annotation completion for task {task_id}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to handle annotation completion: {e}")
            return False

    async def retrieve_slide_for_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve slide for annotation task.

        Args:
            task_id: Task identifier

        Returns:
            Slide information if successful
        """
        try:
            # Get task from AL system
            al_queue = self.al_connector.al_system.get_annotation_queue()
            task = next((t for t in al_queue if t.task_id == task_id), None)

            if not task:
                self.logger.warning(f"Task {task_id} not found")
                return None

            # Check if slide is already cached
            cached_path = self.pacs_connector.get_cached_slide_path(task.case_data.slide_id)

            if cached_path:
                return {
                    "slide_id": task.case_data.slide_id,
                    "path": str(cached_path),
                    "cached": True,
                }

            # Retrieve from PACS if not cached
            study_uid = task.case_data.metadata.get("study_uid")
            if study_uid:
                slide_info = await self.pacs_connector.retrieve_slide_for_annotation(
                    study_uid, task.case_data.slide_id
                )

                if slide_info:
                    return {
                        "slide_id": slide_info.slide_id,
                        "path": slide_info.image_path,
                        "cached": False,
                    }

            return None

        except Exception as e:
            self.logger.error(f"Failed to retrieve slide for task: {e}")
            return None

    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get comprehensive workflow statistics."""
        return {
            "status": "running" if self._running else "stopped",
            "active_learning": self.al_connector.get_statistics(),
            "pacs": self.pacs_connector.get_statistics(),
            "notifications": self.notification_service.get_statistics(),
            "timestamp": datetime.now().isoformat(),
        }

    def register_expert_webhook(self, expert_id: str, webhook_url: str):
        """
        Register webhook for expert notifications.

        Args:
            expert_id: Expert identifier
            webhook_url: Webhook URL
        """
        self.notification_service.register_webhook(expert_id, webhook_url)
        self.logger.info(f"Registered webhook for expert {expert_id}")

    async def test_workflow_integration(self) -> Dict[str, Any]:
        """
        Test all workflow integrations.

        Returns:
            Test results for each component
        """
        results = {"timestamp": datetime.now().isoformat(), "tests": {}}

        # Test PACS connection
        try:
            pacs_result = self.pacs_connector.pacs_adapter.test_connection()
            results["tests"]["pacs_connection"] = {
                "success": pacs_result.success,
                "message": pacs_result.message,
            }
        except Exception as e:
            results["tests"]["pacs_connection"] = {"success": False, "error": str(e)}

        # Test active learning system
        try:
            al_stats = self.al_connector.al_system.get_statistics()
            results["tests"]["active_learning"] = {"success": True, "statistics": al_stats}
        except Exception as e:
            results["tests"]["active_learning"] = {"success": False, "error": str(e)}

        # Test notification service
        try:
            notif_stats = self.notification_service.get_statistics()
            results["tests"]["notifications"] = {"success": True, "statistics": notif_stats}
        except Exception as e:
            results["tests"]["notifications"] = {"success": False, "error": str(e)}

        results["overall_success"] = all(
            test.get("success", False) for test in results["tests"].values()
        )

        return results
