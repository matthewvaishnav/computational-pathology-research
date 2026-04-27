"""
Active Learning Connector

Connects active learning system to annotation interface.
Automatically queues high-uncertainty cases for expert review.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

from ...continuous_learning.active_learning import (
    ActiveLearningSystem,
    CaseForReview,
    AnnotationTask as ALAnnotationTask,
    ExpertAnnotation
)
from ..backend.annotation_models import AnnotationQueueItem
from ..backend.annotation_api import add_task_to_queue, add_slide_to_db
from ..backend.annotation_models import SlideInfo

logger = logging.getLogger(__name__)


class ActiveLearningConnector:
    """
    Connects active learning system to annotation interface.
    
    Responsibilities:
    - Monitor active learning system for uncertain cases
    - Convert AL cases to annotation queue items
    - Submit cases to annotation interface
    - Collect expert feedback and send to AL system
    """
    
    def __init__(
        self,
        active_learning_system: ActiveLearningSystem,
        auto_queue_threshold: float = 0.85,
        poll_interval: float = 60.0
    ):
        """
        Initialize active learning connector.
        
        Args:
            active_learning_system: Active learning system instance
            auto_queue_threshold: Uncertainty threshold for auto-queuing
            poll_interval: Seconds between polling for new cases
        """
        self.al_system = active_learning_system
        self.auto_queue_threshold = auto_queue_threshold
        self.poll_interval = poll_interval
        
        self._running = False
        self._poll_task = None
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start automatic case queuing."""
        if self._running:
            return
        
        self._running = True
        self._poll_task = asyncio.create_task(self._polling_loop())
        
        self.logger.info("Active learning connector started")
    
    async def stop(self):
        """Stop automatic case queuing."""
        self._running = False
        
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Active learning connector stopped")
    
    async def _polling_loop(self):
        """Background loop for polling active learning system."""
        while self._running:
            try:
                await self._process_uncertain_cases()
                await asyncio.sleep(self.poll_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(10)
    
    async def _process_uncertain_cases(self):
        """Process uncertain cases from active learning system."""
        try:
            # Get annotation queue from AL system
            al_queue = self.al_system.get_annotation_queue(limit=50)
            
            # Convert and submit to annotation interface
            for al_task in al_queue:
                await self._submit_case_to_annotation_interface(al_task)
        
        except Exception as e:
            self.logger.error(f"Failed to process uncertain cases: {e}")
    
    async def _submit_case_to_annotation_interface(
        self,
        al_task: ALAnnotationTask
    ):
        """
        Submit case from AL system to annotation interface.
        
        Args:
            al_task: Active learning annotation task
        """
        try:
            case = al_task.case_data
            
            # Create slide info for annotation interface
            slide_info = SlideInfo(
                slide_id=case.slide_id,
                image_path=case.image_path,
                width=10000,  # Will be updated from actual slide
                height=10000,
                tile_size=256,
                max_zoom=10,
                metadata={
                    'case_id': case.case_id,
                    'disease_type': case.disease_type,
                    'uncertainty_score': case.uncertainty_score,
                    'ai_prediction': case.prediction
                }
            )
            
            # Add slide to database
            add_slide_to_db(slide_info)
            
            # Create annotation queue item
            queue_item = AnnotationQueueItem(
                task_id=al_task.task_id,
                slide_id=case.slide_id,
                priority=al_task.priority,
                uncertainty_score=al_task.uncertainty_score,
                ai_prediction=al_task.ai_prediction,
                status='pending',
                created_at=al_task.created_at,
                assigned_expert=al_task.assigned_expert
            )
            
            # Add to annotation queue
            add_task_to_queue(queue_item)
            
            self.logger.info(
                f"Submitted case {case.case_id} to annotation interface "
                f"(uncertainty: {case.uncertainty_score:.2f})"
            )
        
        except Exception as e:
            self.logger.error(f"Failed to submit case to annotation interface: {e}")
    
    async def submit_expert_feedback(
        self,
        task_id: str,
        expert_id: str,
        diagnosis: str,
        confidence: float,
        annotation_time: float,
        comments: str = "",
        grade: Optional[str] = None,
        stage: Optional[str] = None
    ) -> bool:
        """
        Submit expert feedback to active learning system.
        
        Args:
            task_id: Annotation task ID
            expert_id: Expert identifier
            diagnosis: Expert diagnosis
            confidence: Expert confidence (0-1)
            annotation_time: Time spent annotating (seconds)
            comments: Optional comments
            grade: Optional tumor grade
            stage: Optional tumor stage
        
        Returns:
            Success status
        """
        try:
            # Get case ID from task
            al_queue = self.al_system.get_annotation_queue()
            al_task = next((t for t in al_queue if t.task_id == task_id), None)
            
            if not al_task:
                self.logger.warning(f"Task {task_id} not found in AL system")
                return False
            
            # Create expert annotation
            expert_annotation = ExpertAnnotation(
                case_id=al_task.case_data.case_id,
                expert_id=expert_id,
                diagnosis=diagnosis,
                confidence=confidence,
                grade=grade,
                stage=stage,
                comments=comments,
                annotation_time_seconds=annotation_time,
                quality_score=1.0
            )
            
            # Submit to AL system
            success = self.al_system.receive_expert_feedback(
                task_id,
                expert_annotation
            )
            
            if success:
                self.logger.info(f"Submitted expert feedback for task {task_id}")
            
            return success
        
        except Exception as e:
            self.logger.error(f"Failed to submit expert feedback: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        al_stats = self.al_system.get_statistics()
        
        return {
            'active_learning': al_stats,
            'connector_status': 'running' if self._running else 'stopped',
            'auto_queue_threshold': self.auto_queue_threshold,
            'poll_interval': self.poll_interval
        }
