"""
PACS Workflow Orchestration System

This module provides automated workflow orchestration for PACS operations,
including polling for new studies, priority-based processing, and workflow sequencing.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime, timedelta
from queue import PriorityQueue, Queue, Empty
import uuid

from .error_handling import PACSErrorManager, ErrorContext, ErrorType, ErrorSeverity
from .failover import FailoverManager, PACSEndpoint
from ..workflow import ClinicalWorkflowSystem
from ...data.wsi_pipeline import BatchProcessor


logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Stages in the PACS workflow."""
    DISCOVERY = "discovery"
    QUERYING = "querying"
    RETRIEVAL = "retrieval"
    PROCESSING = "processing"
    ANALYSIS = "analysis"
    STORAGE = "storage"
    NOTIFICATION = "notification"
    COMPLETED = "completed"
    FAILED = "failed"


class StudyPriority(Enum):
    """Priority levels for study processing."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class WorkflowTask:
    """Represents a workflow task."""
    task_id: str
    study_uid: str
    patient_id: str
    stage: WorkflowStage
    priority: StudyPriority
    created_at: datetime
    updated_at: datetime
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_context: Optional[ErrorContext] = None
    
    def __lt__(self, other):
        """Enable priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value  # Higher priority first
        return self.created_at < other.created_at  # FIFO for same priority


@dataclass
class StudyWorkflow:
    """Represents the complete workflow for a study."""
    study_uid: str
    patient_id: str
    current_stage: WorkflowStage
    priority: StudyPriority
    created_at: datetime
    updated_at: datetime
    completed_stages: Set[WorkflowStage] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_history: List[ErrorContext] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class WorkflowMetrics:
    """Tracks workflow performance metrics."""
    
    def __init__(self):
        self.total_studies_processed = 0
        self.studies_by_priority = {priority: 0 for priority in StudyPriority}
        self.studies_by_stage = {stage: 0 for stage in WorkflowStage}
        self.average_processing_time = 0.0
        self.success_rate = 0.0
        self.error_rate = 0.0
        self.throughput_per_hour = 0.0
        self.queue_depth = 0
        self.active_workflows = 0
        
        self._processing_times: List[float] = []
        self._successful_studies = 0
        self._failed_studies = 0
        self._start_time = time.time()
    
    def record_study_completion(self, processing_time: float, success: bool, priority: StudyPriority):
        """Record completion of a study."""
        self.total_studies_processed += 1
        self.studies_by_priority[priority] += 1
        
        if success:
            self._successful_studies += 1
            self._processing_times.append(processing_time)
            
            # Keep only recent processing times (last 100)
            if len(self._processing_times) > 100:
                self._processing_times = self._processing_times[-100:]
            
            # Update average processing time
            if self._processing_times:
                self.average_processing_time = sum(self._processing_times) / len(self._processing_times)
        else:
            self._failed_studies += 1
        
        # Update rates
        if self.total_studies_processed > 0:
            self.success_rate = (self._successful_studies / self.total_studies_processed) * 100
            self.error_rate = (self._failed_studies / self.total_studies_processed) * 100
        
        # Update throughput
        elapsed_hours = (time.time() - self._start_time) / 3600
        if elapsed_hours > 0:
            self.throughput_per_hour = self.total_studies_processed / elapsed_hours
    
    def update_stage_count(self, stage: WorkflowStage):
        """Update count for a workflow stage."""
        self.studies_by_stage[stage] += 1
    
    def update_queue_metrics(self, queue_depth: int, active_workflows: int):
        """Update queue and active workflow metrics."""
        self.queue_depth = queue_depth
        self.active_workflows = active_workflows


class WorkflowOrchestrator:
    """Main workflow orchestrator for PACS operations."""
    
    def __init__(
        self,
        failover_manager: FailoverManager,
        error_manager: PACSErrorManager,
        clinical_workflow: ClinicalWorkflowSystem,
        wsi_pipeline: BatchProcessor,
        polling_interval: float = 60.0,
        max_concurrent_workflows: int = 10,
        max_concurrent_retrievals: int = 5
    ):
        self.failover_manager = failover_manager
        self.error_manager = error_manager
        self.clinical_workflow = clinical_workflow
        self.wsi_pipeline = wsi_pipeline
        self.polling_interval = polling_interval
        self.max_concurrent_workflows = max_concurrent_workflows
        self.max_concurrent_retrievals = max_concurrent_retrievals
        
        # Workflow queues
        self.task_queue: PriorityQueue[WorkflowTask] = PriorityQueue()
        self.active_workflows: Dict[str, StudyWorkflow] = {}
        self.completed_workflows: Dict[str, StudyWorkflow] = {}
        
        # Threading and concurrency
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_workflows)
        self.retrieval_semaphore = asyncio.Semaphore(max_concurrent_retrievals)
        
        # Control flags
        self._running = False
        self._polling_task = None
        self._processing_task = None
        
        # Metrics and monitoring
        self.metrics = WorkflowMetrics()
        self.logger = logging.getLogger(f"{__name__}.WorkflowOrchestrator")
        
        # Configuration
        self.config = {
            'discovery_query_filters': {
                'modality': 'SM',  # Slide Microscopy
                'study_date_range': 7,  # Days
                'max_results_per_query': 100
            },
            'priority_mapping': {
                'STAT': StudyPriority.CRITICAL,
                'URGENT': StudyPriority.URGENT,
                'HIGH': StudyPriority.HIGH,
                'ROUTINE': StudyPriority.NORMAL,
                'LOW': StudyPriority.LOW
            },
            'retry_delays': {
                WorkflowStage.QUERYING: 30,
                WorkflowStage.RETRIEVAL: 60,
                WorkflowStage.PROCESSING: 120,
                WorkflowStage.ANALYSIS: 180,
                WorkflowStage.STORAGE: 60,
                WorkflowStage.NOTIFICATION: 30
            }
        }
    
    async def start(self):
        """Start the workflow orchestrator."""
        if self._running:
            return
        
        self._running = True
        
        # Start failover manager
        await self.failover_manager.start()
        
        # Start background tasks
        self._polling_task = asyncio.create_task(self._polling_loop())
        self._processing_task = asyncio.create_task(self._processing_loop())
        
        self.logger.info("Workflow orchestrator started")
    
    async def stop(self):
        """Stop the workflow orchestrator."""
        self._running = False
        
        # Cancel background tasks
        if self._polling_task:
            self._polling_task.cancel()
        if self._processing_task:
            self._processing_task.cancel()
        
        # Wait for tasks to complete
        for task in [self._polling_task, self._processing_task]:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop failover manager
        await self.failover_manager.stop()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Workflow orchestrator stopped")
    
    async def _polling_loop(self):
        """Background loop for polling PACS for new studies."""
        while self._running:
            try:
                await self._discover_new_studies()
                await asyncio.sleep(self.polling_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(10)  # Short delay before retry
    
    async def _processing_loop(self):
        """Background loop for processing workflow tasks."""
        while self._running:
            try:
                # Update metrics
                self.metrics.update_queue_metrics(
                    self.task_queue.qsize(),
                    len(self.active_workflows)
                )
                
                # Process pending tasks
                await self._process_pending_tasks()
                await asyncio.sleep(1)  # Short delay between processing cycles
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _discover_new_studies(self):
        """Discover new WSI studies from PACS."""
        try:
            endpoint = await self.failover_manager.get_healthy_endpoint()
            if not endpoint:
                self.logger.warning("No healthy PACS endpoints available for discovery")
                return
            
            # Build discovery query
            query_filters = self.config['discovery_query_filters']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=query_filters['study_date_range'])
            
            query_params = {
                'StudyDate': f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}",
                'Modality': query_filters['modality'],
                'QueryRetrieveLevel': 'STUDY'
            }
            
            # Execute discovery query with failover
            studies = await self.failover_manager.execute_with_failover(
                self._execute_discovery_query,
                query_params,
                query_filters['max_results_per_query']
            )
            
            # Queue new studies for processing
            new_studies_count = 0
            for study in studies:
                if await self._queue_new_study(study):
                    new_studies_count += 1
            
            if new_studies_count > 0:
                self.logger.info(f"Discovered {new_studies_count} new studies for processing")
                
        except Exception as e:
            error_context = await self.error_manager.handle_error(
                e, "study_discovery", {}, endpoint.name if endpoint else None
            )
            self.logger.error(f"Failed to discover new studies: {e}")
    
    async def _execute_discovery_query(self, connection, query_params: Dict[str, str], max_results: int) -> List[Dict[str, Any]]:
        """Execute discovery query against PACS."""
        # This would use the actual DICOM query implementation
        # For now, return mock data structure
        studies = []
        
        # Mock implementation - in real code this would use pynetdicom
        # to execute C-FIND operations
        
        return studies
    
    async def _queue_new_study(self, study_data: Dict[str, Any]) -> bool:
        """Queue a new study for processing if not already processed."""
        study_uid = study_data.get('StudyInstanceUID')
        patient_id = study_data.get('PatientID')
        
        if not study_uid or not patient_id:
            self.logger.warning("Study missing required identifiers, skipping")
            return False
        
        # Check if already processed or in progress
        if (study_uid in self.active_workflows or 
            study_uid in self.completed_workflows):
            return False
        
        # Determine priority
        priority_str = study_data.get('StudyPriority', 'ROUTINE')
        priority = self.config['priority_mapping'].get(priority_str, StudyPriority.NORMAL)
        
        # Create workflow task
        task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            study_uid=study_uid,
            patient_id=patient_id,
            stage=WorkflowStage.QUERYING,
            priority=priority,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=study_data
        )
        
        # Add to queue
        self.task_queue.put(task)
        
        # Create workflow tracking
        workflow = StudyWorkflow(
            study_uid=study_uid,
            patient_id=patient_id,
            current_stage=WorkflowStage.QUERYING,
            priority=priority,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=study_data
        )
        
        self.active_workflows[study_uid] = workflow
        
        self.logger.debug(f"Queued new study for processing: {study_uid} (priority: {priority.name})")
        return True
    
    async def _process_pending_tasks(self):
        """Process pending workflow tasks."""
        # Limit concurrent processing
        if len(self.active_workflows) >= self.max_concurrent_workflows:
            return
        
        try:
            # Get next task (blocks briefly)
            task = self.task_queue.get(timeout=0.1)
            
            # Process task asynchronously
            asyncio.create_task(self._process_workflow_task(task))
            
        except:
            # No tasks available
            pass
    
    async def _process_workflow_task(self, task: WorkflowTask):
        """Process a single workflow task."""
        workflow = self.active_workflows.get(task.study_uid)
        if not workflow:
            self.logger.error(f"Workflow not found for task: {task.study_uid}")
            return
        
        start_time = time.time()
        success = False
        
        try:
            # Update workflow stage
            workflow.current_stage = task.stage
            workflow.updated_at = datetime.now()
            self.metrics.update_stage_count(task.stage)
            
            # Execute stage-specific processing
            if task.stage == WorkflowStage.QUERYING:
                await self._process_querying_stage(task, workflow)
            elif task.stage == WorkflowStage.RETRIEVAL:
                await self._process_retrieval_stage(task, workflow)
            elif task.stage == WorkflowStage.PROCESSING:
                await self._process_processing_stage(task, workflow)
            elif task.stage == WorkflowStage.ANALYSIS:
                await self._process_analysis_stage(task, workflow)
            elif task.stage == WorkflowStage.STORAGE:
                await self._process_storage_stage(task, workflow)
            elif task.stage == WorkflowStage.NOTIFICATION:
                await self._process_notification_stage(task, workflow)
            
            success = True
            
        except Exception as e:
            # Handle task failure
            await self._handle_task_failure(task, workflow, e)
        
        finally:
            # Record metrics
            processing_time = time.time() - start_time
            workflow.performance_metrics[task.stage.value] = processing_time
            
            if success and workflow.current_stage == WorkflowStage.COMPLETED:
                # Workflow completed successfully
                total_time = sum(workflow.performance_metrics.values())
                self.metrics.record_study_completion(total_time, True, workflow.priority)
                
                # Move to completed workflows
                self.completed_workflows[workflow.study_uid] = workflow
                del self.active_workflows[workflow.study_uid]
                
                self.logger.info(
                    f"Workflow completed for study {workflow.study_uid} "
                    f"({total_time:.2f}s total)"
                )
    
    async def _process_querying_stage(self, task: WorkflowTask, workflow: StudyWorkflow):
        """Process the querying stage."""
        # Query PACS for detailed study information
        query_params = {
            'StudyInstanceUID': task.study_uid,
            'QueryRetrieveLevel': 'SERIES'
        }
        
        series_list = await self.failover_manager.execute_with_failover(
            self._execute_series_query,
            query_params
        )
        
        # Filter for WSI series
        wsi_series = [s for s in series_list if self._is_wsi_series(s)]
        
        if not wsi_series:
            raise Exception(f"No WSI series found for study {task.study_uid}")
        
        # Update workflow metadata
        workflow.metadata['series_list'] = wsi_series
        workflow.completed_stages.add(WorkflowStage.QUERYING)
        
        # Queue next stage
        await self._queue_next_stage(task, WorkflowStage.RETRIEVAL)
    
    async def _process_retrieval_stage(self, task: WorkflowTask, workflow: StudyWorkflow):
        """Process the retrieval stage."""
        async with self.retrieval_semaphore:  # Limit concurrent retrievals
            series_list = workflow.metadata.get('series_list', [])
            
            retrieved_files = []
            for series in series_list:
                files = await self.failover_manager.execute_with_failover(
                    self._execute_series_retrieval,
                    series
                )
                retrieved_files.extend(files)
            
            if not retrieved_files:
                raise Exception(f"No files retrieved for study {task.study_uid}")
            
            # Update workflow metadata
            workflow.metadata['retrieved_files'] = retrieved_files
            workflow.completed_stages.add(WorkflowStage.RETRIEVAL)
            
            # Queue next stage
            await self._queue_next_stage(task, WorkflowStage.PROCESSING)
    
    async def _process_processing_stage(self, task: WorkflowTask, workflow: StudyWorkflow):
        """Process the processing stage."""
        retrieved_files = workflow.metadata.get('retrieved_files', [])
        
        # Process files through WSI pipeline
        processed_data = []
        for file_path in retrieved_files:
            # This would integrate with the actual WSI pipeline
            result = await self._process_wsi_file(file_path)
            processed_data.append(result)
        
        # Update workflow metadata
        workflow.metadata['processed_data'] = processed_data
        workflow.completed_stages.add(WorkflowStage.PROCESSING)
        
        # Queue next stage
        await self._queue_next_stage(task, WorkflowStage.ANALYSIS)
    
    async def _process_analysis_stage(self, task: WorkflowTask, workflow: StudyWorkflow):
        """Process the analysis stage."""
        processed_data = workflow.metadata.get('processed_data', [])
        
        # Run AI analysis
        analysis_results = []
        for data in processed_data:
            # This would integrate with the actual AI analysis pipeline
            result = await self._run_ai_analysis(data)
            analysis_results.append(result)
        
        # Update workflow metadata
        workflow.metadata['analysis_results'] = analysis_results
        workflow.completed_stages.add(WorkflowStage.ANALYSIS)
        
        # Queue next stage
        await self._queue_next_stage(task, WorkflowStage.STORAGE)
    
    async def _process_storage_stage(self, task: WorkflowTask, workflow: StudyWorkflow):
        """Process the storage stage."""
        analysis_results = workflow.metadata.get('analysis_results', [])
        
        # Store results back to PACS as Structured Reports
        stored_reports = []
        for result in analysis_results:
            report_uid = await self.failover_manager.execute_with_failover(
                self._store_structured_report,
                result,
                task.study_uid
            )
            stored_reports.append(report_uid)
        
        # Update workflow metadata
        workflow.metadata['stored_reports'] = stored_reports
        workflow.completed_stages.add(WorkflowStage.STORAGE)
        
        # Queue next stage
        await self._queue_next_stage(task, WorkflowStage.NOTIFICATION)
    
    async def _process_notification_stage(self, task: WorkflowTask, workflow: StudyWorkflow):
        """Process the notification stage."""
        # Send notifications to clinical staff
        await self._send_completion_notifications(workflow)
        
        # Update workflow
        workflow.completed_stages.add(WorkflowStage.NOTIFICATION)
        workflow.current_stage = WorkflowStage.COMPLETED
        workflow.updated_at = datetime.now()
    
    async def _handle_task_failure(self, task: WorkflowTask, workflow: StudyWorkflow, error: Exception):
        """Handle task failure with retry logic."""
        # Create error context
        error_context = await self.error_manager.handle_error(
            error,
            f"workflow_{task.stage.value}",
            task.metadata,
            study_uid=task.study_uid,
            patient_id=task.patient_id
        )
        
        # Add to workflow error history
        workflow.error_history.append(error_context)
        
        # Check if should retry
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.updated_at = datetime.now()
            
            # Calculate retry delay
            delay = self.config['retry_delays'].get(task.stage, 60)
            
            self.logger.warning(
                f"Task failed, retrying in {delay}s: {task.study_uid} "
                f"(attempt {task.retry_count}/{task.max_retries})"
            )
            
            # Schedule retry
            asyncio.create_task(self._schedule_retry(task, delay))
        else:
            # Max retries exceeded, mark workflow as failed
            workflow.current_stage = WorkflowStage.FAILED
            workflow.updated_at = datetime.now()
            
            # Record failure metrics
            total_time = sum(workflow.performance_metrics.values())
            self.metrics.record_study_completion(total_time, False, workflow.priority)
            
            # Move to completed workflows (as failed)
            self.completed_workflows[workflow.study_uid] = workflow
            del self.active_workflows[workflow.study_uid]
            
            self.logger.error(
                f"Workflow failed after {task.max_retries} retries: {task.study_uid}"
            )
    
    async def _schedule_retry(self, task: WorkflowTask, delay: float):
        """Schedule a task retry after delay."""
        await asyncio.sleep(delay)
        self.task_queue.put(task)
    
    async def _queue_next_stage(self, current_task: WorkflowTask, next_stage: WorkflowStage):
        """Queue the next stage of workflow processing."""
        next_task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            study_uid=current_task.study_uid,
            patient_id=current_task.patient_id,
            stage=next_stage,
            priority=current_task.priority,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=current_task.metadata
        )
        
        self.task_queue.put(next_task)
    
    # Mock implementations for integration points
    async def _execute_series_query(self, connection, query_params: Dict[str, str]) -> List[Dict[str, Any]]:
        """Execute series-level query."""
        # Mock implementation
        return []
    
    async def _execute_series_retrieval(self, connection, series_data: Dict[str, Any]) -> List[str]:
        """Execute series retrieval."""
        # Mock implementation
        return []
    
    def _is_wsi_series(self, series_data: Dict[str, Any]) -> bool:
        """Check if series is a WSI series."""
        modality = series_data.get('Modality', '')
        return modality == 'SM'  # Slide Microscopy
    
    async def _process_wsi_file(self, file_path: str) -> Dict[str, Any]:
        """Process WSI file through pipeline."""
        # Mock implementation
        return {'file_path': file_path, 'processed': True}
    
    async def _run_ai_analysis(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run AI analysis on processed data."""
        # Mock implementation
        return {'analysis': 'completed', 'confidence': 0.95}
    
    async def _store_structured_report(self, connection, analysis_result: Dict[str, Any], study_uid: str) -> str:
        """Store structured report to PACS."""
        # Mock implementation
        return f"SR_{study_uid}_{int(time.time())}"
    
    async def _send_completion_notifications(self, workflow: StudyWorkflow):
        """Send completion notifications."""
        # Mock implementation
        self.logger.info(f"Notifications sent for study {workflow.study_uid}")
    
    # Public API methods
    def get_workflow_status(self, study_uid: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow."""
        workflow = self.active_workflows.get(study_uid) or self.completed_workflows.get(study_uid)
        if not workflow:
            return None
        
        return {
            'study_uid': workflow.study_uid,
            'patient_id': workflow.patient_id,
            'current_stage': workflow.current_stage.value,
            'priority': workflow.priority.value,
            'created_at': workflow.created_at.isoformat(),
            'updated_at': workflow.updated_at.isoformat(),
            'completed_stages': [stage.value for stage in workflow.completed_stages],
            'performance_metrics': workflow.performance_metrics,
            'error_count': len(workflow.error_history)
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return {
            'total_studies_processed': self.metrics.total_studies_processed,
            'studies_by_priority': {p.name: count for p, count in self.metrics.studies_by_priority.items()},
            'studies_by_stage': {s.name: count for s, count in self.metrics.studies_by_stage.items()},
            'average_processing_time': self.metrics.average_processing_time,
            'success_rate': self.metrics.success_rate,
            'error_rate': self.metrics.error_rate,
            'throughput_per_hour': self.metrics.throughput_per_hour,
            'queue_depth': self.metrics.queue_depth,
            'active_workflows': self.metrics.active_workflows,
            'endpoint_statistics': self.failover_manager.get_endpoint_statistics()
        }
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get list of active workflows."""
        return [
            self.get_workflow_status(study_uid)
            for study_uid in self.active_workflows.keys()
        ]
    
    def get_failed_workflows(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get list of failed workflows."""
        failed = [
            self.get_workflow_status(study_uid)
            for study_uid, workflow in self.completed_workflows.items()
            if workflow.current_stage == WorkflowStage.FAILED
        ]
        
        # Sort by update time (most recent first)
        failed.sort(key=lambda x: x['updated_at'], reverse=True)
        
        if limit:
            failed = failed[:limit]
        
        return failed
    
    async def retry_failed_workflow(self, study_uid: str) -> bool:
        """Retry a failed workflow."""
        workflow = self.completed_workflows.get(study_uid)
        if not workflow or workflow.current_stage != WorkflowStage.FAILED:
            return False
        
        # Move back to active workflows
        self.active_workflows[study_uid] = workflow
        del self.completed_workflows[study_uid]
        
        # Reset workflow state
        workflow.current_stage = WorkflowStage.QUERYING
        workflow.updated_at = datetime.now()
        workflow.error_history.clear()
        
        # Queue for processing
        task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            study_uid=study_uid,
            patient_id=workflow.patient_id,
            stage=WorkflowStage.QUERYING,
            priority=workflow.priority,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=workflow.metadata
        )
        
        self.task_queue.put(task)
        
        self.logger.info(f"Retrying failed workflow: {study_uid}")
        return True