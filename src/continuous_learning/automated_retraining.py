"""
Automated retraining pipeline for federated learning systems.

Monitors drift detection alerts and triggers retraining when significant
drift is detected. Includes validation, A/B testing, and deployment
automation for medical AI systems.
"""

import logging
import asyncio
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from pathlib import Path

from .drift_detection import ModelDriftDetector, DriftAlert

logger = logging.getLogger(__name__)


class RetrainingStatus(Enum):
    """Status of retraining pipeline."""
    IDLE = "idle"
    TRIGGERED = "triggered"
    PREPARING_DATA = "preparing_data"
    TRAINING = "training"
    VALIDATING = "validating"
    AB_TESTING = "ab_testing"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining."""
    # Trigger conditions
    min_drift_severity: str = "warning"  # "warning" or "critical"
    min_affected_diseases: int = 1
    drift_persistence_hours: float = 2.0  # Hours drift must persist
    max_retraining_frequency_hours: float = 24.0  # Min time between retrainings
    
    # Training configuration
    training_rounds: int = 10
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    min_improvement_threshold: float = 0.01
    
    # A/B testing configuration
    ab_test_duration_hours: float = 48.0
    ab_test_traffic_split: float = 0.5  # 50% new model, 50% old model
    ab_test_success_threshold: float = 0.02  # 2% improvement required
    
    # Deployment configuration
    deployment_strategy: str = "gradual"  # "immediate" or "gradual"
    gradual_rollout_steps: List[float] = None  # [0.1, 0.3, 0.7, 1.0]
    rollout_step_duration_hours: float = 6.0
    
    # Safety configuration
    max_performance_degradation: float = 0.05  # 5% max degradation allowed
    rollback_on_failure: bool = True
    backup_model_retention_days: int = 30

    def __post_init__(self):
        if self.gradual_rollout_steps is None:
            self.gradual_rollout_steps = [0.1, 0.3, 0.7, 1.0]


@dataclass
class RetrainingJob:
    """Represents a retraining job."""
    job_id: str
    trigger_timestamp: float
    affected_diseases: List[str]
    trigger_alerts: List[DriftAlert]
    status: RetrainingStatus
    config: RetrainingConfig
    
    # Progress tracking
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_round: int = 0
    validation_metrics: Dict[str, float] = None
    ab_test_results: Dict[str, Any] = None
    deployment_progress: float = 0.0
    
    # Model information
    old_model_path: Optional[str] = None
    new_model_path: Optional[str] = None
    backup_model_path: Optional[str] = None
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.validation_metrics is None:
            self.validation_metrics = {}
        if self.ab_test_results is None:
            self.ab_test_results = {}


class AutomatedRetrainingPipeline:
    """
    Automated retraining pipeline for federated learning systems.
    
    Monitors drift detection and automatically triggers retraining when
    significant drift is detected. Includes validation, A/B testing,
    and gradual deployment capabilities.
    """
    
    def __init__(
        self,
        drift_detector: ModelDriftDetector,
        config: RetrainingConfig = None,
        model_trainer: Optional[Callable] = None,
        model_validator: Optional[Callable] = None,
        model_deployer: Optional[Callable] = None
    ):
        """
        Initialize automated retraining pipeline.
        
        Args:
            drift_detector: Drift detection system
            config: Retraining configuration
            model_trainer: Function to train new model
            model_validator: Function to validate model
            model_deployer: Function to deploy model
        """
        self.drift_detector = drift_detector
        self.config = config or RetrainingConfig()
        self.model_trainer = model_trainer
        self.model_validator = model_validator
        self.model_deployer = model_deployer
        
        # Job management
        self.active_jobs: Dict[str, RetrainingJob] = {}
        self.completed_jobs: List[RetrainingJob] = []
        self.last_retraining_time: Dict[str, float] = {}
        
        # Callbacks
        self.job_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
        
        # Pipeline state
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Register drift alert callback
        self.drift_detector.add_alert_callback(self._handle_drift_alert)
        
        logger.info("Initialized automated retraining pipeline")
    
    def start_monitoring(self) -> None:
        """Start monitoring for drift and automated retraining."""
        if self.is_running:
            logger.warning("Retraining pipeline already running")
            return
        
        self.is_running = True
        logger.info("Started automated retraining monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring and cancel active jobs."""
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        logger.info("Stopped automated retraining monitoring")
    
    def _handle_drift_alert(self, alert: DriftAlert) -> None:
        """Handle drift alert from drift detector."""
        if not self.is_running:
            return
        
        logger.info(f"Received drift alert: {alert.description} (severity: {alert.severity})")
        
        # Check if alert meets trigger criteria
        if not self._should_trigger_retraining(alert):
            logger.debug(f"Alert does not meet retraining criteria: {alert.description}")
            return
        
        # Check if retraining is already in progress for affected diseases
        for disease in alert.affected_diseases:
            if self._is_retraining_active(disease):
                logger.info(f"Retraining already active for {disease}, skipping")
                return
        
        # Check frequency limits
        for disease in alert.affected_diseases:
            if self._is_too_frequent(disease):
                logger.info(f"Retraining too frequent for {disease}, skipping")
                return
        
        # Trigger retraining
        self._trigger_retraining(alert)
    
    def _should_trigger_retraining(self, alert: DriftAlert) -> bool:
        """Check if alert should trigger retraining."""
        # Check severity
        if alert.severity == "warning" and self.config.min_drift_severity == "critical":
            return False
        
        # Check number of affected diseases
        if len(alert.affected_diseases) < self.config.min_affected_diseases:
            return False
        
        # Check drift persistence (simplified - in practice would check history)
        # For now, assume all alerts are persistent enough
        
        return True
    
    def _is_retraining_active(self, disease: str) -> bool:
        """Check if retraining is active for a disease."""
        for job in self.active_jobs.values():
            if (disease in job.affected_diseases and 
                job.status not in [RetrainingStatus.COMPLETED, RetrainingStatus.FAILED]):
                return True
        return False
    
    def _is_too_frequent(self, disease: str) -> bool:
        """Check if retraining would be too frequent."""
        if disease not in self.last_retraining_time:
            return False
        
        time_since_last = time.time() - self.last_retraining_time[disease]
        min_interval = self.config.max_retraining_frequency_hours * 3600
        
        return time_since_last < min_interval
    
    def _trigger_retraining(self, alert: DriftAlert) -> str:
        """Trigger retraining job."""
        job_id = f"retrain_{int(time.time())}_{hash(str(alert.affected_diseases)) % 10000}"
        
        # Create retraining job
        job = RetrainingJob(
            job_id=job_id,
            trigger_timestamp=alert.timestamp,
            affected_diseases=alert.affected_diseases,
            trigger_alerts=[alert],
            status=RetrainingStatus.TRIGGERED,
            config=self.config
        )
        
        self.active_jobs[job_id] = job
        
        # Update last retraining time
        for disease in alert.affected_diseases:
            self.last_retraining_time[disease] = time.time()
        
        logger.info(f"Triggered retraining job {job_id} for diseases: {alert.affected_diseases}")
        
        # Start job execution asynchronously
        asyncio.create_task(self._execute_retraining_job(job_id))
        
        # Notify callbacks
        self._notify_job_callbacks("triggered", job)
        
        return job_id
    
    async def _execute_retraining_job(self, job_id: str) -> None:
        """Execute retraining job."""
        if job_id not in self.active_jobs:
            logger.error(f"Job {job_id} not found")
            return
        
        job = self.active_jobs[job_id]
        job.start_time = time.time()
        
        try:
            # Phase 1: Data preparation
            await self._prepare_training_data(job)
            
            # Phase 2: Model training
            await self._train_model(job)
            
            # Phase 3: Model validation
            await self._validate_model(job)
            
            # Phase 4: A/B testing
            if self.config.ab_test_duration_hours > 0:
                await self._run_ab_test(job)
            
            # Phase 5: Deployment
            await self._deploy_model(job)
            
            # Mark as completed
            job.status = RetrainingStatus.COMPLETED
            job.end_time = time.time()
            
            logger.info(f"Retraining job {job_id} completed successfully")
            
        except Exception as e:
            job.status = RetrainingStatus.FAILED
            job.error_message = str(e)
            job.end_time = time.time()
            
            logger.error(f"Retraining job {job_id} failed: {e}")
            
            # Attempt rollback if configured
            if self.config.rollback_on_failure:
                await self._rollback_deployment(job)
        
        finally:
            # Move to completed jobs
            self.completed_jobs.append(job)
            del self.active_jobs[job_id]
            
            # Notify callbacks
            self._notify_job_callbacks("completed", job)
    
    async def _prepare_training_data(self, job: RetrainingJob) -> None:
        """Prepare training data for retraining."""
        job.status = RetrainingStatus.PREPARING_DATA
        self._notify_status_callbacks(job)
        
        logger.info(f"Preparing training data for job {job.job_id}")
        
        # Simulate data preparation (in practice, would collect recent data)
        await asyncio.sleep(2)  # Simulate processing time
        
        logger.info(f"Training data prepared for job {job.job_id}")
    
    async def _train_model(self, job: RetrainingJob) -> None:
        """Train new model."""
        job.status = RetrainingStatus.TRAINING
        self._notify_status_callbacks(job)
        
        logger.info(f"Starting model training for job {job.job_id}")
        
        if self.model_trainer:
            # Use provided trainer
            try:
                result = await self._run_async(
                    self.model_trainer,
                    job.affected_diseases,
                    job.config.training_rounds
                )
                job.new_model_path = result.get('model_path')
            except Exception as e:
                raise Exception(f"Model training failed: {e}")
        else:
            # Simulate training
            for round_num in range(1, job.config.training_rounds + 1):
                job.current_round = round_num
                await asyncio.sleep(1)  # Simulate training time
                
                # Simulate early stopping
                if round_num >= 5:  # Simulate convergence
                    break
            
            job.new_model_path = f"models/retrained_{job.job_id}.pth"
        
        logger.info(f"Model training completed for job {job.job_id}")
    
    async def _validate_model(self, job: RetrainingJob) -> None:
        """Validate trained model."""
        job.status = RetrainingStatus.VALIDATING
        self._notify_status_callbacks(job)
        
        logger.info(f"Validating model for job {job.job_id}")
        
        if self.model_validator:
            # Use provided validator
            try:
                result = await self._run_async(
                    self.model_validator,
                    job.new_model_path,
                    job.affected_diseases
                )
                job.validation_metrics = result
            except Exception as e:
                raise Exception(f"Model validation failed: {e}")
        else:
            # Simulate validation
            await asyncio.sleep(3)
            
            # Simulate validation metrics
            job.validation_metrics = {
                disease: {
                    "accuracy": 0.92 + (hash(disease) % 100) / 1000,
                    "precision": 0.90 + (hash(disease) % 100) / 1000,
                    "recall": 0.89 + (hash(disease) % 100) / 1000,
                    "f1_score": 0.895 + (hash(disease) % 100) / 1000
                }
                for disease in job.affected_diseases
            }
        
        # Check if validation meets requirements
        for disease, metrics in job.validation_metrics.items():
            if metrics.get("accuracy", 0) < 0.85:  # Minimum accuracy threshold
                raise Exception(f"Validation failed for {disease}: accuracy too low")
        
        logger.info(f"Model validation passed for job {job.job_id}")
    
    async def _run_ab_test(self, job: RetrainingJob) -> None:
        """Run A/B test comparing old and new models."""
        job.status = RetrainingStatus.AB_TESTING
        self._notify_status_callbacks(job)
        
        logger.info(f"Starting A/B test for job {job.job_id}")
        
        # Simulate A/B test duration
        test_duration = self.config.ab_test_duration_hours * 3600
        start_time = time.time()
        
        while time.time() - start_time < min(test_duration, 10):  # Cap at 10 seconds for demo
            await asyncio.sleep(1)
            
            # Simulate collecting A/B test metrics
            progress = (time.time() - start_time) / min(test_duration, 10)
            
            # Update A/B test results
            job.ab_test_results = {
                "progress": progress,
                "new_model_performance": {
                    disease: 0.91 + progress * 0.02 for disease in job.affected_diseases
                },
                "old_model_performance": {
                    disease: 0.90 for disease in job.affected_diseases
                },
                "statistical_significance": progress > 0.8
            }
        
        # Check A/B test results
        improvement_threshold = self.config.ab_test_success_threshold
        for disease in job.affected_diseases:
            new_perf = job.ab_test_results["new_model_performance"][disease]
            old_perf = job.ab_test_results["old_model_performance"][disease]
            
            if new_perf - old_perf < improvement_threshold:
                raise Exception(f"A/B test failed for {disease}: insufficient improvement")
        
        logger.info(f"A/B test passed for job {job.job_id}")
    
    async def _deploy_model(self, job: RetrainingJob) -> None:
        """Deploy validated model."""
        job.status = RetrainingStatus.DEPLOYING
        self._notify_status_callbacks(job)
        
        logger.info(f"Deploying model for job {job.job_id}")
        
        if self.config.deployment_strategy == "immediate":
            # Immediate deployment
            if self.model_deployer:
                await self._run_async(self.model_deployer, job.new_model_path, 1.0)
            job.deployment_progress = 1.0
        else:
            # Gradual deployment
            for step_progress in self.config.gradual_rollout_steps:
                if self.model_deployer:
                    await self._run_async(self.model_deployer, job.new_model_path, step_progress)
                
                job.deployment_progress = step_progress
                self._notify_status_callbacks(job)
                
                # Wait between rollout steps
                await asyncio.sleep(self.config.rollout_step_duration_hours * 3600 / 10)  # Scaled for demo
        
        logger.info(f"Model deployment completed for job {job.job_id}")
    
    async def _rollback_deployment(self, job: RetrainingJob) -> None:
        """Rollback failed deployment."""
        logger.warning(f"Rolling back deployment for job {job.job_id}")
        
        if self.model_deployer and job.old_model_path:
            try:
                await self._run_async(self.model_deployer, job.old_model_path, 1.0)
                logger.info(f"Rollback completed for job {job.job_id}")
            except Exception as e:
                logger.error(f"Rollback failed for job {job.job_id}: {e}")
    
    async def _run_async(self, func: Callable, *args, **kwargs) -> Any:
        """Run synchronous function asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    def _notify_job_callbacks(self, event: str, job: RetrainingJob) -> None:
        """Notify job event callbacks."""
        for callback in self.job_callbacks:
            try:
                callback(event, job)
            except Exception as e:
                logger.error(f"Error in job callback: {e}")
    
    def _notify_status_callbacks(self, job: RetrainingJob) -> None:
        """Notify status change callbacks."""
        for callback in self.status_callbacks:
            try:
                callback(job)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def add_job_callback(self, callback: Callable) -> None:
        """Add callback for job events."""
        self.job_callbacks.append(callback)
    
    def add_status_callback(self, callback: Callable) -> None:
        """Add callback for status changes."""
        self.status_callbacks.append(callback)
    
    def get_active_jobs(self) -> Dict[str, RetrainingJob]:
        """Get currently active retraining jobs."""
        return self.active_jobs.copy()
    
    def get_job_status(self, job_id: str) -> Optional[RetrainingJob]:
        """Get status of specific job."""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        for job in self.completed_jobs:
            if job.job_id == job_id:
                return job
        
        return None
    
    def get_retraining_history(self, hours: int = 168) -> List[RetrainingJob]:
        """Get retraining history."""
        cutoff_time = time.time() - (hours * 3600)
        
        recent_jobs = []
        
        # Add active jobs
        for job in self.active_jobs.values():
            if job.trigger_timestamp > cutoff_time:
                recent_jobs.append(job)
        
        # Add completed jobs
        for job in self.completed_jobs:
            if job.trigger_timestamp > cutoff_time:
                recent_jobs.append(job)
        
        # Sort by trigger time
        recent_jobs.sort(key=lambda j: j.trigger_timestamp, reverse=True)
        
        return recent_jobs
    
    def export_retraining_report(self, filepath: str) -> None:
        """Export retraining report."""
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "pipeline_status": "running" if self.is_running else "stopped",
            "active_jobs": {
                job_id: asdict(job) for job_id, job in self.active_jobs.items()
            },
            "recent_completed_jobs": [
                asdict(job) for job in self.completed_jobs[-10:]
            ],
            "configuration": asdict(self.config)
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Retraining report exported to {filepath}")
    
    def manual_trigger_retraining(
        self,
        diseases: List[str],
        reason: str = "Manual trigger"
    ) -> str:
        """Manually trigger retraining for specific diseases."""
        # Create synthetic alert
        alert = DriftAlert(
            timestamp=time.time(),
            drift_type="manual",
            severity="critical",
            metric_name="manual_trigger",
            current_value=1.0,
            baseline_value=0.0,
            threshold=0.5,
            description=reason,
            affected_diseases=diseases
        )
        
        return self._trigger_retraining(alert)


if __name__ == "__main__":
    # Demo: Automated retraining pipeline
    
    print("=== Automated Retraining Pipeline Demo ===\n")
    
    # Create drift detector and retraining pipeline
    from .drift_detection import ModelDriftDetector
    
    drift_detector = ModelDriftDetector()
    
    config = RetrainingConfig(
        min_drift_severity="warning",
        training_rounds=5,
        ab_test_duration_hours=0.1,  # 6 minutes for demo
        deployment_strategy="gradual"
    )
    
    pipeline = AutomatedRetrainingPipeline(drift_detector, config)
    
    # Add callbacks
    def job_callback(event: str, job: RetrainingJob):
        print(f"📋 Job {job.job_id}: {event} (status: {job.status.value})")
    
    def status_callback(job: RetrainingJob):
        print(f"🔄 Job {job.job_id}: {job.status.value} (progress: {job.deployment_progress:.1%})")
    
    pipeline.add_job_callback(job_callback)
    pipeline.add_status_callback(status_callback)
    
    # Start monitoring
    pipeline.start_monitoring()
    print("Started automated retraining monitoring")
    
    # Manually trigger retraining
    job_id = pipeline.manual_trigger_retraining(
        diseases=["breast", "lung"],
        reason="Demo: Manual retraining trigger"
    )
    
    print(f"Triggered retraining job: {job_id}")
    
    # Wait for completion
    import asyncio
    
    async def wait_for_completion():
        while job_id in pipeline.active_jobs:
            await asyncio.sleep(1)
        
        # Get final job status
        final_job = pipeline.get_job_status(job_id)
        print(f"\n--- Final Job Status ---")
        print(f"Job ID: {final_job.job_id}")
        print(f"Status: {final_job.status.value}")
        print(f"Duration: {final_job.end_time - final_job.start_time:.1f} seconds")
        print(f"Validation metrics: {final_job.validation_metrics}")
        print(f"Deployment progress: {final_job.deployment_progress:.1%}")
    
    # Run demo
    asyncio.run(wait_for_completion())
    
    # Stop monitoring
    pipeline.stop_monitoring()
    
    print("\n=== Demo Complete ===")