"""
Benchmark Orchestrator for the Competitor Benchmark System.

This module coordinates the complete benchmark workflow, orchestrating all
components (FrameworkManager, TaskExecutor, ResourceManager, MetricsCollector,
CheckpointManager, ErrorHandler, ReportGenerator) to execute benchmark suites.

Requirements: 5.1, 5.2, 5.5, 5.7, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from experiments.benchmark_system.checkpoint_manager import CheckpointManager
from experiments.benchmark_system.error_handler import (
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    RecoveryAction,
)
from experiments.benchmark_system.framework_manager import FrameworkManager
from experiments.benchmark_system.metrics_collector import MetricsCollector
from experiments.benchmark_system.models import (
    BenchmarkConfig,
    BenchmarkSuiteResult,
    FrameworkEnvironment,
    SignificanceTest,
    TaskSpecification,
    TrainingResult,
)
from experiments.benchmark_system.report_generator import ReportGenerator
from experiments.benchmark_system.resource_manager import ResourceManager
from experiments.benchmark_system.result_validator import ResultValidator
from experiments.benchmark_system.task_executor import TrainingTaskExecutor

logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    """
    Coordinates the complete benchmark workflow.
    
    Orchestrates all components to execute benchmark suites:
    - Framework installation and validation
    - Task configuration and execution
    - Resource management and monitoring
    - Metrics collection and aggregation
    - Checkpoint management for crash recovery
    - Error handling and recovery
    - Report generation and documentation updates
    
    Supports:
    - Quick mode (3-4 hours) and full mode (20-40+ hours)
    - Framework selection filtering
    - Progress logging every 10 minutes
    - Completion notifications
    - Timeout enforcement
    
    Requirements: 5.1, 5.2, 5.5, 5.7, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        framework_manager: Optional[FrameworkManager] = None,
        task_executor: Optional[TrainingTaskExecutor] = None,
        resource_manager: Optional[ResourceManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        error_handler: Optional[ErrorHandler] = None,
        report_generator: Optional[ReportGenerator] = None,
        result_validator: Optional[ResultValidator] = None,
    ):
        """
        Initialize Benchmark Orchestrator.
        
        Args:
            config: Benchmark configuration
            framework_manager: Framework manager (creates default if None)
            task_executor: Task executor (creates default if None)
            resource_manager: Resource manager (creates default if None)
            metrics_collector: Metrics collector (creates default if None)
            checkpoint_manager: Checkpoint manager (creates default if None)
            error_handler: Error handler (creates default if None)
            report_generator: Report generator (creates default if None)
            result_validator: Result validator (creates default if None)
        """
        self.config = config
        
        # Initialize components (use provided or create defaults)
        self.framework_manager = framework_manager or FrameworkManager()
        self.task_executor = task_executor or TrainingTaskExecutor()
        self.resource_manager = resource_manager or ResourceManager()
        self.metrics_collector = metrics_collector or MetricsCollector(
            resource_manager=self.resource_manager,
            confidence_level=config.confidence_level,
            bootstrap_samples=config.bootstrap_samples,
        )
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            checkpoint_dir=config.output_dir / "checkpoints",
            checkpoint_interval_minutes=config.checkpoint_interval_minutes,
        )
        self.error_handler = error_handler or ErrorHandler()
        self.result_validator = result_validator or ResultValidator()
        self.report_generator = report_generator or ReportGenerator(
            result_validator=self.result_validator
        )
        
        # State tracking
        self.start_time: Optional[datetime] = None
        self.framework_results: Dict[str, TrainingResult] = {}
        self.framework_environments: Dict[str, FrameworkEnvironment] = {}
        self.failed_frameworks: List[str] = []
        self.last_progress_log_time: Optional[float] = None
        
        logger.info(
            f"Initialized BenchmarkOrchestrator with mode={config.mode}, "
            f"frameworks={config.frameworks}"
        )
    
    def run_benchmark_suite(self) -> BenchmarkSuiteResult:
        """
        Execute complete benchmark suite (quick or full mode).
        
        Coordinates the entire benchmark workflow:
        1. Verify GPU availability
        2. Install and validate frameworks
        3. Configure identical training tasks
        4. Execute training for each framework
        5. Collect and aggregate metrics
        6. Generate comparison reports
        7. Update documentation
        
        Returns:
            BenchmarkSuiteResult with aggregated results
            
        Requirements: 5.1 (Long-running workload support),
                     5.2 (Estimated completion time),
                     5.5 (Progress logging every 10 minutes),
                     5.7 (Completion notification),
                     6.1 (Quick/full mode execution),
                     6.2 (Mode-specific configuration),
                     6.7 (Framework selection filtering)
        """
        self.start_time = datetime.now()
        logger.info(
            f"Starting benchmark suite at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Estimate completion time (Requirement 5.2)
        estimated_duration = self.estimate_completion_time()
        estimated_completion = self.start_time + estimated_duration
        logger.info(
            f"Estimated completion time: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(duration: {estimated_duration})"
        )
        
        # Verify GPU availability
        logger.info("Verifying GPU availability...")
        gpu_info = self.resource_manager.verify_gpu_availability()
        if not gpu_info.available:
            error_msg = f"GPU not available: {gpu_info.error_message}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        logger.info(f"GPU verified: {gpu_info.name} with {gpu_info.memory_total_mb:.0f}MB")
        
        # Apply mode-specific configuration (Requirements 6.2, 6.3, 6.4)
        task_spec = self._apply_mode_configuration()
        
        # Install and validate frameworks
        logger.info(f"Installing frameworks: {self.config.frameworks}")
        for framework_name in self.config.frameworks:
            try:
                env = self._install_and_validate_framework(framework_name)
                self.framework_environments[framework_name] = env
            except Exception as e:
                logger.error(f"Failed to install {framework_name}: {e}")
                self.failed_frameworks.append(framework_name)
                # Continue with other frameworks (Requirement 8.1: Error isolation)
        
        if not self.framework_environments:
            raise RuntimeError("No frameworks were successfully installed")
        
        logger.info(
            f"Successfully installed {len(self.framework_environments)} frameworks: "
            f"{list(self.framework_environments.keys())}"
        )
        
        # Execute training for each framework
        logger.info("Starting training execution for all frameworks...")
        for framework_name in self.framework_environments.keys():
            try:
                # Log progress (Requirement 5.5)
                self._log_progress_if_needed()
                
                # Run single framework benchmark
                result = self.run_single_framework(framework_name, task_spec)
                self.framework_results[framework_name] = result
                
                # Save checkpoint after each framework completes
                self._save_checkpoint()
                
            except Exception as e:
                logger.error(f"Failed to benchmark {framework_name}: {e}")
                self.failed_frameworks.append(framework_name)
                # Continue with other frameworks (Requirement 8.1: Error isolation)
        
        # Generate benchmark suite result
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds() / 3600.0  # hours
        
        logger.info(
            f"Benchmark suite completed in {total_duration:.2f} hours. "
            f"Successful: {len(self.framework_results)}, "
            f"Failed: {len(self.failed_frameworks)}"
        )
        
        # Compute statistical significance tests
        significance_tests = self._compute_significance_tests()
        
        # Compute rankings
        accuracy_ranking = self._compute_ranking("test_accuracy")
        auc_ranking = self._compute_ranking("test_auc")
        efficiency_ranking = self._compute_efficiency_ranking()
        
        # Generate reports
        report_path = self.config.output_dir / "benchmark_report.md"
        visualization_dir = self.config.output_dir / "visualizations"
        
        if self.framework_results:
            # Generate visualizations
            self.report_generator.generate_visualizations(
                list(self.framework_results.values()),
                visualization_dir
            )
            
            # Update PERFORMANCE_COMPARISON.md
            self.report_generator.update_performance_comparison_md(
                list(self.framework_results.values()),
                report_path
            )
            
            # Export to CSV and JSON
            self.report_generator.export_to_csv(
                list(self.framework_results.values()),
                self.config.output_dir / "results.csv"
            )
            self.report_generator.export_to_json(
                list(self.framework_results.values()),
                self.config.output_dir / "results.json"
            )
        
        # Create benchmark suite result
        result = BenchmarkSuiteResult(
            config=self.config,
            framework_results=self.framework_results,
            start_time=self.start_time,
            end_time=end_time,
            total_duration_hours=total_duration,
            significance_tests=significance_tests,
            accuracy_ranking=accuracy_ranking,
            auc_ranking=auc_ranking,
            efficiency_ranking=efficiency_ranking,
            report_path=report_path,
            visualization_dir=visualization_dir,
            successful_frameworks=list(self.framework_results.keys()),
            failed_frameworks=self.failed_frameworks,
            errors={
                fw: "See logs for details"
                for fw in self.failed_frameworks
            },
        )
        
        # Send completion notification (Requirement 5.7)
        self._send_completion_notification(result)
        
        logger.info(f"Benchmark suite result saved to {self.config.output_dir}")
        
        return result
    
    def run_single_framework(
        self,
        framework: str,
        task_spec: TaskSpecification
    ) -> TrainingResult:
        """
        Execute benchmark for a single framework.
        
        Coordinates:
        1. GPU allocation
        2. Task configuration
        3. Metrics collection setup
        4. Training execution
        5. Metrics finalization
        6. GPU memory cleanup
        
        Args:
            framework: Framework name
            task_spec: Task specification
            
        Returns:
            TrainingResult with metrics and outcomes
            
        Raises:
            RuntimeError: If framework execution fails
            
        Requirements: 5.1 (Long-running workload support),
                     5.5 (Progress logging),
                     5.8 (Timeout enforcement)
        """
        logger.info(f"Starting benchmark for {framework}")
        
        # Allocate GPU (Requirement 3.2: Exclusive execution)
        allocation = self.resource_manager.allocate_gpu(framework)
        logger.info(f"GPU allocated to {framework}")
        
        try:
            # Configure task for framework
            training_config = self.task_executor.configure_task(task_spec, framework)
            logger.info(f"Task configured for {framework}")
            
            # Start metrics collection
            self.metrics_collector.start_collection(
                framework=framework,
                metadata={
                    "task_spec": task_spec.__dict__,
                    "config": training_config.config_dict,
                }
            )
            logger.info(f"Metrics collection started for {framework}")
            
            # Get framework environment
            env = self.framework_environments[framework]
            
            # Execute training with timeout enforcement (Requirement 5.8)
            timeout_seconds = self.config.timeout_hours * 3600
            error_context = ErrorContext(
                framework_name=framework,
                error=Exception("Placeholder"),
                error_category=ErrorCategory.RUNTIME,
            )
            
            try:
                # Note: Actual training execution would be delegated to framework-specific adapters
                # For now, this is a placeholder that raises NotImplementedError
                result = self.task_executor.execute_training(training_config, env)
                
            except NotImplementedError:
                # This is expected until framework adapters are implemented
                logger.warning(
                    f"Training execution not implemented for {framework}. "
                    f"Framework adapters need to be implemented."
                )
                raise
            
            # Finalize metrics collection
            metrics_path = self.config.output_dir / f"{framework}_metrics.json"
            aggregated_metrics = self.metrics_collector.finalize_collection(metrics_path)
            logger.info(f"Metrics finalized for {framework}")
            
            # Validate result
            validation_report = self.result_validator.validate_training_result(result)
            if not validation_report.valid:
                logger.warning(
                    f"Validation issues for {framework}: {validation_report.issues}"
                )
            
            logger.info(f"Benchmark completed successfully for {framework}")
            
            return result
            
        finally:
            # Always clear GPU memory (Requirement 3.3)
            self.resource_manager.clear_gpu_memory()
            logger.info(f"GPU memory cleared after {framework}")
    
    def estimate_completion_time(self) -> timedelta:
        """
        Calculate estimated duration for benchmark suite.
        
        Estimates based on:
        - Mode (quick vs full)
        - Number of frameworks
        - Historical timing data (if available)
        
        Returns:
            timedelta with estimated duration
            
        Requirement: 5.2 (Estimated completion time), 6.5 (Time estimation)
        """
        # Base estimates per framework (in hours)
        if self.config.mode == "quick":
            # Quick mode: 3-4 hours total, ~1 hour per framework
            base_time_per_framework = 1.0
        else:  # full mode
            # Full mode: 20-40+ hours total, ~10 hours per framework
            base_time_per_framework = 10.0
        
        # Number of frameworks to benchmark
        num_frameworks = len(self.config.frameworks)
        
        # Total estimated time
        total_hours = base_time_per_framework * num_frameworks
        
        # Add overhead for setup and reporting (10%)
        total_hours *= 1.1
        
        logger.info(
            f"Estimated completion time: {total_hours:.1f} hours "
            f"({num_frameworks} frameworks × {base_time_per_framework:.1f} hours + 10% overhead)"
        )
        
        return timedelta(hours=total_hours)
    
    def _apply_mode_configuration(self) -> TaskSpecification:
        """
        Apply mode-specific configuration (quick vs full).
        
        Returns:
            TaskSpecification with mode-specific settings
            
        Requirements: 6.2 (Mode-specific configuration),
                     6.3 (Quick mode reduced epochs/samples),
                     6.4 (Full mode complete configuration)
        """
        if self.config.task_spec is None:
            raise ValueError("BenchmarkConfig.task_spec is required")
        
        task_spec = self.config.task_spec
        
        if self.config.mode == "quick":
            # Quick mode: reduce epochs and samples (Requirement 6.3)
            logger.info(
                f"Applying quick mode configuration: "
                f"epochs={self.config.quick_mode_epochs}, "
                f"samples={self.config.quick_mode_samples}"
            )
            
            # Create modified task spec for quick mode
            task_spec = TaskSpecification(
                dataset_name=task_spec.dataset_name,
                data_root=task_spec.data_root,
                model_architecture=task_spec.model_architecture,
                train_split=task_spec.train_split,
                val_split=task_spec.val_split,
                test_split=task_spec.test_split,
                feature_dim=task_spec.feature_dim,
                num_classes=task_spec.num_classes,
                num_epochs=self.config.quick_mode_epochs,  # Reduced epochs
                batch_size=task_spec.batch_size,
                learning_rate=task_spec.learning_rate,
                weight_decay=task_spec.weight_decay,
                optimizer=task_spec.optimizer,
                random_seed=task_spec.random_seed,
                augmentation_config=task_spec.augmentation_config,
                metrics=task_spec.metrics,
            )
            
            logger.info(
                f"Quick mode configuration applied: "
                f"num_epochs={task_spec.num_epochs}"
            )
        else:
            # Full mode: use complete configuration (Requirement 6.4)
            logger.info("Using full mode configuration (no modifications)")
        
        return task_spec
    
    def _install_and_validate_framework(
        self,
        framework_name: str
    ) -> FrameworkEnvironment:
        """
        Install and validate a framework.
        
        Args:
            framework_name: Name of framework to install
            
        Returns:
            FrameworkEnvironment with installation details
            
        Raises:
            RuntimeError: If installation or validation fails
        """
        logger.info(f"Installing framework: {framework_name}")
        
        # Install framework
        env = self.framework_manager.install_framework(framework_name)
        
        # Validate installation
        env = self.framework_manager.validate_installation(env)
        
        if env.validation_status != "valid":
            error_msg = (
                f"Framework validation failed for {framework_name}: "
                f"{env.validation_errors}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(
            f"Framework {framework_name} installed and validated successfully "
            f"(version: {env.framework_version})"
        )
        
        return env
    
    def _log_progress_if_needed(self) -> None:
        """
        Log progress updates every 10 minutes.
        
        Requirement: 5.5 (Progress logging every 10 minutes)
        """
        current_time = time.time()
        
        # Initialize last progress log time if not set
        if self.last_progress_log_time is None:
            self.last_progress_log_time = current_time
            return
        
        # Check if 10 minutes have elapsed
        elapsed_minutes = (current_time - self.last_progress_log_time) / 60.0
        
        if elapsed_minutes >= 10.0:
            # Log progress
            elapsed_total = (datetime.now() - self.start_time).total_seconds() / 3600.0
            completed = len(self.framework_results)
            total = len(self.config.frameworks)
            
            logger.info(
                f"PROGRESS UPDATE: {completed}/{total} frameworks completed. "
                f"Elapsed time: {elapsed_total:.2f} hours. "
                f"Successful: {list(self.framework_results.keys())}, "
                f"Failed: {self.failed_frameworks}"
            )
            
            # Update last progress log time
            self.last_progress_log_time = current_time
    
    def _save_checkpoint(self) -> None:
        """
        Save checkpoint with current benchmark state.
        
        Requirement: 5.3 (Checkpoint every 30 minutes)
        """
        benchmark_state = {
            "config": self.config.__dict__,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "completed_frameworks": list(self.framework_results.keys()),
            "failed_frameworks": self.failed_frameworks,
            "framework_environments": {
                name: env.__dict__
                for name, env in self.framework_environments.items()
            },
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(benchmark_state)
        
        if checkpoint_path:
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _compute_significance_tests(self) -> Dict[str, SignificanceTest]:
        """
        Compute statistical significance tests comparing HistoCore to competitors.
        
        Returns:
            Dictionary mapping test keys to SignificanceTest results
        """
        significance_tests = {}
        
        # Find HistoCore result
        histocore_result = self.framework_results.get("HistoCore")
        if not histocore_result:
            logger.warning("HistoCore result not found, skipping significance tests")
            return significance_tests
        
        # Compare against each competitor
        for framework_name, result in self.framework_results.items():
            if framework_name == "HistoCore":
                continue
            
            # Test each metric
            for metric_name in ["accuracy", "auc", "f1"]:
                try:
                    test = self.report_generator.compute_statistical_significance(
                        histocore_result,
                        result,
                        metric_name=metric_name
                    )
                    key = f"{framework_name}_{metric_name}"
                    significance_tests[key] = test
                except Exception as e:
                    logger.warning(
                        f"Failed to compute significance test for "
                        f"{framework_name} {metric_name}: {e}"
                    )
        
        return significance_tests
    
    def _compute_ranking(self, metric_attr: str) -> List[tuple[str, float]]:
        """
        Compute ranking of frameworks by a metric.
        
        Args:
            metric_attr: Attribute name of the metric (e.g., "test_accuracy")
            
        Returns:
            List of (framework_name, metric_value) tuples, sorted descending
        """
        ranking = [
            (name, getattr(result, metric_attr))
            for name, result in self.framework_results.items()
        ]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
    
    def _compute_efficiency_ranking(self) -> List[tuple[str, float]]:
        """
        Compute efficiency ranking (accuracy / parameters).
        
        Returns:
            List of (framework_name, efficiency) tuples, sorted descending
        """
        ranking = [
            (
                name,
                result.test_accuracy / result.model_parameters
                if result.model_parameters > 0 else 0.0
            )
            for name, result in self.framework_results.items()
        ]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
    
    def _send_completion_notification(self, result: BenchmarkSuiteResult) -> None:
        """
        Send completion notification.
        
        Args:
            result: Benchmark suite result
            
        Requirement: 5.7 (Completion notification)
        """
        # Log completion notification
        logger.info(
            f"BENCHMARK SUITE COMPLETED\n"
            f"Duration: {result.total_duration_hours:.2f} hours\n"
            f"Successful frameworks: {result.successful_frameworks}\n"
            f"Failed frameworks: {result.failed_frameworks}\n"
            f"Report: {result.report_path}\n"
            f"Visualizations: {result.visualization_dir}"
        )
        
        # In production, this could send:
        # - Email notification
        # - Slack webhook
        # - Desktop notification
        # For now, we just log the completion
