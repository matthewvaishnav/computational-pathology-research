"""
A/B testing deployment system for federated learning models.

Provides controlled deployment of retrained models with statistical
analysis, safety monitoring, and automated rollback capabilities
for medical AI systems.
"""

import logging
import asyncio
import numpy as np
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from pathlib import Path
from scipy import stats
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ABTestStatus(Enum):
    """Status of A/B test."""
    PREPARING = "preparing"
    RUNNING = "running"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    # Test parameters
    test_duration_hours: float = 48.0
    traffic_split: float = 0.5  # 50% new model, 50% old model
    min_samples_per_group: int = 100
    
    # Statistical parameters
    significance_level: float = 0.05  # 5% significance level
    power: float = 0.8  # 80% statistical power
    min_effect_size: float = 0.02  # 2% minimum improvement
    
    # Safety parameters
    max_performance_degradation: float = 0.05  # 5% max degradation
    safety_check_interval_minutes: float = 30.0
    early_stopping_enabled: bool = True
    
    # Rollback parameters
    auto_rollback_on_failure: bool = True
    rollback_threshold_degradation: float = 0.1  # 10% degradation triggers rollback
    
    # Monitoring parameters
    metrics_collection_interval_seconds: float = 60.0
    statistical_test_interval_minutes: float = 60.0


@dataclass
class ABTestMetrics:
    """Metrics for A/B test group."""
    group_name: str  # "control" or "treatment"
    timestamp: float
    
    # Sample information
    total_samples: int
    positive_samples: int
    negative_samples: int
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    
    # Clinical metrics
    sensitivity: float
    specificity: float
    
    # Confidence metrics
    mean_confidence: float
    confidence_std: float
    
    # Response time metrics
    mean_response_time_ms: float
    p95_response_time_ms: float
    
    # Error metrics
    error_rate: float
    timeout_rate: float


@dataclass
class ABTestResult:
    """Result of A/B test analysis."""
    test_id: str
    analysis_timestamp: float
    
    # Test configuration
    duration_hours: float
    traffic_split: float
    
    # Sample sizes
    control_samples: int
    treatment_samples: int
    
    # Statistical results
    primary_metric: str
    control_mean: float
    treatment_mean: float
    effect_size: float
    relative_improvement: float
    
    # Statistical significance
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    statistical_power: float
    
    # Decision
    recommendation: str  # "deploy", "rollback", "continue_testing"
    confidence_level: str  # "high", "medium", "low"
    
    # Detailed metrics
    control_metrics: ABTestMetrics
    treatment_metrics: ABTestMetrics
    
    # Safety analysis
    safety_passed: bool
    safety_issues: List[str]


@dataclass
class ABTestExperiment:
    """A/B test experiment."""
    test_id: str
    start_time: float
    config: ABTestConfig
    
    # Models
    control_model_path: str
    treatment_model_path: str
    
    # Test state
    status: ABTestStatus
    current_traffic_split: float
    
    # Metrics collection
    control_metrics_history: List[ABTestMetrics]
    treatment_metrics_history: List[ABTestMetrics]
    
    # Results
    final_result: Optional[ABTestResult] = None
    end_time: Optional[float] = None
    
    # Error handling
    error_message: Optional[str] = None
    rollback_reason: Optional[str] = None

    def __post_init__(self):
        if self.control_metrics_history is None:
            self.control_metrics_history = []
        if self.treatment_metrics_history is None:
            self.treatment_metrics_history = []


class ABTestingDeployment:
    """
    A/B testing deployment system for federated learning models.
    
    Manages controlled deployment of new models with statistical analysis,
    safety monitoring, and automated decision making.
    """
    
    def __init__(
        self,
        config: ABTestConfig = None,
        model_router: Optional[Callable] = None,
        metrics_collector: Optional[Callable] = None
    ):
        """
        Initialize A/B testing deployment system.
        
        Args:
            config: A/B testing configuration
            model_router: Function to route traffic between models
            metrics_collector: Function to collect performance metrics
        """
        self.config = config or ABTestConfig()
        self.model_router = model_router
        self.metrics_collector = metrics_collector
        
        # Active experiments
        self.active_experiments: Dict[str, ABTestExperiment] = {}
        self.completed_experiments: List[ABTestExperiment] = []
        
        # Callbacks
        self.status_callbacks: List[Callable] = []
        self.result_callbacks: List[Callable] = []
        
        # Monitoring tasks
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("Initialized A/B testing deployment system")
    
    def start_ab_test(
        self,
        control_model_path: str,
        treatment_model_path: str,
        test_config: Optional[ABTestConfig] = None
    ) -> str:
        """
        Start A/B test between control and treatment models.
        
        Args:
            control_model_path: Path to control (baseline) model
            treatment_model_path: Path to treatment (new) model
            test_config: Optional test-specific configuration
            
        Returns:
            Test ID for tracking
        """
        test_id = f"ab_test_{int(time.time())}_{hash(treatment_model_path) % 10000}"
        config = test_config or self.config
        
        # Create experiment
        experiment = ABTestExperiment(
            test_id=test_id,
            start_time=time.time(),
            config=config,
            control_model_path=control_model_path,
            treatment_model_path=treatment_model_path,
            status=ABTestStatus.PREPARING,
            current_traffic_split=config.traffic_split,
            control_metrics_history=[],
            treatment_metrics_history=[]
        )
        
        self.active_experiments[test_id] = experiment
        
        logger.info(f"Starting A/B test {test_id}: {control_model_path} vs {treatment_model_path}")
        
        # Start experiment execution
        asyncio.create_task(self._run_ab_test(test_id))
        
        return test_id
    
    async def _run_ab_test(self, test_id: str) -> None:
        """Run A/B test experiment."""
        if test_id not in self.active_experiments:
            logger.error(f"Test {test_id} not found")
            return
        
        experiment = self.active_experiments[test_id]
        
        try:
            # Phase 1: Preparation
            await self._prepare_ab_test(experiment)
            
            # Phase 2: Run test
            await self._execute_ab_test(experiment)
            
            # Phase 3: Analysis
            await self._analyze_ab_test(experiment)
            
            # Phase 4: Decision and deployment
            await self._make_deployment_decision(experiment)
            
        except Exception as e:
            experiment.status = ABTestStatus.FAILED
            experiment.error_message = str(e)
            experiment.end_time = time.time()
            
            logger.error(f"A/B test {test_id} failed: {e}")
            
            # Attempt rollback
            if self.config.auto_rollback_on_failure:
                await self._rollback_experiment(experiment, f"Test failed: {e}")
        
        finally:
            # Clean up monitoring
            if test_id in self.monitoring_tasks:
                self.monitoring_tasks[test_id].cancel()
                del self.monitoring_tasks[test_id]
            
            # Move to completed experiments
            self.completed_experiments.append(experiment)
            del self.active_experiments[test_id]
            
            # Notify callbacks
            self._notify_result_callbacks(experiment)
    
    async def _prepare_ab_test(self, experiment: ABTestExperiment) -> None:
        """Prepare A/B test experiment."""
        experiment.status = ABTestStatus.PREPARING
        self._notify_status_callbacks(experiment)
        
        logger.info(f"Preparing A/B test {experiment.test_id}")
        
        # Validate models exist
        if not Path(experiment.control_model_path).exists():
            raise FileNotFoundError(f"Control model not found: {experiment.control_model_path}")
        
        if not Path(experiment.treatment_model_path).exists():
            raise FileNotFoundError(f"Treatment model not found: {experiment.treatment_model_path}")
        
        # Configure traffic routing
        if self.model_router:
            await self._run_async(
                self.model_router,
                experiment.control_model_path,
                experiment.treatment_model_path,
                experiment.current_traffic_split
            )
        
        # Start monitoring
        self.monitoring_tasks[experiment.test_id] = asyncio.create_task(
            self._monitor_ab_test(experiment)
        )
        
        logger.info(f"A/B test {experiment.test_id} prepared")
    
    async def _execute_ab_test(self, experiment: ABTestExperiment) -> None:
        """Execute A/B test."""
        experiment.status = ABTestStatus.RUNNING
        self._notify_status_callbacks(experiment)
        
        logger.info(f"Running A/B test {experiment.test_id}")
        
        # Run for configured duration
        test_duration = experiment.config.test_duration_hours * 3600
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            # Check if test should be stopped early
            if await self._should_stop_early(experiment):
                logger.info(f"Early stopping triggered for test {experiment.test_id}")
                break
            
            # Wait for next check
            await asyncio.sleep(60)  # Check every minute
        
        logger.info(f"A/B test {experiment.test_id} execution completed")
    
    async def _monitor_ab_test(self, experiment: ABTestExperiment) -> None:
        """Monitor A/B test metrics."""
        while experiment.status == ABTestStatus.RUNNING:
            try:
                # Collect metrics for both groups
                control_metrics = await self._collect_group_metrics(
                    experiment, "control"
                )
                treatment_metrics = await self._collect_group_metrics(
                    experiment, "treatment"
                )
                
                # Store metrics
                experiment.control_metrics_history.append(control_metrics)
                experiment.treatment_metrics_history.append(treatment_metrics)
                
                # Check safety conditions
                await self._check_safety_conditions(experiment)
                
                # Perform statistical tests if enough data
                if (len(experiment.control_metrics_history) >= 5 and
                    control_metrics.total_samples >= experiment.config.min_samples_per_group and
                    treatment_metrics.total_samples >= experiment.config.min_samples_per_group):
                    
                    await self._perform_statistical_analysis(experiment)
                
            except Exception as e:
                logger.error(f"Error monitoring A/B test {experiment.test_id}: {e}")
            
            # Wait for next collection
            await asyncio.sleep(experiment.config.metrics_collection_interval_seconds)
    
    async def _collect_group_metrics(
        self,
        experiment: ABTestExperiment,
        group: str
    ) -> ABTestMetrics:
        """Collect metrics for a test group."""
        if self.metrics_collector:
            # Use provided metrics collector
            raw_metrics = await self._run_async(
                self.metrics_collector,
                group,
                experiment.test_id
            )
            
            return ABTestMetrics(
                group_name=group,
                timestamp=time.time(),
                total_samples=raw_metrics.get('total_samples', 0),
                positive_samples=raw_metrics.get('positive_samples', 0),
                negative_samples=raw_metrics.get('negative_samples', 0),
                accuracy=raw_metrics.get('accuracy', 0.0),
                precision=raw_metrics.get('precision', 0.0),
                recall=raw_metrics.get('recall', 0.0),
                f1_score=raw_metrics.get('f1_score', 0.0),
                auc_roc=raw_metrics.get('auc_roc', 0.0),
                sensitivity=raw_metrics.get('sensitivity', 0.0),
                specificity=raw_metrics.get('specificity', 0.0),
                mean_confidence=raw_metrics.get('mean_confidence', 0.0),
                confidence_std=raw_metrics.get('confidence_std', 0.0),
                mean_response_time_ms=raw_metrics.get('mean_response_time_ms', 0.0),
                p95_response_time_ms=raw_metrics.get('p95_response_time_ms', 0.0),
                error_rate=raw_metrics.get('error_rate', 0.0),
                timeout_rate=raw_metrics.get('timeout_rate', 0.0)
            )
        else:
            # Generate synthetic metrics for demo
            np.random.seed(hash(f"{experiment.test_id}_{group}_{time.time()}") % 2**32)
            
            # Simulate different performance for control vs treatment
            base_accuracy = 0.88 if group == "control" else 0.90
            base_samples = 150 + int(time.time() - experiment.start_time) // 60 * 10
            
            return ABTestMetrics(
                group_name=group,
                timestamp=time.time(),
                total_samples=base_samples,
                positive_samples=int(base_samples * 0.3),
                negative_samples=int(base_samples * 0.7),
                accuracy=base_accuracy + np.random.normal(0, 0.02),
                precision=base_accuracy + np.random.normal(0, 0.02),
                recall=base_accuracy + np.random.normal(0, 0.02),
                f1_score=base_accuracy + np.random.normal(0, 0.02),
                auc_roc=base_accuracy + 0.05 + np.random.normal(0, 0.02),
                sensitivity=base_accuracy + np.random.normal(0, 0.02),
                specificity=base_accuracy + np.random.normal(0, 0.02),
                mean_confidence=0.85 + np.random.normal(0, 0.05),
                confidence_std=0.15 + np.random.normal(0, 0.02),
                mean_response_time_ms=200 + np.random.normal(0, 20),
                p95_response_time_ms=400 + np.random.normal(0, 50),
                error_rate=0.01 + np.random.normal(0, 0.005),
                timeout_rate=0.005 + np.random.normal(0, 0.002)
            )
    
    async def _check_safety_conditions(self, experiment: ABTestExperiment) -> None:
        """Check safety conditions during A/B test."""
        if not experiment.control_metrics_history or not experiment.treatment_metrics_history:
            return
        
        latest_control = experiment.control_metrics_history[-1]
        latest_treatment = experiment.treatment_metrics_history[-1]
        
        # Check for significant performance degradation
        accuracy_degradation = latest_control.accuracy - latest_treatment.accuracy
        if accuracy_degradation > experiment.config.rollback_threshold_degradation:
            await self._rollback_experiment(
                experiment,
                f"Significant accuracy degradation: {accuracy_degradation:.3f}"
            )
            return
        
        # Check error rates
        if latest_treatment.error_rate > 0.05:  # 5% error rate threshold
            await self._rollback_experiment(
                experiment,
                f"High error rate in treatment group: {latest_treatment.error_rate:.3f}"
            )
            return
        
        # Check response times
        if latest_treatment.mean_response_time_ms > latest_control.mean_response_time_ms * 2:
            await self._rollback_experiment(
                experiment,
                f"Response time degradation in treatment group"
            )
            return
    
    async def _should_stop_early(self, experiment: ABTestExperiment) -> bool:
        """Check if test should be stopped early."""
        if not experiment.config.early_stopping_enabled:
            return False
        
        if len(experiment.control_metrics_history) < 10:  # Need sufficient data
            return False
        
        # Perform statistical test
        result = await self._perform_statistical_analysis(experiment)
        
        if result and result.is_significant:
            # Check if effect size is large enough
            if abs(result.effect_size) > experiment.config.min_effect_size * 2:
                logger.info(f"Early stopping: significant result with large effect size")
                return True
        
        return False
    
    async def _perform_statistical_analysis(
        self,
        experiment: ABTestExperiment
    ) -> Optional[ABTestResult]:
        """Perform statistical analysis of A/B test."""
        if not experiment.control_metrics_history or not experiment.treatment_metrics_history:
            return None
        
        # Get latest metrics
        control_metrics = experiment.control_metrics_history[-1]
        treatment_metrics = experiment.treatment_metrics_history[-1]
        
        # Primary metric is accuracy
        control_accuracy = control_metrics.accuracy
        treatment_accuracy = treatment_metrics.accuracy
        
        # Calculate effect size and relative improvement
        effect_size = treatment_accuracy - control_accuracy
        relative_improvement = effect_size / control_accuracy if control_accuracy > 0 else 0
        
        # Perform statistical test (simplified)
        # In practice, would use proper statistical tests based on sample sizes
        control_samples = control_metrics.total_samples
        treatment_samples = treatment_metrics.total_samples
        
        # Simulate statistical test
        pooled_std = 0.02  # Assumed standard deviation
        se_diff = pooled_std * np.sqrt(1/control_samples + 1/treatment_samples)
        t_stat = effect_size / se_diff if se_diff > 0 else 0
        
        # Calculate p-value (simplified)
        df = control_samples + treatment_samples - 2
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if df > 0 else 1.0
        
        # Confidence interval
        t_critical = stats.t.ppf(1 - experiment.config.significance_level/2, df) if df > 0 else 1.96
        margin_error = t_critical * se_diff
        confidence_interval = (effect_size - margin_error, effect_size + margin_error)
        
        # Statistical significance
        is_significant = p_value < experiment.config.significance_level
        
        # Statistical power (simplified calculation)
        statistical_power = 0.8 if is_significant else 0.6  # Simplified
        
        # Make recommendation
        if is_significant and effect_size > experiment.config.min_effect_size:
            recommendation = "deploy"
            confidence_level = "high"
        elif is_significant and effect_size < -experiment.config.max_performance_degradation:
            recommendation = "rollback"
            confidence_level = "high"
        elif not is_significant and len(experiment.control_metrics_history) > 20:
            recommendation = "rollback"  # No improvement after sufficient testing
            confidence_level = "medium"
        else:
            recommendation = "continue_testing"
            confidence_level = "low"
        
        # Safety analysis
        safety_passed = True
        safety_issues = []
        
        if treatment_metrics.error_rate > control_metrics.error_rate * 1.5:
            safety_passed = False
            safety_issues.append("Increased error rate in treatment group")
        
        if treatment_metrics.mean_response_time_ms > control_metrics.mean_response_time_ms * 1.3:
            safety_passed = False
            safety_issues.append("Increased response time in treatment group")
        
        return ABTestResult(
            test_id=experiment.test_id,
            analysis_timestamp=time.time(),
            duration_hours=(time.time() - experiment.start_time) / 3600,
            traffic_split=experiment.current_traffic_split,
            control_samples=control_samples,
            treatment_samples=treatment_samples,
            primary_metric="accuracy",
            control_mean=control_accuracy,
            treatment_mean=treatment_accuracy,
            effect_size=effect_size,
            relative_improvement=relative_improvement,
            p_value=p_value,
            confidence_interval=confidence_interval,
            is_significant=is_significant,
            statistical_power=statistical_power,
            recommendation=recommendation,
            confidence_level=confidence_level,
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            safety_passed=safety_passed,
            safety_issues=safety_issues
        )
    
    async def _analyze_ab_test(self, experiment: ABTestExperiment) -> None:
        """Analyze A/B test results."""
        experiment.status = ABTestStatus.ANALYZING
        self._notify_status_callbacks(experiment)
        
        logger.info(f"Analyzing A/B test {experiment.test_id}")
        
        # Perform final statistical analysis
        result = await self._perform_statistical_analysis(experiment)
        experiment.final_result = result
        
        if result:
            logger.info(
                f"A/B test {experiment.test_id} analysis complete: "
                f"effect_size={result.effect_size:.4f}, "
                f"p_value={result.p_value:.4f}, "
                f"recommendation={result.recommendation}"
            )
        
        experiment.status = ABTestStatus.COMPLETED
        experiment.end_time = time.time()
    
    async def _make_deployment_decision(self, experiment: ABTestExperiment) -> None:
        """Make deployment decision based on A/B test results."""
        if not experiment.final_result:
            logger.warning(f"No results available for test {experiment.test_id}")
            return
        
        result = experiment.final_result
        
        if result.recommendation == "deploy":
            logger.info(f"Deploying treatment model for test {experiment.test_id}")
            # In practice, would deploy the treatment model
            
        elif result.recommendation == "rollback":
            logger.info(f"Rolling back to control model for test {experiment.test_id}")
            await self._rollback_experiment(experiment, "Statistical analysis recommends rollback")
            
        else:  # continue_testing
            logger.info(f"Test {experiment.test_id} inconclusive, maintaining current deployment")
    
    async def _rollback_experiment(self, experiment: ABTestExperiment, reason: str) -> None:
        """Rollback experiment to control model."""
        experiment.status = ABTestStatus.ROLLED_BACK
        experiment.rollback_reason = reason
        experiment.end_time = time.time()
        
        logger.warning(f"Rolling back A/B test {experiment.test_id}: {reason}")
        
        # Route all traffic to control model
        if self.model_router:
            await self._run_async(
                self.model_router,
                experiment.control_model_path,
                experiment.control_model_path,  # Both control
                0.0  # 0% treatment traffic
            )
        
        self._notify_status_callbacks(experiment)
    
    async def _run_async(self, func: Callable, *args, **kwargs) -> Any:
        """Run synchronous function asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    def _notify_status_callbacks(self, experiment: ABTestExperiment) -> None:
        """Notify status change callbacks."""
        for callback in self.status_callbacks:
            try:
                callback(experiment)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def _notify_result_callbacks(self, experiment: ABTestExperiment) -> None:
        """Notify result callbacks."""
        for callback in self.result_callbacks:
            try:
                callback(experiment)
            except Exception as e:
                logger.error(f"Error in result callback: {e}")
    
    def add_status_callback(self, callback: Callable) -> None:
        """Add callback for status changes."""
        self.status_callbacks.append(callback)
    
    def add_result_callback(self, callback: Callable) -> None:
        """Add callback for test results."""
        self.result_callbacks.append(callback)
    
    def get_experiment_status(self, test_id: str) -> Optional[ABTestExperiment]:
        """Get status of A/B test experiment."""
        if test_id in self.active_experiments:
            return self.active_experiments[test_id]
        
        for experiment in self.completed_experiments:
            if experiment.test_id == test_id:
                return experiment
        
        return None
    
    def get_active_experiments(self) -> Dict[str, ABTestExperiment]:
        """Get all active experiments."""
        return self.active_experiments.copy()
    
    def stop_experiment(self, test_id: str, reason: str = "Manual stop") -> bool:
        """Stop running experiment."""
        if test_id not in self.active_experiments:
            return False
        
        experiment = self.active_experiments[test_id]
        asyncio.create_task(self._rollback_experiment(experiment, reason))
        
        return True
    
    def export_experiment_report(self, test_id: str, filepath: str) -> None:
        """Export experiment report."""
        experiment = self.get_experiment_status(test_id)
        if not experiment:
            logger.error(f"Experiment {test_id} not found")
            return
        
        report = {
            "test_id": experiment.test_id,
            "start_time": experiment.start_time,
            "end_time": experiment.end_time,
            "status": experiment.status.value,
            "control_model": experiment.control_model_path,
            "treatment_model": experiment.treatment_model_path,
            "config": asdict(experiment.config),
            "final_result": asdict(experiment.final_result) if experiment.final_result else None,
            "rollback_reason": experiment.rollback_reason,
            "metrics_history": {
                "control": [asdict(m) for m in experiment.control_metrics_history],
                "treatment": [asdict(m) for m in experiment.treatment_metrics_history]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Experiment report exported to {filepath}")


if __name__ == "__main__":
    # Demo: A/B testing deployment
    
    print("=== A/B Testing Deployment Demo ===\n")
    
    # Create A/B testing system
    config = ABTestConfig(
        test_duration_hours=0.1,  # 6 minutes for demo
        traffic_split=0.5,
        min_samples_per_group=50,
        early_stopping_enabled=True
    )
    
    ab_system = ABTestingDeployment(config)
    
    # Add callbacks
    def status_callback(experiment: ABTestExperiment):
        print(f"🔄 Test {experiment.test_id}: {experiment.status.value}")
    
    def result_callback(experiment: ABTestExperiment):
        if experiment.final_result:
            result = experiment.final_result
            print(f"📊 Test {experiment.test_id} Results:")
            print(f"   Effect size: {result.effect_size:.4f}")
            print(f"   P-value: {result.p_value:.4f}")
            print(f"   Recommendation: {result.recommendation}")
            print(f"   Safety passed: {result.safety_passed}")
    
    ab_system.add_status_callback(status_callback)
    ab_system.add_result_callback(result_callback)
    
    # Start A/B test
    test_id = ab_system.start_ab_test(
        control_model_path="models/baseline_model.pth",
        treatment_model_path="models/retrained_model.pth"
    )
    
    print(f"Started A/B test: {test_id}")
    
    # Wait for completion
    async def wait_for_completion():
        while test_id in ab_system.active_experiments:
            await asyncio.sleep(1)
        
        # Get final experiment
        final_experiment = ab_system.get_experiment_status(test_id)
        print(f"\n--- Final Results ---")
        print(f"Status: {final_experiment.status.value}")
        print(f"Duration: {(final_experiment.end_time - final_experiment.start_time) / 60:.1f} minutes")
        
        if final_experiment.final_result:
            result = final_experiment.final_result
            print(f"Control accuracy: {result.control_mean:.4f}")
            print(f"Treatment accuracy: {result.treatment_mean:.4f}")
            print(f"Relative improvement: {result.relative_improvement:.2%}")
    
    # Run demo
    asyncio.run(wait_for_completion())
    
    # Export report
    ab_system.export_experiment_report(test_id, f"ab_test_report_{test_id}.json")
    
    print("\n=== Demo Complete ===")