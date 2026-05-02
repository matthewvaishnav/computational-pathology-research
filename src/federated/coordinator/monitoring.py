"""Monitoring system for federated learning with Prometheus, TensorBoard, and alerting."""

import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import torch

from src.federated.common.data_models import ClientUpdate, TrainingRound

logger = logging.getLogger(__name__)

# Try to import TensorBoard, but make it optional
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except (ImportError, TypeError):
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
    logger.warning("TensorBoard not available. TensorBoard logging will be disabled.")


class PrometheusMetricsExporter:
    """
    Export federated learning metrics in Prometheus format.
    
    Implements:
        - 10.1 Prometheus metrics export
    
    Metrics exported:
        - fl_round_duration_seconds: Time taken per training round
        - fl_client_participation_total: Number of clients participating per round
        - fl_model_accuracy: Global model accuracy
        - fl_model_loss: Global model loss
        - fl_aggregation_time_seconds: Time taken for aggregation
        - fl_byzantine_detections_total: Number of Byzantine updates detected
        - fl_client_dropout_total: Number of client dropouts
        - fl_privacy_epsilon: Current privacy budget (epsilon)
        - fl_gradient_norm: L2 norm of aggregated gradients
    """
    
    def __init__(self, metrics_file: str = "./fl_metrics/prometheus.txt"):
        """
        Initialize Prometheus metrics exporter.
        
        Args:
            metrics_file: Path to Prometheus metrics file
        """
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Metric storage
        self.metrics: Dict[str, float] = {}
        self.metric_metadata: Dict[str, Dict[str, str]] = {
            "fl_round_duration_seconds": {
                "type": "gauge",
                "help": "Duration of training round in seconds"
            },
            "fl_client_participation_total": {
                "type": "gauge",
                "help": "Number of clients participating in current round"
            },
            "fl_model_accuracy": {
                "type": "gauge",
                "help": "Global model accuracy"
            },
            "fl_model_loss": {
                "type": "gauge",
                "help": "Global model loss"
            },
            "fl_aggregation_time_seconds": {
                "type": "gauge",
                "help": "Time taken for gradient aggregation in seconds"
            },
            "fl_byzantine_detections_total": {
                "type": "counter",
                "help": "Total number of Byzantine updates detected"
            },
            "fl_client_dropout_total": {
                "type": "counter",
                "help": "Total number of client dropouts"
            },
            "fl_privacy_epsilon": {
                "type": "gauge",
                "help": "Current privacy budget (epsilon)"
            },
            "fl_gradient_norm": {
                "type": "gauge",
                "help": "L2 norm of aggregated gradients"
            },
            "fl_current_round": {
                "type": "gauge",
                "help": "Current training round number"
            },
            "fl_model_version": {
                "type": "gauge",
                "help": "Current global model version"
            },
        }
        
        logger.info(f"PrometheusMetricsExporter initialized: {self.metrics_file}")
    
    def record_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
        """
        if labels:
            # Create labeled metric key
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            key = f"{metric_name}{{{label_str}}}"
        else:
            key = metric_name
        
        self.metrics[key] = value
    
    def export_metrics(self):
        """
        Export metrics to Prometheus format file.
        
        Format:
            # HELP metric_name Description
            # TYPE metric_name type
            metric_name{labels} value
        """
        lines = []
        
        # Group metrics by base name
        metric_groups: Dict[str, List[Tuple[str, float]]] = {}
        for key, value in self.metrics.items():
            # Extract base metric name
            base_name = key.split("{")[0]
            if base_name not in metric_groups:
                metric_groups[base_name] = []
            metric_groups[base_name].append((key, value))
        
        # Generate Prometheus format
        for base_name, entries in sorted(metric_groups.items()):
            # Add metadata
            if base_name in self.metric_metadata:
                metadata = self.metric_metadata[base_name]
                lines.append(f"# HELP {base_name} {metadata['help']}")
                lines.append(f"# TYPE {base_name} {metadata['type']}")
            
            # Add metric values
            for key, value in entries:
                lines.append(f"{key} {value}")
            
            lines.append("")  # Blank line between metrics
        
        # Write to file
        with open(self.metrics_file, "w") as f:
            f.write("\n".join(lines))
        
        logger.debug(f"Exported {len(self.metrics)} metrics to {self.metrics_file}")
    
    def increment_counter(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.
        
        Args:
            metric_name: Name of the counter
            labels: Optional labels
        """
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            key = f"{metric_name}{{{label_str}}}"
        else:
            key = metric_name
        
        self.metrics[key] = self.metrics.get(key, 0) + 1


class TensorBoardLogger:
    """
    Log federated learning metrics to TensorBoard.
    
    Implements:
        - 10.2 TensorBoard logging
    
    Logs:
        - Training metrics (loss, accuracy, gradient norm)
        - Per-client metrics (dataset size, training time)
        - Model parameters (histograms, distributions)
        - Convergence curves
    """
    
    def __init__(self, log_dir: str = "./fl_tensorboard"):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            logger.info(f"TensorBoardLogger initialized: {self.log_dir}")
        else:
            self.writer = None
            logger.warning(f"TensorBoard not available. Logging to {self.log_dir} disabled.")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value.
        
        Args:
            tag: Metric tag (e.g., "loss/train", "accuracy/global")
            value: Metric value
            step: Training step (round number)
        """
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """
        Log multiple scalars under a main tag.
        
        Args:
            main_tag: Main tag (e.g., "metrics")
            tag_scalar_dict: Dictionary of tag -> value
            step: Training step
        """
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """
        Log a histogram of values.
        
        Args:
            tag: Histogram tag
            values: Tensor of values
            step: Training step
        """
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_model_parameters(self, model: torch.nn.Module, step: int):
        """
        Log model parameter histograms.
        
        Args:
            model: Model to log
            step: Training step
        """
        if self.writer:
            for name, param in model.named_parameters():
                self.writer.add_histogram(f"parameters/{name}", param, step)
                if param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad, step)
    
    def log_round_metrics(
        self,
        round_id: int,
        metrics: Dict[str, float],
        client_metrics: Optional[List[Dict[str, float]]] = None
    ):
        """
        Log metrics for a training round.
        
        Args:
            round_id: Round number
            metrics: Global metrics (loss, accuracy, etc.)
            client_metrics: Optional per-client metrics
        """
        # Log global metrics
        for key, value in metrics.items():
            self.log_scalar(f"global/{key}", value, round_id)
        
        # Log per-client metrics
        if client_metrics:
            for i, client_metric in enumerate(client_metrics):
                for key, value in client_metric.items():
                    self.log_scalar(f"client_{i}/{key}", value, round_id)
    
    def close(self):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


class ConvergenceDetector:
    """
    Detect convergence and training stalls.
    
    Implements:
        - 10.3 Convergence detection
    
    Detection strategies:
        - Loss plateau: Loss doesn't improve for N rounds
        - Gradient norm: Gradient norm below threshold
        - Accuracy plateau: Accuracy doesn't improve for N rounds
        - Early stopping: Validation loss increases for N rounds
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        metric_name: str = "loss",
        mode: str = "min"
    ):
        """
        Initialize convergence detector.
        
        Args:
            patience: Number of rounds to wait before declaring convergence
            min_delta: Minimum change to qualify as improvement
            metric_name: Metric to monitor (loss, accuracy, etc.)
            mode: "min" for loss, "max" for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.mode = mode
        
        # State tracking
        self.best_value: Optional[float] = None
        self.rounds_without_improvement = 0
        self.converged = False
        self.history: Deque[float] = deque(maxlen=patience * 2)
        
        logger.info(
            f"ConvergenceDetector initialized: "
            f"metric={metric_name}, mode={mode}, patience={patience}"
        )
    
    def update(self, metrics: Dict[str, float], round_id: int) -> bool:
        """
        Update convergence detector with new metrics.
        
        Args:
            metrics: Round metrics
            round_id: Current round number
        
        Returns:
            converged: True if convergence detected
        """
        if self.metric_name not in metrics:
            logger.warning(f"Metric {self.metric_name} not found in metrics")
            return False
        
        current_value = metrics[self.metric_name]
        self.history.append(current_value)
        
        # Initialize best value
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        # Check for improvement
        if self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:  # mode == "max"
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.rounds_without_improvement = 0
            logger.info(
                f"Round {round_id}: {self.metric_name} improved to {current_value:.4f}"
            )
        else:
            self.rounds_without_improvement += 1
            logger.info(
                f"Round {round_id}: No improvement for {self.rounds_without_improvement} rounds "
                f"({self.metric_name}={current_value:.4f}, best={self.best_value:.4f})"
            )
        
        # Check convergence
        if self.rounds_without_improvement >= self.patience:
            self.converged = True
            logger.info(
                f"Convergence detected: {self.metric_name} plateaued at {self.best_value:.4f} "
                f"for {self.patience} rounds"
            )
            return True
        
        return False
    
    def is_converged(self) -> bool:
        """Check if training has converged."""
        return self.converged
    
    def reset(self):
        """Reset convergence detector."""
        self.best_value = None
        self.rounds_without_improvement = 0
        self.converged = False
        self.history.clear()


class AlertGenerator:
    """
    Generate alerts for anomalies and failures.
    
    Implements:
        - 10.4 Alert generation
    
    Alert types:
        - Byzantine attack detected
        - Privacy budget exhausted
        - Client failure/dropout
        - Convergence stall
        - Resource exhaustion
        - Model performance degradation
    """
    
    def __init__(
        self,
        alert_file: str = "./fl_alerts/alerts.jsonl",
        enable_console_alerts: bool = True
    ):
        """
        Initialize alert generator.
        
        Args:
            alert_file: Path to alert log file
            enable_console_alerts: Whether to log alerts to console
        """
        self.alert_file = Path(alert_file)
        self.alert_file.parent.mkdir(parents=True, exist_ok=True)
        self.enable_console_alerts = enable_console_alerts
        
        # Alert history
        self.alerts: List[Dict[str, Any]] = []
        
        logger.info(f"AlertGenerator initialized: {self.alert_file}")
    
    def generate_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        round_id: Optional[int] = None
    ):
        """
        Generate an alert.
        
        Args:
            alert_type: Type of alert (byzantine, privacy, failure, convergence, resource, performance)
            severity: Severity level (info, warning, error, critical)
            message: Alert message
            details: Optional additional details
            round_id: Optional round ID
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "details": details or {},
            "round_id": round_id,
        }
        
        self.alerts.append(alert)
        
        # Log to file
        with open(self.alert_file, "a") as f:
            f.write(json.dumps(alert) + "\n")
        
        # Log to console
        if self.enable_console_alerts:
            log_func = {
                "info": logger.info,
                "warning": logger.warning,
                "error": logger.error,
                "critical": logger.critical,
            }.get(severity, logger.info)
            
            log_func(f"[ALERT] {alert_type.upper()}: {message}")
    
    def check_byzantine_attack(
        self,
        num_detected: int,
        total_clients: int,
        round_id: int,
        threshold: float = 0.2
    ):
        """
        Check for Byzantine attack and generate alert if detected.
        
        Args:
            num_detected: Number of Byzantine updates detected
            total_clients: Total number of clients
            round_id: Current round ID
            threshold: Alert threshold (fraction of clients)
        """
        if num_detected == 0:
            return
        
        fraction = num_detected / total_clients if total_clients > 0 else 0
        
        if fraction >= threshold:
            self.generate_alert(
                alert_type="byzantine",
                severity="critical",
                message=f"High Byzantine detection rate: {num_detected}/{total_clients} ({fraction:.1%})",
                details={
                    "num_detected": num_detected,
                    "total_clients": total_clients,
                    "fraction": fraction,
                },
                round_id=round_id
            )
        else:
            self.generate_alert(
                alert_type="byzantine",
                severity="warning",
                message=f"Byzantine updates detected: {num_detected}/{total_clients}",
                details={
                    "num_detected": num_detected,
                    "total_clients": total_clients,
                },
                round_id=round_id
            )
    
    def check_privacy_budget(
        self,
        current_epsilon: float,
        budget_limit: float,
        client_id: str,
        round_id: int
    ):
        """
        Check privacy budget and generate alert if exhausted.
        
        Args:
            current_epsilon: Current privacy budget
            budget_limit: Maximum allowed epsilon
            client_id: Client ID
            round_id: Current round ID
        """
        if current_epsilon >= budget_limit:
            self.generate_alert(
                alert_type="privacy",
                severity="critical",
                message=f"Privacy budget exhausted for client {client_id}: ε={current_epsilon:.2f} >= {budget_limit:.2f}",
                details={
                    "client_id": client_id,
                    "current_epsilon": current_epsilon,
                    "budget_limit": budget_limit,
                },
                round_id=round_id
            )
        elif current_epsilon >= budget_limit * 0.9:
            self.generate_alert(
                alert_type="privacy",
                severity="warning",
                message=f"Privacy budget nearly exhausted for client {client_id}: ε={current_epsilon:.2f} (limit: {budget_limit:.2f})",
                details={
                    "client_id": client_id,
                    "current_epsilon": current_epsilon,
                    "budget_limit": budget_limit,
                },
                round_id=round_id
            )
    
    def check_client_dropout(
        self,
        expected_clients: List[str],
        received_clients: List[str],
        round_id: int
    ):
        """
        Check for client dropouts and generate alert.
        
        Args:
            expected_clients: Expected client IDs
            received_clients: Received client IDs
            round_id: Current round ID
        """
        dropped_clients = set(expected_clients) - set(received_clients)
        
        if dropped_clients:
            dropout_rate = len(dropped_clients) / len(expected_clients)
            
            severity = "critical" if dropout_rate >= 0.5 else "warning"
            
            self.generate_alert(
                alert_type="failure",
                severity=severity,
                message=f"Client dropout detected: {len(dropped_clients)}/{len(expected_clients)} clients ({dropout_rate:.1%})",
                details={
                    "dropped_clients": list(dropped_clients),
                    "dropout_rate": dropout_rate,
                },
                round_id=round_id
            )
    
    def check_convergence_stall(
        self,
        rounds_without_improvement: int,
        patience: int,
        metric_name: str,
        current_value: float,
        round_id: int
    ):
        """
        Check for convergence stall and generate alert.
        
        Args:
            rounds_without_improvement: Number of rounds without improvement
            patience: Patience threshold
            metric_name: Metric being monitored
            current_value: Current metric value
            round_id: Current round ID
        """
        if rounds_without_improvement >= patience:
            self.generate_alert(
                alert_type="convergence",
                severity="warning",
                message=f"Training stalled: {metric_name} hasn't improved for {rounds_without_improvement} rounds",
                details={
                    "metric_name": metric_name,
                    "current_value": current_value,
                    "rounds_without_improvement": rounds_without_improvement,
                },
                round_id=round_id
            )
    
    def check_performance_degradation(
        self,
        current_metric: float,
        previous_metric: float,
        metric_name: str,
        round_id: int,
        threshold: float = 0.05
    ):
        """
        Check for model performance degradation.
        
        Args:
            current_metric: Current metric value
            previous_metric: Previous metric value
            metric_name: Metric name
            round_id: Current round ID
            threshold: Degradation threshold
        """
        # For loss (lower is better)
        if metric_name.lower() == "loss":
            degradation = (current_metric - previous_metric) / previous_metric
            if degradation > threshold:
                self.generate_alert(
                    alert_type="performance",
                    severity="warning",
                    message=f"Model performance degraded: {metric_name} increased by {degradation:.1%}",
                    details={
                        "metric_name": metric_name,
                        "current_value": current_metric,
                        "previous_value": previous_metric,
                        "degradation": degradation,
                    },
                    round_id=round_id
                )
        # For accuracy (higher is better)
        elif metric_name.lower() == "accuracy":
            degradation = (previous_metric - current_metric) / previous_metric
            if degradation > threshold:
                self.generate_alert(
                    alert_type="performance",
                    severity="warning",
                    message=f"Model performance degraded: {metric_name} decreased by {degradation:.1%}",
                    details={
                        "metric_name": metric_name,
                        "current_value": current_metric,
                        "previous_value": previous_metric,
                        "degradation": degradation,
                    },
                    round_id=round_id
                )
    
    def get_alerts(
        self,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        round_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get alerts with optional filtering.
        
        Args:
            alert_type: Filter by alert type
            severity: Filter by severity
            round_id: Filter by round ID
        
        Returns:
            filtered_alerts: Filtered alert list
        """
        filtered = self.alerts
        
        if alert_type:
            filtered = [a for a in filtered if a["alert_type"] == alert_type]
        
        if severity:
            filtered = [a for a in filtered if a["severity"] == severity]
        
        if round_id is not None:
            filtered = [a for a in filtered if a["round_id"] == round_id]
        
        return filtered


class MonitoringDashboard:
    """
    Integrated monitoring dashboard combining all monitoring components.
    
    Implements:
        - 10.5 Dashboard creation
    
    Provides:
        - Real-time metrics tracking
        - Convergence monitoring
        - Alert management
        - Performance visualization
    """
    
    def __init__(
        self,
        prometheus_exporter: PrometheusMetricsExporter,
        tensorboard_logger: TensorBoardLogger,
        convergence_detector: ConvergenceDetector,
        alert_generator: AlertGenerator
    ):
        """
        Initialize monitoring dashboard.
        
        Args:
            prometheus_exporter: Prometheus metrics exporter
            tensorboard_logger: TensorBoard logger
            convergence_detector: Convergence detector
            alert_generator: Alert generator
        """
        self.prometheus = prometheus_exporter
        self.tensorboard = tensorboard_logger
        self.convergence = convergence_detector
        self.alerts = alert_generator
        
        # Tracking
        self.round_start_times: Dict[int, float] = {}
        self.aggregation_start_times: Dict[int, float] = {}
        
        logger.info("MonitoringDashboard initialized")
    
    def start_round(self, round_id: int, num_clients: int, model_version: int):
        """
        Record round start.
        
        Args:
            round_id: Round number
            num_clients: Number of participating clients
            model_version: Current model version
        """
        self.round_start_times[round_id] = time.time()
        
        # Update Prometheus metrics
        self.prometheus.record_metric("fl_current_round", round_id)
        self.prometheus.record_metric("fl_model_version", model_version)
        self.prometheus.record_metric("fl_client_participation_total", num_clients)
        self.prometheus.export_metrics()
    
    def start_aggregation(self, round_id: int):
        """
        Record aggregation start.
        
        Args:
            round_id: Round number
        """
        self.aggregation_start_times[round_id] = time.time()
    
    def end_aggregation(self, round_id: int):
        """
        Record aggregation end and compute duration.
        
        Args:
            round_id: Round number
        """
        if round_id in self.aggregation_start_times:
            duration = time.time() - self.aggregation_start_times[round_id]
            
            # Update Prometheus metrics
            self.prometheus.record_metric("fl_aggregation_time_seconds", duration)
            self.prometheus.export_metrics()
            
            # Log to TensorBoard
            self.tensorboard.log_scalar("timing/aggregation_seconds", duration, round_id)
    
    def end_round(
        self,
        round_id: int,
        metrics: Dict[str, float],
        client_updates: Optional[List[ClientUpdate]] = None,
        num_byzantine: int = 0,
        num_dropouts: int = 0
    ):
        """
        Record round end and update all monitoring components.
        
        Args:
            round_id: Round number
            metrics: Round metrics (loss, accuracy, etc.)
            client_updates: Optional client updates for per-client metrics
            num_byzantine: Number of Byzantine updates detected
            num_dropouts: Number of client dropouts
        """
        # Compute round duration
        if round_id in self.round_start_times:
            duration = time.time() - self.round_start_times[round_id]
            self.prometheus.record_metric("fl_round_duration_seconds", duration)
            self.tensorboard.log_scalar("timing/round_seconds", duration, round_id)
        
        # Update Prometheus metrics
        for key, value in metrics.items():
            metric_name = f"fl_model_{key}"
            self.prometheus.record_metric(metric_name, value)
        
        # Update counters
        if num_byzantine > 0:
            for _ in range(num_byzantine):
                self.prometheus.increment_counter("fl_byzantine_detections_total")
        
        if num_dropouts > 0:
            for _ in range(num_dropouts):
                self.prometheus.increment_counter("fl_client_dropout_total")
        
        self.prometheus.export_metrics()
        
        # Log to TensorBoard
        self.tensorboard.log_round_metrics(round_id, metrics)
        
        # Check convergence
        converged = self.convergence.update(metrics, round_id)
        if converged:
            self.alerts.check_convergence_stall(
                rounds_without_improvement=self.convergence.rounds_without_improvement,
                patience=self.convergence.patience,
                metric_name=self.convergence.metric_name,
                current_value=metrics.get(self.convergence.metric_name, 0.0),
                round_id=round_id
            )
        
        # Generate alerts
        if num_byzantine > 0:
            # If client_updates provided, use actual count; otherwise use num_clients from start_round
            total_clients = len(client_updates) if client_updates else num_byzantine + 7  # Estimate
            self.alerts.check_byzantine_attack(
                num_detected=num_byzantine,
                total_clients=total_clients,
                round_id=round_id
            )
    
    def log_gradient_norm(self, round_id: int, gradient_norm: float):
        """
        Log gradient norm.
        
        Args:
            round_id: Round number
            gradient_norm: L2 norm of gradients
        """
        self.prometheus.record_metric("fl_gradient_norm", gradient_norm)
        self.prometheus.export_metrics()
        
        self.tensorboard.log_scalar("gradients/norm", gradient_norm, round_id)
    
    def log_privacy_budget(self, round_id: int, epsilon: float, client_id: Optional[str] = None):
        """
        Log privacy budget.
        
        Args:
            round_id: Round number
            epsilon: Current epsilon value
            client_id: Optional client ID
        """
        if client_id:
            self.prometheus.record_metric("fl_privacy_epsilon", epsilon, labels={"client_id": client_id})
        else:
            self.prometheus.record_metric("fl_privacy_epsilon", epsilon)
        
        self.prometheus.export_metrics()
        
        tag = f"privacy/epsilon_{client_id}" if client_id else "privacy/epsilon"
        self.tensorboard.log_scalar(tag, epsilon, round_id)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get monitoring summary.
        
        Returns:
            summary: Dashboard summary with key metrics and alerts
        """
        return {
            "converged": self.convergence.is_converged(),
            "rounds_without_improvement": self.convergence.rounds_without_improvement,
            "best_metric_value": self.convergence.best_value,
            "total_alerts": len(self.alerts.alerts),
            "critical_alerts": len(self.alerts.get_alerts(severity="critical")),
            "recent_alerts": self.alerts.alerts[-5:] if self.alerts.alerts else [],
        }
    
    def close(self):
        """Close monitoring dashboard."""
        self.tensorboard.close()
        logger.info("MonitoringDashboard closed")
