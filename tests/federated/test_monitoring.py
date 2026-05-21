"""Unit tests for federated learning monitoring system."""

import time

import pytest
import torch
import torch.nn as nn

from src.features.federated.pathology_fl.coordinator.monitoring import (
    AlertGenerator,
    ConvergenceDetector,
    MonitoringDashboard,
    PrometheusMetricsExporter,
    TensorBoardLogger,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestPrometheusMetricsExporter:
    """Test Prometheus metrics exporter."""

    def test_initialization(self, tmp_path):
        """Test exporter initialization."""
        metrics_file = tmp_path / "prometheus.txt"
        exporter = PrometheusMetricsExporter(metrics_file=str(metrics_file))

        assert exporter.metrics_file == metrics_file
        assert exporter.metrics_file.parent.exists()
        assert len(exporter.metric_metadata) > 0

    def test_record_metric(self, tmp_path):
        """Test recording a metric."""
        exporter = PrometheusMetricsExporter(metrics_file=str(tmp_path / "prometheus.txt"))

        exporter.record_metric("fl_model_loss", 0.5)
        assert "fl_model_loss" in exporter.metrics
        assert exporter.metrics["fl_model_loss"] == 0.5

    def test_record_metric_with_labels(self, tmp_path):
        """Test recording a metric with labels."""
        exporter = PrometheusMetricsExporter(metrics_file=str(tmp_path / "prometheus.txt"))

        exporter.record_metric("fl_privacy_epsilon", 0.8, labels={"client_id": "client1"})
        assert 'fl_privacy_epsilon{client_id="client1"}' in exporter.metrics

    def test_export_metrics(self, tmp_path):
        """Test exporting metrics to file."""
        metrics_file = tmp_path / "prometheus.txt"
        exporter = PrometheusMetricsExporter(metrics_file=str(metrics_file))

        exporter.record_metric("fl_model_loss", 0.5)
        exporter.record_metric("fl_model_accuracy", 0.95)
        exporter.export_metrics()

        assert metrics_file.exists()
        content = metrics_file.read_text()
        assert "fl_model_loss 0.5" in content
        assert "fl_model_accuracy 0.95" in content
        assert "# HELP" in content
        assert "# TYPE" in content

    def test_increment_counter(self, tmp_path):
        """Test incrementing a counter."""
        exporter = PrometheusMetricsExporter(metrics_file=str(tmp_path / "prometheus.txt"))

        exporter.increment_counter("fl_byzantine_detections_total")
        exporter.increment_counter("fl_byzantine_detections_total")
        exporter.increment_counter("fl_byzantine_detections_total")

        assert exporter.metrics["fl_byzantine_detections_total"] == 3


class TestTensorBoardLogger:
    """Test TensorBoard logger."""

    def test_initialization(self, tmp_path):
        """Test logger initialization."""
        log_dir = tmp_path / "tensorboard"
        logger = TensorBoardLogger(log_dir=str(log_dir))

        assert logger.log_dir == log_dir
        assert logger.log_dir.exists()
        # Writer may be None if TensorBoard is not available

        logger.close()

    def test_log_scalar(self, tmp_path):
        """Test logging a scalar value."""
        logger = TensorBoardLogger(log_dir=str(tmp_path / "tensorboard"))

        # Should not raise
        logger.log_scalar("loss/train", 0.5, step=1)
        logger.log_scalar("accuracy/global", 0.95, step=1)

        logger.close()

    def test_log_scalars(self, tmp_path):
        """Test logging multiple scalars."""
        logger = TensorBoardLogger(log_dir=str(tmp_path / "tensorboard"))

        metrics = {"loss": 0.5, "accuracy": 0.95}
        logger.log_scalars("metrics", metrics, step=1)

        logger.close()

    def test_log_histogram(self, tmp_path):
        """Test logging a histogram."""
        logger = TensorBoardLogger(log_dir=str(tmp_path / "tensorboard"))

        values = torch.randn(100)
        logger.log_histogram("gradients/fc.weight", values, step=1)

        logger.close()

    def test_log_model_parameters(self, tmp_path):
        """Test logging model parameters."""
        logger = TensorBoardLogger(log_dir=str(tmp_path / "tensorboard"))
        model = SimpleModel()

        # Forward pass to create gradients
        x = torch.randn(10, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        logger.log_model_parameters(model, step=1)

        logger.close()

    def test_log_round_metrics(self, tmp_path):
        """Test logging round metrics."""
        logger = TensorBoardLogger(log_dir=str(tmp_path / "tensorboard"))

        metrics = {"loss": 0.5, "accuracy": 0.95}
        client_metrics = [
            {"loss": 0.4, "accuracy": 0.96},
            {"loss": 0.6, "accuracy": 0.94},
        ]

        logger.log_round_metrics(round_id=1, metrics=metrics, client_metrics=client_metrics)

        logger.close()


class TestConvergenceDetector:
    """Test convergence detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = ConvergenceDetector(patience=5, min_delta=0.001, metric_name="loss", mode="min")

        assert detector.patience == 5
        assert detector.min_delta == 0.001
        assert detector.metric_name == "loss"
        assert detector.mode == "min"
        assert detector.best_value is None
        assert detector.rounds_without_improvement == 0
        assert not detector.converged

    def test_update_with_improvement(self):
        """Test update with improvement."""
        detector = ConvergenceDetector(patience=3, min_delta=0.01, metric_name="loss", mode="min")

        # First update
        converged = detector.update({"loss": 1.0}, round_id=1)
        assert not converged
        assert detector.best_value == 1.0
        assert detector.rounds_without_improvement == 0

        # Improvement
        converged = detector.update({"loss": 0.8}, round_id=2)
        assert not converged
        assert detector.best_value == 0.8
        assert detector.rounds_without_improvement == 0

    def test_update_without_improvement(self):
        """Test update without improvement."""
        detector = ConvergenceDetector(patience=3, min_delta=0.01, metric_name="loss", mode="min")

        detector.update({"loss": 1.0}, round_id=1)
        detector.update({"loss": 1.0}, round_id=2)

        assert detector.rounds_without_improvement == 1
        assert not detector.converged

    def test_convergence_detection(self):
        """Test convergence detection."""
        detector = ConvergenceDetector(patience=3, min_delta=0.01, metric_name="loss", mode="min")

        detector.update({"loss": 1.0}, round_id=1)
        detector.update({"loss": 1.0}, round_id=2)
        detector.update({"loss": 1.0}, round_id=3)
        converged = detector.update({"loss": 1.0}, round_id=4)

        assert converged
        assert detector.converged
        assert detector.rounds_without_improvement == 3

    def test_accuracy_mode(self):
        """Test detector in accuracy mode (maximize)."""
        detector = ConvergenceDetector(
            patience=3, min_delta=0.01, metric_name="accuracy", mode="max"
        )

        # First update
        detector.update({"accuracy": 0.8}, round_id=1)
        assert detector.best_value == 0.8

        # Improvement
        detector.update({"accuracy": 0.85}, round_id=2)
        assert detector.best_value == 0.85
        assert detector.rounds_without_improvement == 0

        # No improvement
        detector.update({"accuracy": 0.84}, round_id=3)
        assert detector.rounds_without_improvement == 1

    def test_reset(self):
        """Test resetting detector."""
        detector = ConvergenceDetector(patience=3, min_delta=0.01, metric_name="loss", mode="min")

        detector.update({"loss": 1.0}, round_id=1)
        detector.update({"loss": 1.0}, round_id=2)

        detector.reset()

        assert detector.best_value is None
        assert detector.rounds_without_improvement == 0
        assert not detector.converged
        assert len(detector.history) == 0


class TestAlertGenerator:
    """Test alert generator."""

    def test_initialization(self, tmp_path):
        """Test generator initialization."""
        alert_file = tmp_path / "alerts.jsonl"
        generator = AlertGenerator(alert_file=str(alert_file))

        assert generator.alert_file == alert_file
        assert generator.alert_file.parent.exists()
        assert len(generator.alerts) == 0

    def test_generate_alert(self, tmp_path):
        """Test generating an alert."""
        alert_file = tmp_path / "alerts.jsonl"
        generator = AlertGenerator(alert_file=str(alert_file), enable_console_alerts=False)

        generator.generate_alert(
            alert_type="byzantine",
            severity="warning",
            message="Byzantine update detected",
            details={"client_id": "client1"},
            round_id=1,
        )

        assert len(generator.alerts) == 1
        assert generator.alerts[0]["alert_type"] == "byzantine"
        assert generator.alerts[0]["severity"] == "warning"
        assert alert_file.exists()

    def test_check_byzantine_attack(self, tmp_path):
        """Test Byzantine attack check."""
        generator = AlertGenerator(
            alert_file=str(tmp_path / "alerts.jsonl"), enable_console_alerts=False
        )

        # Low detection rate
        generator.check_byzantine_attack(num_detected=1, total_clients=10, round_id=1)
        assert len(generator.alerts) == 1
        assert generator.alerts[0]["severity"] == "warning"

        # High detection rate
        generator.check_byzantine_attack(num_detected=3, total_clients=10, round_id=2)
        assert len(generator.alerts) == 2
        assert generator.alerts[1]["severity"] == "critical"

    def test_check_privacy_budget(self, tmp_path):
        """Test privacy budget check."""
        generator = AlertGenerator(
            alert_file=str(tmp_path / "alerts.jsonl"), enable_console_alerts=False
        )

        # Budget OK
        generator.check_privacy_budget(
            current_epsilon=0.5, budget_limit=1.0, client_id="client1", round_id=1
        )
        assert len(generator.alerts) == 0

        # Budget warning
        generator.check_privacy_budget(
            current_epsilon=0.95, budget_limit=1.0, client_id="client1", round_id=2
        )
        assert len(generator.alerts) == 1
        assert generator.alerts[0]["severity"] == "warning"

        # Budget exhausted
        generator.check_privacy_budget(
            current_epsilon=1.1, budget_limit=1.0, client_id="client1", round_id=3
        )
        assert len(generator.alerts) == 2
        assert generator.alerts[1]["severity"] == "critical"

    def test_check_client_dropout(self, tmp_path):
        """Test client dropout check."""
        generator = AlertGenerator(
            alert_file=str(tmp_path / "alerts.jsonl"), enable_console_alerts=False
        )

        expected = ["client1", "client2", "client3", "client4"]
        received = ["client1", "client3"]

        generator.check_client_dropout(
            expected_clients=expected, received_clients=received, round_id=1
        )

        assert len(generator.alerts) == 1
        assert generator.alerts[0]["alert_type"] == "failure"
        assert "client2" in generator.alerts[0]["details"]["dropped_clients"]
        assert "client4" in generator.alerts[0]["details"]["dropped_clients"]

    def test_check_convergence_stall(self, tmp_path):
        """Test convergence stall check."""
        generator = AlertGenerator(
            alert_file=str(tmp_path / "alerts.jsonl"), enable_console_alerts=False
        )

        generator.check_convergence_stall(
            rounds_without_improvement=5,
            patience=5,
            metric_name="loss",
            current_value=0.5,
            round_id=10,
        )

        assert len(generator.alerts) == 1
        assert generator.alerts[0]["alert_type"] == "convergence"
        assert generator.alerts[0]["severity"] == "warning"

    def test_check_performance_degradation(self, tmp_path):
        """Test performance degradation check."""
        generator = AlertGenerator(
            alert_file=str(tmp_path / "alerts.jsonl"), enable_console_alerts=False
        )

        # Loss degradation
        generator.check_performance_degradation(
            current_metric=0.6, previous_metric=0.5, metric_name="loss", round_id=1, threshold=0.05
        )

        assert len(generator.alerts) == 1
        assert generator.alerts[0]["alert_type"] == "performance"

        # Accuracy degradation
        generator.check_performance_degradation(
            current_metric=0.85,
            previous_metric=0.95,
            metric_name="accuracy",
            round_id=2,
            threshold=0.05,
        )

        assert len(generator.alerts) == 2

    def test_get_alerts_filtering(self, tmp_path):
        """Test alert filtering."""
        generator = AlertGenerator(
            alert_file=str(tmp_path / "alerts.jsonl"), enable_console_alerts=False
        )

        generator.generate_alert("byzantine", "warning", "Test 1", round_id=1)
        generator.generate_alert("privacy", "critical", "Test 2", round_id=1)
        generator.generate_alert("byzantine", "critical", "Test 3", round_id=2)

        # Filter by type
        byzantine_alerts = generator.get_alerts(alert_type="byzantine")
        assert len(byzantine_alerts) == 2

        # Filter by severity
        critical_alerts = generator.get_alerts(severity="critical")
        assert len(critical_alerts) == 2

        # Filter by round
        round1_alerts = generator.get_alerts(round_id=1)
        assert len(round1_alerts) == 2


class TestMonitoringDashboard:
    """Test monitoring dashboard."""

    @pytest.fixture
    def dashboard(self, tmp_path):
        """Create monitoring dashboard for testing."""
        prometheus = PrometheusMetricsExporter(metrics_file=str(tmp_path / "prometheus.txt"))
        tensorboard = TensorBoardLogger(log_dir=str(tmp_path / "tensorboard"))
        convergence = ConvergenceDetector(
            patience=3, min_delta=0.01, metric_name="loss", mode="min"
        )
        alerts = AlertGenerator(
            alert_file=str(tmp_path / "alerts.jsonl"), enable_console_alerts=False
        )

        dashboard = MonitoringDashboard(prometheus, tensorboard, convergence, alerts)

        yield dashboard

        dashboard.close()

    def test_initialization(self, dashboard):
        """Test dashboard initialization."""
        assert dashboard.prometheus is not None
        assert dashboard.tensorboard is not None
        assert dashboard.convergence is not None
        assert dashboard.alerts is not None

    def test_start_round(self, dashboard):
        """Test starting a round."""
        dashboard.start_round(round_id=1, num_clients=5, model_version=0)

        assert 1 in dashboard.round_start_times
        assert dashboard.prometheus.metrics["fl_current_round"] == 1
        assert dashboard.prometheus.metrics["fl_model_version"] == 0
        assert dashboard.prometheus.metrics["fl_client_participation_total"] == 5

    def test_aggregation_timing(self, dashboard):
        """Test aggregation timing."""
        dashboard.start_aggregation(round_id=1)
        time.sleep(0.1)  # Simulate aggregation
        dashboard.end_aggregation(round_id=1)

        assert "fl_aggregation_time_seconds" in dashboard.prometheus.metrics
        assert dashboard.prometheus.metrics["fl_aggregation_time_seconds"] >= 0.1

    def test_end_round(self, dashboard):
        """Test ending a round."""
        dashboard.start_round(round_id=1, num_clients=5, model_version=0)
        time.sleep(0.1)  # Simulate round

        metrics = {"loss": 0.5, "accuracy": 0.95}
        dashboard.end_round(round_id=1, metrics=metrics)

        assert "fl_round_duration_seconds" in dashboard.prometheus.metrics
        assert dashboard.prometheus.metrics["fl_model_loss"] == 0.5
        assert dashboard.prometheus.metrics["fl_model_accuracy"] == 0.95

    def test_log_gradient_norm(self, dashboard):
        """Test logging gradient norm."""
        dashboard.log_gradient_norm(round_id=1, gradient_norm=2.5)

        assert dashboard.prometheus.metrics["fl_gradient_norm"] == 2.5

    def test_log_privacy_budget(self, dashboard):
        """Test logging privacy budget."""
        dashboard.log_privacy_budget(round_id=1, epsilon=0.8, client_id="client1")

        assert 'fl_privacy_epsilon{client_id="client1"}' in dashboard.prometheus.metrics

    def test_get_summary(self, dashboard):
        """Test getting dashboard summary."""
        dashboard.start_round(round_id=1, num_clients=5, model_version=0)
        dashboard.end_round(round_id=1, metrics={"loss": 0.5})

        summary = dashboard.get_summary()

        assert "converged" in summary
        assert "rounds_without_improvement" in summary
        assert "best_metric_value" in summary
        assert "total_alerts" in summary
        assert "critical_alerts" in summary
        assert "recent_alerts" in summary

    def test_byzantine_detection_alert(self, dashboard):
        """Test Byzantine detection generates alert."""
        dashboard.start_round(round_id=1, num_clients=10, model_version=0)
        dashboard.end_round(round_id=1, metrics={"loss": 0.5}, num_byzantine=3)

        alerts = dashboard.alerts.get_alerts(alert_type="byzantine")
        # Should generate alert when Byzantine updates detected
        assert len(alerts) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
