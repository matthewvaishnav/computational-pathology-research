"""
Unit tests for Metrics Collector component.

Tests metric recording for all metric types, JSON serialization,
confidence interval computation, and validation.

Requirements: 4.1-4.10
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from experiments.benchmark_system.metrics_collector import (
    AggregatedMetrics,
    CollectionSession,
    EpochMetrics,
    MetricsCollector,
    SystemMetrics,
)
from experiments.benchmark_system.resource_manager import ResourceMetrics


class TestMetricsCollector:
    """Test suite for MetricsCollector."""

    @pytest.fixture
    def mock_resource_manager(self):
        """Create mock ResourceManager."""
        manager = Mock()
        manager.monitor_resources.return_value = ResourceMetrics(
            gpu_memory_used_mb=8000.0,
            gpu_memory_total_mb=12000.0,
            gpu_memory_percent=66.7,
            gpu_temperature=70.0,
            gpu_utilization=80.0,
            timestamp=time.time(),
        )
        return manager

    @pytest.fixture
    def collector(self, mock_resource_manager):
        """Create MetricsCollector instance for testing."""
        return MetricsCollector(
            resource_manager=mock_resource_manager,
            confidence_level=0.95,
            bootstrap_samples=100,  # Reduced for faster tests
        )

    def test_init(self, mock_resource_manager):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector(
            resource_manager=mock_resource_manager,
            confidence_level=0.99,
            bootstrap_samples=500,
        )

        assert collector.resource_manager == mock_resource_manager
        assert collector.confidence_level == 0.99
        assert collector.bootstrap_samples == 500
        assert collector.current_session is None

    def test_init_default_resource_manager(self):
        """Test MetricsCollector with default ResourceManager."""
        collector = MetricsCollector()

        assert collector.resource_manager is not None
        assert collector.confidence_level == 0.95
        assert collector.bootstrap_samples == 1000

    def test_start_collection(self, collector):
        """
        Test starting metrics collection session.

        Requirement 4.6: Timestamp synchronization
        """
        metadata = {"task": "test_task", "dataset": "test_dataset"}

        session = collector.start_collection("HistoCore", metadata=metadata)

        assert session.framework_name == "HistoCore"
        assert "HistoCore" in session.session_id
        assert session.start_time > 0
        assert session.metadata == metadata
        assert len(session.epoch_metrics) == 0
        assert len(session.system_metrics) == 0
        assert collector.current_session == session

    def test_start_collection_no_metadata(self, collector):
        """Test starting collection without metadata."""
        session = collector.start_collection("PathML")

        assert session.framework_name == "PathML"
        assert session.metadata == {}

    def test_start_collection_replaces_active_session(self, collector, caplog):
        """Test starting new session while one is active."""
        # Start first session
        session1 = collector.start_collection("HistoCore")

        # Start second session (should finalize first)
        with caplog.at_level("WARNING"):
            session2 = collector.start_collection("PathML")

        assert collector.current_session == session2
        assert session2.framework_name == "PathML"
        assert any("active" in record.message.lower() for record in caplog.records)

    def test_record_epoch_metrics(self, collector):
        """
        Test recording per-epoch training metrics.

        Requirements: 4.1 (per-epoch metrics), 4.3 (timing), 4.4 (efficiency),
                     4.6 (timestamp), 4.10 (validation)
        """
        collector.start_collection("HistoCore")

        collector.record_epoch_metrics(
            epoch=1,
            train_loss=0.5,
            train_accuracy=0.85,
            val_loss=0.6,
            val_accuracy=0.82,
            learning_rate=1e-4,
            epoch_duration_seconds=120.5,
            samples_per_second=150.0,
            train_auc=0.90,
            val_auc=0.88,
            train_f1=0.84,
            val_f1=0.81,
        )

        assert len(collector.current_session.epoch_metrics) == 1

        epoch_metrics = collector.current_session.epoch_metrics[0]
        assert epoch_metrics.epoch == 1
        assert epoch_metrics.train_loss == 0.5
        assert epoch_metrics.train_accuracy == 0.85
        assert epoch_metrics.val_loss == 0.6
        assert epoch_metrics.val_accuracy == 0.82
        assert epoch_metrics.learning_rate == 1e-4
        assert epoch_metrics.epoch_duration_seconds == 120.5
        assert epoch_metrics.samples_per_second == 150.0
        assert epoch_metrics.train_auc == 0.90
        assert epoch_metrics.val_auc == 0.88
        assert epoch_metrics.train_f1 == 0.84
        assert epoch_metrics.val_f1 == 0.81
        assert epoch_metrics.timestamp > 0

    def test_record_epoch_metrics_minimal(self, collector):
        """Test recording epoch metrics without optional fields."""
        collector.start_collection("HistoCore")

        collector.record_epoch_metrics(
            epoch=1,
            train_loss=0.5,
            train_accuracy=0.85,
            val_loss=0.6,
            val_accuracy=0.82,
            learning_rate=1e-4,
            epoch_duration_seconds=120.5,
            samples_per_second=150.0,
        )

        epoch_metrics = collector.current_session.epoch_metrics[0]
        assert epoch_metrics.train_auc is None
        assert epoch_metrics.val_auc is None
        assert epoch_metrics.train_f1 is None
        assert epoch_metrics.val_f1 is None

    def test_record_epoch_metrics_no_session(self, collector):
        """Test recording epoch metrics without active session."""
        with pytest.raises(RuntimeError) as exc_info:
            collector.record_epoch_metrics(
                epoch=1,
                train_loss=0.5,
                train_accuracy=0.85,
                val_loss=0.6,
                val_accuracy=0.82,
                learning_rate=1e-4,
                epoch_duration_seconds=120.5,
                samples_per_second=150.0,
            )

        assert "no active collection session" in str(exc_info.value).lower()

    def test_record_epoch_metrics_multiple_epochs(self, collector):
        """Test recording metrics for multiple epochs."""
        collector.start_collection("HistoCore")

        for epoch in range(1, 4):
            collector.record_epoch_metrics(
                epoch=epoch,
                train_loss=0.5 / epoch,
                train_accuracy=0.7 + (epoch * 0.05),
                val_loss=0.6 / epoch,
                val_accuracy=0.68 + (epoch * 0.05),
                learning_rate=1e-4,
                epoch_duration_seconds=120.0,
                samples_per_second=150.0,
            )

        assert len(collector.current_session.epoch_metrics) == 3
        assert collector.current_session.epoch_metrics[0].epoch == 1
        assert collector.current_session.epoch_metrics[2].epoch == 3

    def test_record_system_metrics(self, collector, mock_resource_manager):
        """
        Test capturing system metrics.

        Requirements: 4.2 (system metrics), 4.6 (timestamp)
        """
        collector.start_collection("HistoCore")

        system_metrics = collector.record_system_metrics()

        assert system_metrics.gpu_memory_used_mb == 8000.0
        assert system_metrics.gpu_memory_total_mb == 12000.0
        assert system_metrics.gpu_memory_percent == 66.7
        assert system_metrics.gpu_temperature == 70.0
        assert system_metrics.gpu_utilization == 80.0
        assert system_metrics.timestamp > 0

        assert len(collector.current_session.system_metrics) == 1
        mock_resource_manager.monitor_resources.assert_called_once()

    def test_record_system_metrics_no_session(self, collector):
        """Test recording system metrics without active session."""
        with pytest.raises(RuntimeError) as exc_info:
            collector.record_system_metrics()

        assert "no active collection session" in str(exc_info.value).lower()

    def test_record_system_metrics_multiple_calls(self, collector, mock_resource_manager):
        """Test recording system metrics multiple times."""
        collector.start_collection("HistoCore")

        # Record system metrics 3 times
        for i in range(3):
            mock_resource_manager.monitor_resources.return_value = ResourceMetrics(
                gpu_memory_used_mb=8000.0 + (i * 100),
                gpu_memory_total_mb=12000.0,
                gpu_memory_percent=66.7 + i,
                gpu_temperature=70.0 + i,
                gpu_utilization=80.0 + i,
                timestamp=time.time(),
            )
            collector.record_system_metrics()

        assert len(collector.current_session.system_metrics) == 3
        assert collector.current_session.system_metrics[0].gpu_temperature == 70.0
        assert collector.current_session.system_metrics[2].gpu_temperature == 72.0

    def test_finalize_collection(self, collector):
        """
        Test finalizing collection and computing aggregated metrics.

        Requirements: 4.5 (statistical metrics), 4.8 (aggregation),
                     4.9 (confidence intervals)
        """
        collector.start_collection("HistoCore")

        # Record some epoch metrics
        for epoch in range(1, 4):
            collector.record_epoch_metrics(
                epoch=epoch,
                train_loss=0.5 - (epoch * 0.05),
                train_accuracy=0.7 + (epoch * 0.05),
                val_loss=0.6 - (epoch * 0.05),
                val_accuracy=0.68 + (epoch * 0.05),
                learning_rate=1e-4,
                epoch_duration_seconds=120.0,
                samples_per_second=150.0,
            )

        # Record some system metrics
        for _ in range(3):
            collector.record_system_metrics()

        aggregated = collector.finalize_collection()

        # Check aggregated metrics
        assert aggregated.mean_train_loss > 0
        assert aggregated.std_train_loss >= 0
        assert aggregated.mean_val_accuracy > 0
        assert aggregated.std_val_accuracy >= 0
        assert aggregated.total_training_time_seconds == 360.0  # 3 epochs * 120s
        assert aggregated.mean_epoch_duration_seconds == 120.0
        assert aggregated.mean_samples_per_second == 150.0
        assert aggregated.peak_gpu_memory_mb > 0
        assert aggregated.mean_gpu_temperature > 0

        # Check confidence intervals
        assert len(aggregated.train_accuracy_ci) == 2
        assert len(aggregated.val_accuracy_ci) == 2
        assert aggregated.train_accuracy_ci[0] <= aggregated.train_accuracy_ci[1]
        assert aggregated.val_accuracy_ci[0] <= aggregated.val_accuracy_ci[1]

        # Session should be cleared
        assert collector.current_session is None

    def test_finalize_collection_with_output_path(self, collector):
        """
        Test finalizing collection and saving to JSON.

        Requirement 4.7: JSON serialization
        """
        collector.start_collection("HistoCore", metadata={"test": "data"})

        # Record metrics
        collector.record_epoch_metrics(
            epoch=1,
            train_loss=0.5,
            train_accuracy=0.85,
            val_loss=0.6,
            val_accuracy=0.82,
            learning_rate=1e-4,
            epoch_duration_seconds=120.0,
            samples_per_second=150.0,
        )
        collector.record_system_metrics()

        # Finalize with output path
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            aggregated = collector.finalize_collection(output_path=output_path)

            # Check file was created
            assert output_path.exists()

            # Load and verify JSON
            with open(output_path) as f:
                data = json.load(f)

            assert data["framework_name"] == "HistoCore"
            assert data["metadata"]["test"] == "data"
            assert len(data["epoch_metrics"]) == 1
            assert len(data["system_metrics"]) == 1
            assert "aggregated_metrics" in data
            assert data["aggregated_metrics"]["mean_train_loss"] == 0.5

    def test_finalize_collection_no_session(self, collector):
        """Test finalizing without active session."""
        with pytest.raises(RuntimeError) as exc_info:
            collector.finalize_collection()

        assert "no active collection session" in str(exc_info.value).lower()

    def test_finalize_collection_no_epoch_metrics(self, collector):
        """Test finalizing with no epoch metrics."""
        collector.start_collection("HistoCore")

        with pytest.raises(ValueError) as exc_info:
            collector.finalize_collection()

        assert "no epoch metrics" in str(exc_info.value).lower()

    def test_compute_confidence_intervals(self, collector):
        """
        Test bootstrap confidence interval computation.

        Requirement 4.9: Bootstrap confidence intervals (95%)
        """
        values = [0.80, 0.82, 0.85, 0.87, 0.90]

        lower, upper = collector.compute_confidence_intervals(values)

        assert lower <= upper
        assert lower >= min(values) * 0.9  # Reasonable lower bound
        assert upper <= max(values) * 1.1  # Reasonable upper bound

    def test_compute_confidence_intervals_single_value(self, collector):
        """Test confidence interval with single value."""
        values = [0.85]

        lower, upper = collector.compute_confidence_intervals(values)

        assert lower == 0.85
        assert upper == 0.85

    def test_compute_confidence_intervals_empty(self, collector):
        """Test confidence interval with empty values."""
        with pytest.raises(ValueError) as exc_info:
            collector.compute_confidence_intervals([])

        assert "empty values" in str(exc_info.value).lower()

    def test_compute_confidence_intervals_custom_params(self, collector):
        """Test confidence interval with custom parameters."""
        values = [0.80, 0.82, 0.85, 0.87, 0.90]

        lower, upper = collector.compute_confidence_intervals(
            values,
            confidence_level=0.99,
            n_bootstrap=50,
        )

        assert lower <= upper

    def test_validate_metrics_valid(self, collector):
        """
        Test metrics validation with valid values.

        Requirement 4.10: Metrics validation
        """
        collector.start_collection("HistoCore")

        # Should not raise exception
        collector.record_epoch_metrics(
            epoch=1,
            train_loss=0.5,
            train_accuracy=0.85,
            val_loss=0.6,
            val_accuracy=0.82,
            learning_rate=1e-4,
            epoch_duration_seconds=120.0,
            samples_per_second=150.0,
        )

    def test_validate_metrics_nan_loss(self, collector):
        """Test validation rejects NaN loss."""
        collector.start_collection("HistoCore")

        with pytest.raises(ValueError) as exc_info:
            collector.record_epoch_metrics(
                epoch=1,
                train_loss=float("nan"),
                train_accuracy=0.85,
                val_loss=0.6,
                val_accuracy=0.82,
                learning_rate=1e-4,
                epoch_duration_seconds=120.0,
                samples_per_second=150.0,
            )

        assert "finite" in str(exc_info.value).lower()

    def test_validate_metrics_inf_accuracy(self, collector):
        """Test validation rejects Inf accuracy."""
        collector.start_collection("HistoCore")

        with pytest.raises(ValueError) as exc_info:
            collector.record_epoch_metrics(
                epoch=1,
                train_loss=0.5,
                train_accuracy=float("inf"),
                val_loss=0.6,
                val_accuracy=0.82,
                learning_rate=1e-4,
                epoch_duration_seconds=120.0,
                samples_per_second=150.0,
            )

        assert "finite" in str(exc_info.value).lower()

    def test_validate_metrics_accuracy_out_of_range_high(self, collector):
        """Test validation rejects accuracy > 1.0."""
        collector.start_collection("HistoCore")

        with pytest.raises(ValueError) as exc_info:
            collector.record_epoch_metrics(
                epoch=1,
                train_loss=0.5,
                train_accuracy=1.5,
                val_loss=0.6,
                val_accuracy=0.82,
                learning_rate=1e-4,
                epoch_duration_seconds=120.0,
                samples_per_second=150.0,
            )

        assert "train_accuracy" in str(exc_info.value).lower()
        assert "[0.0, 1.0]" in str(exc_info.value)

    def test_validate_metrics_accuracy_out_of_range_low(self, collector):
        """Test validation rejects accuracy < 0.0."""
        collector.start_collection("HistoCore")

        with pytest.raises(ValueError) as exc_info:
            collector.record_epoch_metrics(
                epoch=1,
                train_loss=0.5,
                train_accuracy=0.85,
                val_loss=0.6,
                val_accuracy=-0.1,
                learning_rate=1e-4,
                epoch_duration_seconds=120.0,
                samples_per_second=150.0,
            )

        assert "val_accuracy" in str(exc_info.value).lower()

    def test_validate_metrics_negative_loss(self, collector):
        """Test validation rejects negative loss."""
        collector.start_collection("HistoCore")

        with pytest.raises(ValueError) as exc_info:
            collector.record_epoch_metrics(
                epoch=1,
                train_loss=-0.5,
                train_accuracy=0.85,
                val_loss=0.6,
                val_accuracy=0.82,
                learning_rate=1e-4,
                epoch_duration_seconds=120.0,
                samples_per_second=150.0,
            )

        assert "train_loss" in str(exc_info.value).lower()
        assert ">= 0" in str(exc_info.value)

    def test_validate_metrics_zero_learning_rate(self, collector):
        """Test validation rejects zero learning rate."""
        collector.start_collection("HistoCore")

        with pytest.raises(ValueError) as exc_info:
            collector.record_epoch_metrics(
                epoch=1,
                train_loss=0.5,
                train_accuracy=0.85,
                val_loss=0.6,
                val_accuracy=0.82,
                learning_rate=0.0,
                epoch_duration_seconds=120.0,
                samples_per_second=150.0,
            )

        assert "learning_rate" in str(exc_info.value).lower()
        assert "> 0" in str(exc_info.value)

    def test_validate_metrics_negative_duration(self, collector):
        """Test validation rejects negative epoch duration."""
        collector.start_collection("HistoCore")

        with pytest.raises(ValueError) as exc_info:
            collector.record_epoch_metrics(
                epoch=1,
                train_loss=0.5,
                train_accuracy=0.85,
                val_loss=0.6,
                val_accuracy=0.82,
                learning_rate=1e-4,
                epoch_duration_seconds=-10.0,
                samples_per_second=150.0,
            )

        assert "epoch_duration_seconds" in str(exc_info.value).lower()

    def test_validate_metrics_zero_throughput(self, collector):
        """Test validation rejects zero throughput."""
        collector.start_collection("HistoCore")

        with pytest.raises(ValueError) as exc_info:
            collector.record_epoch_metrics(
                epoch=1,
                train_loss=0.5,
                train_accuracy=0.85,
                val_loss=0.6,
                val_accuracy=0.82,
                learning_rate=1e-4,
                epoch_duration_seconds=120.0,
                samples_per_second=0.0,
            )

        assert "samples_per_second" in str(exc_info.value).lower()

    def test_validate_metrics_invalid_auc(self, collector):
        """Test validation rejects invalid AUC values."""
        collector.start_collection("HistoCore")

        with pytest.raises(ValueError) as exc_info:
            collector.record_epoch_metrics(
                epoch=1,
                train_loss=0.5,
                train_accuracy=0.85,
                val_loss=0.6,
                val_accuracy=0.82,
                learning_rate=1e-4,
                epoch_duration_seconds=120.0,
                samples_per_second=150.0,
                train_auc=1.5,
            )

        assert "train_auc" in str(exc_info.value).lower()

    def test_validate_metrics_invalid_f1(self, collector):
        """Test validation rejects invalid F1 values."""
        collector.start_collection("HistoCore")

        with pytest.raises(ValueError) as exc_info:
            collector.record_epoch_metrics(
                epoch=1,
                train_loss=0.5,
                train_accuracy=0.85,
                val_loss=0.6,
                val_accuracy=0.82,
                learning_rate=1e-4,
                epoch_duration_seconds=120.0,
                samples_per_second=150.0,
                val_f1=-0.1,
            )

        assert "val_f1" in str(exc_info.value).lower()

    def test_epoch_metrics_dataclass(self):
        """Test EpochMetrics dataclass."""
        metrics = EpochMetrics(
            epoch=1,
            train_loss=0.5,
            train_accuracy=0.85,
            val_loss=0.6,
            val_accuracy=0.82,
            learning_rate=1e-4,
            epoch_duration_seconds=120.0,
            samples_per_second=150.0,
            timestamp=time.time(),
        )

        assert metrics.epoch == 1
        assert metrics.train_loss == 0.5
        assert metrics.train_auc is None

    def test_system_metrics_dataclass(self):
        """Test SystemMetrics dataclass."""
        metrics = SystemMetrics(
            gpu_memory_used_mb=8000.0,
            gpu_memory_total_mb=12000.0,
            gpu_memory_percent=66.7,
            gpu_temperature=70.0,
            gpu_utilization=80.0,
            timestamp=time.time(),
        )

        assert metrics.gpu_memory_used_mb == 8000.0
        assert metrics.gpu_temperature == 70.0

    def test_aggregated_metrics_dataclass(self):
        """Test AggregatedMetrics dataclass."""
        metrics = AggregatedMetrics(
            mean_train_loss=0.5,
            std_train_loss=0.05,
            mean_val_loss=0.6,
            std_val_loss=0.06,
            mean_train_accuracy=0.85,
            std_train_accuracy=0.02,
            mean_val_accuracy=0.82,
            std_val_accuracy=0.03,
            total_training_time_seconds=360.0,
            mean_epoch_duration_seconds=120.0,
            std_epoch_duration_seconds=5.0,
            mean_samples_per_second=150.0,
            std_samples_per_second=10.0,
            mean_gpu_utilization=80.0,
            std_gpu_utilization=5.0,
            peak_gpu_memory_mb=9000.0,
            mean_gpu_memory_mb=8000.0,
            peak_gpu_temperature=75.0,
            mean_gpu_temperature=70.0,
            train_accuracy_ci=(0.83, 0.87),
            val_accuracy_ci=(0.80, 0.84),
            train_loss_ci=(0.45, 0.55),
            val_loss_ci=(0.54, 0.66),
        )

        assert metrics.mean_train_loss == 0.5
        assert metrics.train_accuracy_ci == (0.83, 0.87)

    def test_collection_session_dataclass(self):
        """Test CollectionSession dataclass."""
        session = CollectionSession(
            framework_name="HistoCore",
            session_id="test_session",
            start_time=time.time(),
        )

        assert session.framework_name == "HistoCore"
        assert len(session.epoch_metrics) == 0
        assert len(session.system_metrics) == 0
        assert session.metadata == {}


class TestMetricsCollectorIntegration:
    """Integration tests for MetricsCollector."""

    def test_full_collection_workflow(self):
        """Test complete metrics collection workflow."""
        collector = MetricsCollector(bootstrap_samples=100)

        # Start collection
        session = collector.start_collection(
            "HistoCore",
            metadata={"dataset": "PatchCamelyon", "model": "resnet18"},
        )

        # Simulate training for 3 epochs
        for epoch in range(1, 4):
            # Record epoch metrics
            collector.record_epoch_metrics(
                epoch=epoch,
                train_loss=0.5 - (epoch * 0.05),
                train_accuracy=0.7 + (epoch * 0.05),
                val_loss=0.6 - (epoch * 0.05),
                val_accuracy=0.68 + (epoch * 0.05),
                learning_rate=1e-4,
                epoch_duration_seconds=120.0,
                samples_per_second=150.0,
            )

            # Record system metrics during epoch
            collector.record_system_metrics()

        # Finalize and save
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            aggregated = collector.finalize_collection(output_path=output_path)

            # Verify aggregated metrics
            assert aggregated.total_training_time_seconds == 360.0
            assert aggregated.mean_epoch_duration_seconds == 120.0
            assert aggregated.mean_samples_per_second == 150.0
            assert aggregated.mean_train_accuracy > 0.7
            assert aggregated.mean_val_accuracy > 0.68

            # Verify JSON file
            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)

            assert data["framework_name"] == "HistoCore"
            assert len(data["epoch_metrics"]) == 3
            assert len(data["system_metrics"]) == 3
            assert data["metadata"]["dataset"] == "PatchCamelyon"

    def test_multiple_sessions_sequential(self):
        """Test running multiple collection sessions sequentially."""
        collector = MetricsCollector(bootstrap_samples=100)

        # First session
        collector.start_collection("HistoCore")
        collector.record_epoch_metrics(
            epoch=1,
            train_loss=0.5,
            train_accuracy=0.85,
            val_loss=0.6,
            val_accuracy=0.82,
            learning_rate=1e-4,
            epoch_duration_seconds=120.0,
            samples_per_second=150.0,
        )
        aggregated1 = collector.finalize_collection()

        # Second session
        collector.start_collection("PathML")
        collector.record_epoch_metrics(
            epoch=1,
            train_loss=0.55,
            train_accuracy=0.80,
            val_loss=0.65,
            val_accuracy=0.78,
            learning_rate=1e-4,
            epoch_duration_seconds=130.0,
            samples_per_second=140.0,
        )
        aggregated2 = collector.finalize_collection()

        # Verify both sessions produced different results
        assert aggregated1.mean_train_accuracy != aggregated2.mean_train_accuracy
        assert aggregated1.mean_epoch_duration_seconds != aggregated2.mean_epoch_duration_seconds
