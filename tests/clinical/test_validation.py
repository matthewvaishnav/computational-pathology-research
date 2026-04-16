"""
Unit tests for model validation and monitoring infrastructure.

Tests cover:
- ModelValidator class functionality
- Bootstrap confidence interval calculation
- Performance tracking and drift detection
- Subpopulation and taxonomy validation
- Alert system and retraining recommendations
- PerformanceMonitor class
"""

import json
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.clinical.classifier import MultiClassDiseaseClassifier
from src.clinical.validation import ModelValidator, PerformanceMonitor


class TestModelValidator(unittest.TestCase):
    """Test cases for ModelValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ModelValidator(
            accuracy_threshold=0.90,
            auc_threshold=0.95,
            max_history_length=10,
            drift_detection_window=3,
            performance_degradation_threshold=0.05,
        )

        # Create mock model
        self.mock_model = MagicMock(spec=MultiClassDiseaseClassifier)
        self.mock_model.parameters.return_value = iter([torch.tensor([1.0])])

        # Create mock taxonomy
        self.mock_taxonomy = MagicMock()
        self.mock_taxonomy.get_taxonomy_info.return_value = {
            "primary": {"benign": [0], "malignant": [1]},
            "secondary": {"low_grade": [0, 1], "high_grade": [2, 3]},
        }

        self.mock_model.taxonomy = self.mock_taxonomy

        # Create sample data
        self.sample_size = 100
        self.num_classes = 2

        # Create synthetic dataset
        features = torch.randn(self.sample_size, 10, 1024)  # WSI features
        labels = torch.randint(0, self.num_classes, (self.sample_size,))
        metadata = [
            {
                "age": np.random.randint(20, 80),
                "sex": np.random.choice(["M", "F"]),
                "smoking_status": np.random.choice(["never", "former", "current"]),
            }
            for _ in range(self.sample_size)
        ]

        # Create dataset and dataloader
        dataset = TensorDataset(features, labels)
        self.data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

        # Mock the dataloader to include metadata
        self.mock_data_loader = MagicMock()
        self.mock_data_loader.__iter__ = lambda self: iter(
            [
                {
                    "wsi_features": features[i: i + 16],
                    "labels": labels[i: i + 16],
                    "metadata": metadata[i: i + 16],
                }
                for i in range(0, len(features), 16)
            ]
        )

    def test_initialization(self):
        """Test ModelValidator initialization."""
        validator = ModelValidator(
            accuracy_threshold=0.85, auc_threshold=0.90, max_history_length=50
        )

        self.assertEqual(validator.accuracy_threshold, 0.85)
        self.assertEqual(validator.auc_threshold, 0.90)
        self.assertEqual(validator.max_history_length, 50)
        self.assertEqual(len(validator.performance_history), 0)
        self.assertEqual(len(validator.alert_callbacks), 0)

    def test_validate_model_basic(self):
        """Test basic model validation functionality."""

        # Mock model forward pass
        def mock_forward(batch):
            batch_size = batch["labels"].shape[0]
            # Return logits that will give reasonable accuracy
            logits = torch.randn(batch_size, self.num_classes)
            # Bias towards correct predictions for higher accuracy
            for i in range(batch_size):
                logits[i, batch["labels"][i]] += 2.0
            return logits

        self.mock_model.side_effect = mock_forward
        self.mock_model.eval = MagicMock()

        # Run validation
        results = self.validator.validate_model(
            self.mock_model, self.mock_data_loader, bootstrap_samples=10  # Small number for testing
        )

        # Check results structure
        self.assertIn("accuracy", results)
        self.assertIn("auc", results)
        self.assertIn("precision", results)
        self.assertIn("recall", results)
        self.assertIn("f1_score", results)
        self.assertIn("confusion_matrix", results)
        self.assertIn("bootstrap_ci", results)
        self.assertIn("validation_passed", results)
        self.assertIn("num_samples", results)
        self.assertIn("num_classes", results)

        # Check data types
        self.assertIsInstance(results["accuracy"], float)
        self.assertIsInstance(results["auc"], float)
        self.assertIsInstance(results["precision"], list)
        self.assertIsInstance(results["recall"], list)
        self.assertIsInstance(results["f1_score"], list)
        self.assertIsInstance(results["confusion_matrix"], list)
        self.assertIsInstance(results["bootstrap_ci"], dict)
        self.assertIsInstance(results["validation_passed"], bool)

        # Check ranges
        self.assertGreaterEqual(results["accuracy"], 0.0)
        self.assertLessEqual(results["accuracy"], 1.0)
        self.assertGreaterEqual(results["auc"], 0.0)
        self.assertLessEqual(results["auc"], 1.0)

        # Check that performance history was updated
        self.assertEqual(len(self.validator.performance_history), 1)

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval calculation."""
        # Create deterministic data for testing
        labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        predictions = np.array([0, 0, 1, 1, 0, 1, 1, 0])  # Some errors
        probabilities = np.array(
            [
                [0.8, 0.2],
                [0.9, 0.1],
                [0.3, 0.7],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.1, 0.9],
                [0.4, 0.6],
                [0.6, 0.4],
            ]
        )

        ci_results = self.validator._calculate_bootstrap_confidence_intervals(
            labels, predictions, probabilities, n_bootstrap=100, confidence_level=0.95
        )

        # Check that confidence intervals are returned
        self.assertIn("accuracy", ci_results)
        self.assertIn("auc", ci_results)

        # Check CI structure
        accuracy_ci = ci_results["accuracy"]
        self.assertIsInstance(accuracy_ci, tuple)
        self.assertEqual(len(accuracy_ci), 2)
        self.assertLessEqual(accuracy_ci[0], accuracy_ci[1])  # Lower <= Upper

        auc_ci = ci_results["auc"]
        self.assertIsInstance(auc_ci, tuple)
        self.assertEqual(len(auc_ci), 2)
        self.assertLessEqual(auc_ci[0], auc_ci[1])  # Lower <= Upper

    def test_concept_drift_detection(self):
        """Test concept drift detection functionality."""
        # Add some performance history with declining performance
        for i in range(10):
            accuracy = 0.95 - (i * 0.02)  # Declining accuracy
            auc = 0.98 - (i * 0.01)  # Declining AUC

            self.validator.performance_history.append(
                {"accuracy": accuracy, "auc": auc, "timestamp": time.time() + i}
            )

        # Detect drift
        drift_results = self.validator.detect_concept_drift()

        # Check results structure
        self.assertIn("drift_detected", drift_results)
        self.assertIn("drift_type", drift_results)
        self.assertIn("drift_magnitude", drift_results)
        self.assertIn("confidence", drift_results)

        # Should detect drift due to declining performance
        self.assertTrue(drift_results["drift_detected"])
        self.assertIsNotNone(drift_results["drift_type"])
        self.assertGreater(drift_results["drift_magnitude"], 0)

    def test_performance_degradation_detection(self):
        """Test performance degradation detection."""
        # Add performance history with good performance
        for i in range(5):
            self.validator.performance_history.append(
                {"accuracy": 0.92, "auc": 0.96, "timestamp": time.time() + i}
            )

        # Should not detect degradation
        self.assertFalse(self.validator.detect_performance_degradation())

        # Add poor performance
        self.validator.performance_history.append(
            {
                "accuracy": 0.85,  # Below threshold
                "auc": 0.93,  # Below threshold
                "timestamp": time.time() + 6,
            }
        )

        # Should detect degradation
        self.assertTrue(self.validator.detect_performance_degradation())

    def test_retraining_recommendations(self):
        """Test retraining recommendation generation."""
        # Add performance history with poor performance
        self.validator.performance_history.append(
            {
                "accuracy": 0.85,  # Below threshold
                "auc": 0.90,  # Below threshold
                "validation_passed": False,
                "subpopulation_results": {
                    "age_groups": {
                        "under_40": {"accuracy": 0.80, "num_samples": 20},
                        "over_65": {"accuracy": 0.88, "num_samples": 30},
                    }
                },
            }
        )

        recommendations = self.validator.recommend_retraining()

        # Check structure
        self.assertIn("should_retrain", recommendations)
        self.assertIn("urgency", recommendations)
        self.assertIn("reasons", recommendations)
        self.assertIn("suggested_actions", recommendations)

        # Should recommend retraining
        self.assertTrue(recommendations["should_retrain"])
        self.assertIn(recommendations["urgency"], ["low", "medium", "high", "critical"])
        self.assertIsInstance(recommendations["reasons"], list)
        self.assertIsInstance(recommendations["suggested_actions"], list)
        self.assertGreater(len(recommendations["reasons"]), 0)
        self.assertGreater(len(recommendations["suggested_actions"]), 0)

    def test_alert_system(self):
        """Test alert callback system."""
        # Add alert callback
        alert_calls = []

        def test_callback(alert_type, data):
            alert_calls.append((alert_type, data))

        self.validator.add_alert_callback(test_callback)

        # Trigger alert
        test_data = {"accuracy": 0.85, "auc": 0.90}
        self.validator._trigger_alerts("validation_failed", test_data)

        # Check that callback was called
        self.assertEqual(len(alert_calls), 1)
        self.assertEqual(alert_calls[0][0], "validation_failed")
        self.assertEqual(alert_calls[0][1], test_data)

    def test_performance_summary(self):
        """Test performance summary generation."""
        # Empty history
        summary = self.validator.get_performance_summary()
        self.assertEqual(summary["status"], "no_data")

        # Add some performance history
        for i in range(5):
            self.validator.performance_history.append(
                {
                    "accuracy": 0.92 + (i * 0.01),  # Improving
                    "auc": 0.96,
                    "validation_passed": True,
                    "timestamp": time.time() + i,
                }
            )

        summary = self.validator.get_performance_summary()

        # Check structure
        self.assertIn("status", summary)
        self.assertIn("latest_accuracy", summary)
        self.assertIn("latest_auc", summary)
        self.assertIn("accuracy_trend", summary)
        self.assertIn("validation_passed", summary)
        self.assertIn("total_evaluations", summary)

        # Check values
        self.assertEqual(summary["status"], "healthy")
        self.assertEqual(summary["accuracy_trend"], "improving")
        self.assertTrue(summary["validation_passed"])
        self.assertEqual(summary["total_evaluations"], 5)

    def test_export_performance_history(self):
        """Test performance history export."""
        # Add some performance history
        self.validator.performance_history.append(
            {
                "accuracy": 0.92,
                "auc": 0.96,
                "confusion_matrix": [[10, 2], [1, 15]],
                "timestamp": time.time(),
            }
        )

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        self.validator.export_performance_history(temp_path)

        # Read back and verify
        with open(temp_path, "r") as f:
            exported_data = json.load(f)

        self.assertIn("performance_history", exported_data)
        self.assertIn("configuration", exported_data)
        self.assertEqual(len(exported_data["performance_history"]), 1)
        self.assertEqual(exported_data["configuration"]["accuracy_threshold"], 0.90)

    def test_subpopulation_validation(self):
        """Test validation by patient subpopulation."""
        # Create metadata with different subpopulations
        metadata = [
            {"age": 35, "sex": "M", "smoking_status": "never"},
            {"age": 45, "sex": "F", "smoking_status": "former"},
            {"age": 70, "sex": "M", "smoking_status": "current"},
            {"age": 25, "sex": "F", "smoking_status": "never"},
        ]

        labels = np.array([0, 1, 1, 0])
        predictions = np.array([0, 1, 0, 0])  # One error
        probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1]])

        results = self.validator._validate_by_subpopulation(
            metadata, labels, predictions, probabilities
        )

        # Check structure
        self.assertIn("age_groups", results)
        self.assertIn("sex", results)
        self.assertIn("smoking_status", results)

        # Check that subpopulations were created
        self.assertIn("under_40", results["age_groups"])
        self.assertIn("over_65", results["age_groups"])
        self.assertIn("M", results["sex"])
        self.assertIn("F", results["sex"])


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for PerformanceMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ModelValidator()
        self.monitor = PerformanceMonitor(
            validator=self.validator,
            monitoring_interval=1.0,  # 1 second for testing
            alert_email="test@example.com",
            alert_webhook="http://example.com/webhook",
        )

        # Create mock model and data loader
        self.mock_model = MagicMock(spec=MultiClassDiseaseClassifier)
        self.mock_data_loader = MagicMock()

    def test_initialization(self):
        """Test PerformanceMonitor initialization."""
        self.assertEqual(self.monitor.validator, self.validator)
        self.assertEqual(self.monitor.monitoring_interval, 1.0)
        self.assertEqual(self.monitor.alert_email, "test@example.com")
        self.assertEqual(self.monitor.alert_webhook, "http://example.com/webhook")
        self.assertFalse(self.monitor.is_monitoring)
        self.assertEqual(len(self.validator.alert_callbacks), 1)

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        # Start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.is_monitoring)
        self.assertGreater(self.monitor.last_check_time, 0)

        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.is_monitoring)

    @patch("src.clinical.validation.time.time")
    def test_check_performance_timing(self, mock_time):
        """Test performance check timing logic."""
        # Mock time progression
        mock_time.side_effect = [0, 0.5, 2.0]  # Start, too early, then time to check

        self.monitor.start_monitoring()

        # First check - too early
        with patch.object(self.validator, "track_performance") as mock_track:
            result = self.monitor.check_performance(self.mock_model, self.mock_data_loader)
            self.assertFalse(result)
            mock_track.assert_not_called()

        # Second check - time to run
        with patch.object(self.validator, "track_performance") as mock_track:
            mock_track.return_value = {"accuracy": 0.92, "auc": 0.96}
            with patch.object(self.validator, "get_performance_summary") as mock_summary:
                mock_summary.return_value = {"status": "healthy"}

                result = self.monitor.check_performance(self.mock_model, self.mock_data_loader)
                self.assertTrue(result)
                mock_track.assert_called_once()

    def test_alert_handling(self):
        """Test alert handling functionality."""
        # Test alert message formatting
        test_data = {"accuracy": 0.85, "auc": 0.90}

        message = self.monitor._format_alert_message("validation_failed", test_data)
        self.assertIn("validation failed", message.lower())
        self.assertIn("0.85", message)
        self.assertIn("0.90", message)

        # Test drift alert
        drift_data = {"drift_detection": {"drift_type": "accuracy", "drift_magnitude": 0.08}}
        message = self.monitor._format_alert_message("concept_drift", drift_data)
        self.assertIn("concept drift", message.lower())
        self.assertIn("accuracy", message)

    def test_alert_callbacks(self):
        """Test that alerts trigger callbacks correctly."""
        alert_calls = []

        def test_callback(alert_type, data):
            alert_calls.append((alert_type, data))

        # Replace the monitor's alert callback
        self.validator.alert_callbacks = [test_callback]

        # Trigger alert through validator
        test_data = {"accuracy": 0.85}
        self.validator._trigger_alerts("performance_degradation", test_data)

        # Check that callback was called
        self.assertEqual(len(alert_calls), 1)
        self.assertEqual(alert_calls[0][0], "performance_degradation")


class TestValidationIntegration(unittest.TestCase):
    """Integration tests for validation system."""

    def setUp(self):
        """Set up integration test fixtures."""

        # Create a simple mock model that can actually run
        class SimpleMockModel(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.classifier = nn.Linear(1024, num_classes)
                self.taxonomy = None

            def forward(self, batch):
                # Simple forward pass for testing
                if isinstance(batch, dict) and "wsi_features" in batch:
                    features = batch["wsi_features"]
                    # Average pool over patches
                    pooled = features.mean(dim=1)  # [batch, 1024]
                    return self.classifier(pooled)
                else:
                    return self.classifier(batch)

        self.model = SimpleMockModel()

        # Create synthetic dataset
        self.batch_size = 16
        self.num_patches = 10
        self.feature_dim = 1024
        self.num_samples = 64
        self.num_classes = 2

        # Generate synthetic data
        features = torch.randn(self.num_samples, self.num_patches, self.feature_dim)
        labels = torch.randint(0, self.num_classes, (self.num_samples,))

        # Create dataset and dataloader
        dataset = TensorDataset(features, labels)
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Create validator
        self.validator = ModelValidator(
            accuracy_threshold=0.5,  # Lower threshold for random model
            auc_threshold=0.5,
            max_history_length=10,
        )

    def test_end_to_end_validation(self):
        """Test complete validation workflow."""
        # Store reference to test instance
        test_instance = self

        # Mock the data loader to return proper batch format
        def mock_iter():
            for features, labels in test_instance.data_loader:
                yield {
                    "wsi_features": features,
                    "labels": labels,
                    "metadata": [
                        {
                            "age": np.random.randint(20, 80),
                            "sex": np.random.choice(["M", "F"]),
                            "smoking_status": np.random.choice(["never", "former", "current"]),
                        }
                        for _ in range(len(labels))
                    ],
                }

        mock_data_loader = MagicMock()
        mock_data_loader.__iter__ = lambda self: mock_iter()

        # Run validation
        results = self.validator.validate_model(
            test_instance.model, mock_data_loader, bootstrap_samples=10
        )

        # Check that validation completed successfully
        self.assertIsInstance(results, dict)
        self.assertIn("accuracy", results)
        self.assertIn("validation_passed", results)
        self.assertGreater(results["num_samples"], 0)

        # Check that performance history was updated
        self.assertEqual(len(self.validator.performance_history), 1)

        # Test performance tracking
        tracking_results = self.validator.track_performance(self.model, mock_data_loader)

        self.assertIn("drift_detection", tracking_results)
        self.assertIn("performance_degradation", tracking_results)

        # Check that history now has 2 entries
        self.assertEqual(len(self.validator.performance_history), 2)


if __name__ == "__main__":
    unittest.main()
