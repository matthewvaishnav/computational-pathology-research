#!/usr/bin/env python3
"""
Example demonstrating clinical model validation and monitoring.

This example shows how to use the ModelValidator and PerformanceMonitor
classes to validate model performance and monitor for concept drift.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.clinical.validation import ModelValidator, PerformanceMonitor
from src.clinical.classifier import MultiClassDiseaseClassifier
from src.clinical.taxonomy import DiseaseTaxonomy


def create_sample_data(num_samples=100, num_classes=3):
    """Create sample data for demonstration."""
    # Generate synthetic WSI features
    wsi_features = torch.randn(num_samples, 50, 1024)  # 50 patches per slide

    # Generate labels
    labels = torch.randint(0, num_classes, (num_samples,))

    # Generate metadata
    metadata = []
    for i in range(num_samples):
        metadata.append(
            {
                "age": np.random.randint(20, 80),
                "sex": np.random.choice(["M", "F"]),
                "smoking_status": np.random.choice(["never", "former", "current"]),
                "patient_id": f"patient_{i:03d}",
            }
        )

    return wsi_features, labels, metadata


def create_mock_dataloader(features, labels, metadata, batch_size=16):
    """Create a DataLoader that returns properly formatted batches."""

    class MockDataset:
        def __init__(self, features, labels, metadata):
            self.features = features
            self.labels = labels
            self.metadata = metadata

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return {
                "wsi_features": self.features[idx],
                "labels": self.labels[idx],
                "metadata": self.metadata[idx],
            }

    def collate_fn(batch):
        return {
            "wsi_features": torch.stack([item["wsi_features"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "metadata": [item["metadata"] for item in batch],
        }

    dataset = MockDataset(features, labels, metadata)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def create_mock_model(num_classes=3):
    """Create a mock model for demonstration."""

    class MockModel(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.classifier = torch.nn.Linear(1024, num_classes)
            self.taxonomy = None

        def forward(self, batch):
            features = batch["wsi_features"]
            # Average pool over patches
            pooled = features.mean(dim=1)  # [batch, 1024]
            return self.classifier(pooled)

        def eval(self):
            return self

        def parameters(self):
            return self.classifier.parameters()

    return MockModel(num_classes)


def demonstrate_validation():
    """Demonstrate model validation functionality."""
    print("🔬 Clinical Model Validation Example")
    print("=" * 50)

    # Create sample data
    print("📊 Creating sample data...")
    features, labels, metadata = create_sample_data(num_samples=200, num_classes=3)

    # Split into validation and monitoring sets
    val_features, val_labels, val_metadata = features[:100], labels[:100], metadata[:100]
    mon_features, mon_labels, mon_metadata = features[100:], labels[100:], metadata[100:]

    # Create data loaders
    val_loader = create_mock_dataloader(val_features, val_labels, val_metadata)
    mon_loader = create_mock_dataloader(mon_features, mon_labels, mon_metadata)

    # Create model
    print("🤖 Creating mock model...")
    model = create_mock_model(num_classes=3)

    # Create validator
    print("✅ Initializing ModelValidator...")
    validator = ModelValidator(
        accuracy_threshold=0.70,  # Lower threshold for random model
        auc_threshold=0.70,
        max_history_length=20,
        drift_detection_window=3,
        performance_degradation_threshold=0.10,
    )

    # Add alert callback
    def alert_callback(alert_type, data):
        print(f"🚨 ALERT: {alert_type}")
        if alert_type == "validation_failed":
            print(f"   Accuracy: {data.get('accuracy', 0):.3f}")
            print(f"   AUC: {data.get('auc', 0):.3f}")
        elif alert_type == "concept_drift":
            drift_info = data.get("drift_detection", {})
            print(f"   Drift Type: {drift_info.get('drift_type', 'unknown')}")
            print(f"   Magnitude: {drift_info.get('drift_magnitude', 0):.3f}")

    validator.add_alert_callback(alert_callback)

    # Run initial validation
    print("\n🔍 Running initial model validation...")
    results = validator.validate_model(model, val_loader, bootstrap_samples=50)

    print(f"   Accuracy: {results['accuracy']:.3f}")
    print(f"   AUC: {results['auc']:.3f}")
    print(f"   Validation Passed: {results['validation_passed']}")
    print(f"   Number of Samples: {results['num_samples']}")

    if "bootstrap_ci" in results:
        print(f"   Bootstrap CI (Accuracy): {results['bootstrap_ci'].get('accuracy', 'N/A')}")
        print(f"   Bootstrap CI (AUC): {results['bootstrap_ci'].get('auc', 'N/A')}")

    if "subpopulation_results" in results:
        print("\n👥 Subpopulation Results:")
        for pop_type, populations in results["subpopulation_results"].items():
            print(f"   {pop_type}:")
            for pop_name, metrics in populations.items():
                print(
                    f"     {pop_name}: {metrics['accuracy']:.3f} ({metrics['num_samples']} samples)"
                )

    # Simulate performance monitoring over time
    print("\n📈 Simulating performance monitoring over time...")

    # Create performance monitor
    monitor = PerformanceMonitor(
        validator=validator,
        monitoring_interval=0.1,  # Very short for demo
        alert_email="admin@hospital.com",
    )

    monitor.start_monitoring()

    # Simulate multiple monitoring cycles with degrading performance
    for cycle in range(5):
        print(f"\n   Monitoring Cycle {cycle + 1}:")

        # Simulate model degradation by adding noise to predictions
        if cycle > 2:
            # Add noise to model weights to simulate degradation
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)

        # Track performance
        tracking_results = validator.track_performance(model, mon_loader)

        print(f"     Accuracy: {tracking_results['accuracy']:.3f}")
        print(f"     Drift Detected: {tracking_results['drift_detection']['drift_detected']}")
        print(f"     Performance Degradation: {tracking_results['performance_degradation']}")

    monitor.stop_monitoring()

    # Get final performance summary
    print("\n📋 Final Performance Summary:")
    summary = validator.get_performance_summary()
    print(f"   Status: {summary['status']}")
    print(f"   Latest Accuracy: {summary['latest_accuracy']:.3f}")
    print(f"   Accuracy Trend: {summary['accuracy_trend']}")
    print(f"   Total Evaluations: {summary['total_evaluations']}")
    print(f"   Drift Detected: {summary['drift_detected']}")
    print(f"   Performance Degradation: {summary['performance_degradation']}")

    # Get retraining recommendations
    print("\n🔄 Retraining Recommendations:")
    recommendations = summary["retraining_recommendation"]
    print(f"   Should Retrain: {recommendations['should_retrain']}")
    print(f"   Urgency: {recommendations['urgency']}")
    print(f"   Reasons: {len(recommendations['reasons'])}")

    if recommendations["reasons"]:
        print("   Specific Reasons:")
        for reason in recommendations["reasons"]:
            print(f"     - {reason}")

    if recommendations["suggested_actions"]:
        print("   Suggested Actions:")
        for action in recommendations["suggested_actions"][:3]:  # Show first 3
            print(f"     - {action}")

    print("\n✨ Validation example completed!")


if __name__ == "__main__":
    demonstrate_validation()
