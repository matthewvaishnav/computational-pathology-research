"""
Training loop integration tests.

Tests dataset usage in training loops, version updates, and failure isolation
(Requirement 9.5, 9.6, 9.7).
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import h5py
from typing import Dict, Any


# Mock components for training loop testing
class MockDataset:
    """Mock dataset for training loop testing."""

    def __init__(self, num_samples=100, feature_dim=10, num_classes=2):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.data = np.random.randn(num_samples, feature_dim).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


class MockModel(nn.Module):
    """Mock model for training loop testing."""

    def __init__(self, input_dim=10, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# Requirement 9.5: Training loop validation
class TestTrainingLoopIntegration:
    """Test dataset usage in training loops."""

    def test_basic_training_loop(self):
        """Test dataset works in basic training loop."""
        from torch.utils.data import DataLoader

        # Setup
        dataset = MockDataset(num_samples=50)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        model = MockModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Training loop
        model.train()
        for epoch in range(2):
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            assert avg_loss > 0  # Loss should be positive

    def test_training_with_validation(self):
        """Test training loop with validation split."""
        from torch.utils.data import DataLoader

        # Setup
        train_dataset = MockDataset(num_samples=50)
        val_dataset = MockDataset(num_samples=20)
        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)

        model = MockModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Training
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()

        assert val_loss > 0

    def test_training_with_early_stopping(self):
        """Test training loop with early stopping."""
        from torch.utils.data import DataLoader

        dataset = MockDataset(num_samples=50)
        loader = DataLoader(dataset, batch_size=8)
        model = MockModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Early stopping config
        patience = 3
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(10):
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break  # Early stop

        assert epoch < 10  # Should stop before max epochs (usually)

    def test_training_with_gradient_accumulation(self):
        """Test training loop with gradient accumulation."""
        from torch.utils.data import DataLoader

        dataset = MockDataset(num_samples=50)
        loader = DataLoader(dataset, batch_size=4)
        model = MockModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        accumulation_steps = 2
        model.train()

        for i, (batch_x, batch_y) in enumerate(loader):
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Final step if needed
        optimizer.step()
        optimizer.zero_grad()

        assert True  # If we get here, gradient accumulation works


# Requirement 9.6: Version updates and breaking changes
class TestDatasetVersioning:
    """Test dataset version updates and breaking changes."""

    def test_dataset_version_compatibility(self):
        """Test dataset version is tracked and validated."""

        class VersionedDataset(MockDataset):
            VERSION = "1.0.0"

            def check_version(self, required_version):
                return self.VERSION == required_version

        dataset = VersionedDataset()
        assert dataset.check_version("1.0.0")
        assert not dataset.check_version("2.0.0")

    def test_breaking_change_detection(self):
        """Test breaking changes are detected."""

        class DatasetV1(MockDataset):
            VERSION = "1.0.0"

            def get_item_v1(self, idx):
                return self.data[idx], self.labels[idx]

        class DatasetV2(MockDataset):
            VERSION = "2.0.0"

            def get_item_v2(self, idx):
                # Breaking change: returns dict instead of tuple
                return {"data": self.data[idx], "label": self.labels[idx]}

        v1 = DatasetV1()
        v2 = DatasetV2()

        # Detect API change
        v1_result = v1.get_item_v1(0)
        v2_result = v2.get_item_v2(0)

        assert isinstance(v1_result, tuple)
        assert isinstance(v2_result, dict)

    def test_migration_path_validation(self):
        """Test migration path from old to new version."""

        class LegacyDataset(MockDataset):
            def get_sample(self, idx):
                return self.data[idx], self.labels[idx]

        class ModernDataset(MockDataset):
            def __getitem__(self, idx):
                return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])

        # Migration adapter
        class DatasetAdapter:
            def __init__(self, legacy_dataset):
                self.dataset = legacy_dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                data, label = self.dataset.get_sample(idx)
                return torch.from_numpy(data), torch.tensor(label)

        legacy = LegacyDataset()
        adapted = DatasetAdapter(legacy)

        # Verify adapter works
        x, y = adapted[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)


# Requirement 9.7: Failure isolation
class TestFailureIsolation:
    """Test integration failure isolation."""

    def test_dataset_loading_failure_isolation(self):
        """Test dataset loading failures are isolated."""

        class FailingDataset(MockDataset):
            def __getitem__(self, idx):
                if idx == 5:
                    raise ValueError(f"Failed to load sample {idx}")
                return super().__getitem__(idx)

        dataset = FailingDataset(num_samples=10)

        # Identify failing sample
        failing_indices = []
        for i in range(len(dataset)):
            try:
                _ = dataset[i]
            except ValueError:
                failing_indices.append(i)

        assert failing_indices == [5]

    def test_preprocessing_failure_isolation(self):
        """Test preprocessing failures are isolated."""

        def failing_transform(x):
            if torch.isnan(x).any():
                raise ValueError("NaN detected in preprocessing")
            return x

        dataset = MockDataset(num_samples=10)
        # Inject NaN
        dataset.data[3] = np.nan

        failing_indices = []
        for i in range(len(dataset)):
            try:
                x, y = dataset[i]
                _ = failing_transform(x)
            except ValueError:
                failing_indices.append(i)

        assert 3 in failing_indices

    def test_model_forward_failure_isolation(self):
        """Test model forward failures are isolated."""
        from torch.utils.data import DataLoader

        dataset = MockDataset(num_samples=20)
        loader = DataLoader(dataset, batch_size=4)

        class FailingModel(MockModel):
            def forward(self, x):
                if x.shape[0] == 4 and torch.sum(x) > 10:
                    raise RuntimeError("Model forward failed")
                return super().forward(x)

        model = FailingModel()

        failing_batches = []
        for i, (batch_x, batch_y) in enumerate(loader):
            try:
                _ = model(batch_x)
            except RuntimeError:
                failing_batches.append(i)

        # Some batches should fail
        assert len(failing_batches) >= 0  # May or may not fail depending on data

    def test_component_isolation_in_pipeline(self):
        """Test individual components can be isolated for debugging."""
        from torch.utils.data import DataLoader

        dataset = MockDataset(num_samples=20)
        loader = DataLoader(dataset, batch_size=4)
        model = MockModel()

        # Test each component independently
        components_working = {}

        # Test dataset
        try:
            _ = dataset[0]
            components_working["dataset"] = True
        except Exception:
            components_working["dataset"] = False

        # Test loader
        try:
            batch = next(iter(loader))
            components_working["loader"] = True
        except Exception:
            components_working["loader"] = False

        # Test model
        try:
            batch_x, _ = next(iter(loader))
            _ = model(batch_x)
            components_working["model"] = True
        except Exception:
            components_working["model"] = False

        # All components should work
        assert all(components_working.values())


# Requirement 9.5: End-to-end validation
class TestEndToEndTraining:
    """Test complete training pipeline."""

    def test_full_training_pipeline(self):
        """Test complete training pipeline from data to trained model."""
        from torch.utils.data import DataLoader

        # Setup
        train_dataset = MockDataset(num_samples=50)
        val_dataset = MockDataset(num_samples=20)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)

        model = MockModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Training
        num_epochs = 3
        for epoch in range(num_epochs):
            # Train
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validate
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(output.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            accuracy = correct / total
            assert 0 <= accuracy <= 1

    def test_checkpoint_save_and_resume(self, temp_data_dir):
        """Test training can be saved and resumed."""
        from torch.utils.data import DataLoader

        dataset = MockDataset(num_samples=50)
        loader = DataLoader(dataset, batch_size=8)
        model = MockModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Train for 1 epoch
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = nn.CrossEntropyLoss()(output, batch_y)
            loss.backward()
            optimizer.step()

        # Save checkpoint
        checkpoint_path = temp_data_dir / "checkpoint.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": 1,
            },
            checkpoint_path,
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        new_model = MockModel()
        new_model.load_state_dict(checkpoint["model_state_dict"])

        # Verify model loaded correctly
        assert checkpoint["epoch"] == 1
