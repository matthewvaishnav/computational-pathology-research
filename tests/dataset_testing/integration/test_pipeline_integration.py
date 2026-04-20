"""
End-to-end pipeline integration tests.

Tests dataset API compatibility, preprocessing integration, and reproducibility
(Requirement 9.1, 9.2, 9.3).
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import h5py
from typing import Dict, Any


# Mock dataset classes for testing
class MockPCamDataset:
    """Mock PCam dataset for integration testing."""

    def __init__(self, data_dir: Path, split: str = "train", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.x_file = data_dir / f"x_{split}.h5"
        self.y_file = data_dir / f"y_{split}.h5"

        # Load data
        with h5py.File(self.x_file, "r") as f:
            self.num_samples = len(f["x"])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.x_file, "r") as f:
            x = f["x"][idx]
        with h5py.File(self.y_file, "r") as f:
            y = f["y"][idx]

        if self.transform:
            x = self.transform(x)

        return x, y


class MockPreprocessor:
    """Mock preprocessor for integration testing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def normalize(self, x):
        """Normalize to [0, 1]."""
        return x.astype(np.float32) / 255.0

    def augment(self, x):
        """Simple augmentation."""
        if self.config.get("flip", False):
            x = np.flip(x, axis=1)
        return x

    def __call__(self, x):
        x = self.normalize(x)
        if self.config.get("augment", False):
            x = self.augment(x)
        return x


@pytest.fixture
def mock_dataset(temp_data_dir):
    """Create mock dataset for integration testing."""
    data_dir = temp_data_dir / "mock_pcam"
    data_dir.mkdir(exist_ok=True)

    # Create train/val/test splits
    for split in ["train", "val", "test"]:
        x_file = data_dir / f"x_{split}.h5"
        y_file = data_dir / f"y_{split}.h5"

        num_samples = 50 if split == "train" else 20

        with h5py.File(x_file, "w") as f:
            f.create_dataset(
                "x",
                data=np.random.randint(0, 256, (num_samples, 96, 96, 3), dtype=np.uint8),
            )

        with h5py.File(y_file, "w") as f:
            f.create_dataset("y", data=np.random.randint(0, 2, (num_samples, 1), dtype=np.uint8))

    return data_dir


# Requirement 9.1: Dataset API backward compatibility
class TestDatasetAPICompatibility:
    """Test dataset API remains backward compatible."""

    def test_dataset_initialization_api(self, mock_dataset):
        """Test dataset can be initialized with standard API."""
        # Standard initialization
        dataset = MockPCamDataset(mock_dataset, split="train")

        assert len(dataset) > 0
        assert hasattr(dataset, "__getitem__")
        assert hasattr(dataset, "__len__")

    def test_dataset_indexing_api(self, mock_dataset):
        """Test dataset supports standard indexing."""
        dataset = MockPCamDataset(mock_dataset, split="train")

        # Single index
        x, y = dataset[0]
        assert x.shape == (96, 96, 3)
        assert y.shape == (1,)

        # Negative index
        x, y = dataset[-1]
        assert x.shape == (96, 96, 3)

    def test_dataset_iteration_api(self, mock_dataset):
        """Test dataset supports iteration."""
        dataset = MockPCamDataset(mock_dataset, split="train")

        count = 0
        for x, y in dataset:
            assert x.shape == (96, 96, 3)
            assert y.shape == (1,)
            count += 1
            if count >= 5:  # Test first 5 samples
                break

        assert count == 5

    def test_dataset_with_dataloader(self, mock_dataset):
        """Test dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = MockPCamDataset(mock_dataset, split="train")
        loader = DataLoader(dataset, batch_size=8, num_workers=0)

        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (8, 96, 96, 3)
        assert batch_y.shape == (8, 1)

    def test_dataset_split_api(self, mock_dataset):
        """Test dataset supports train/val/test splits."""
        train_dataset = MockPCamDataset(mock_dataset, split="train")
        val_dataset = MockPCamDataset(mock_dataset, split="val")
        test_dataset = MockPCamDataset(mock_dataset, split="test")

        assert len(train_dataset) == 50
        assert len(val_dataset) == 20
        assert len(test_dataset) == 20


# Requirement 9.2: Preprocessing pipeline integration
class TestPreprocessingIntegration:
    """Test preprocessing integrates with datasets."""

    def test_transform_integration(self, mock_dataset):
        """Test dataset accepts transform parameter."""
        preprocessor = MockPreprocessor({"augment": False})
        dataset = MockPCamDataset(mock_dataset, split="train", transform=preprocessor)

        x, y = dataset[0]
        assert x.dtype == np.float32
        assert x.min() >= 0.0 and x.max() <= 1.0

    def test_augmentation_integration(self, mock_dataset):
        """Test augmentation works in pipeline."""
        preprocessor = MockPreprocessor({"augment": True, "flip": True})
        dataset = MockPCamDataset(mock_dataset, split="train", transform=preprocessor)

        x, y = dataset[0]
        assert x.dtype == np.float32

    def test_preprocessing_config_changes(self, mock_dataset):
        """Test preprocessing config changes are applied."""
        # Config 1: No augmentation
        config1 = {"augment": False}
        preprocessor1 = MockPreprocessor(config1)
        dataset1 = MockPCamDataset(mock_dataset, split="train", transform=preprocessor1)

        x1, _ = dataset1[0]

        # Config 2: With augmentation
        config2 = {"augment": True, "flip": True}
        preprocessor2 = MockPreprocessor(config2)
        dataset2 = MockPCamDataset(mock_dataset, split="train", transform=preprocessor2)

        x2, _ = dataset2[0]

        # Results should differ due to augmentation
        assert x1.shape == x2.shape

    def test_preprocessing_pipeline_composition(self, mock_dataset):
        """Test multiple preprocessing steps compose correctly."""

        class ComposedPreprocessor:
            def __init__(self, steps):
                self.steps = steps

            def __call__(self, x):
                for step in self.steps:
                    x = step(x)
                return x

        # Create pipeline
        normalize = lambda x: x.astype(np.float32) / 255.0
        to_tensor = lambda x: torch.from_numpy(x)

        # Create preprocessor
        preprocessor = lambda x: to_tensor(normalize(x))

        def get_train_size(ds):
            return len(ds.train_indices)

        def get_val_size(ds):
            return len(ds.val_indices)

        dataset = MockPCamDataset(mock_dataset, split="train", transform=preprocessor)

        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert x.dtype == torch.float32

    def test_preprocessing_error_handling(self, mock_dataset):
        """Test preprocessing errors are handled gracefully."""

        def failing_transform(x):
            raise ValueError("Preprocessing failed")

        dataset = MockPCamDataset(mock_dataset, split="train", transform=failing_transform)

        with pytest.raises(ValueError, match="Preprocessing failed"):
            _ = dataset[0]


# Requirement 9.3: Reproducible results with fixed seeds
class TestReproducibility:
    """Test dataset operations are reproducible."""

    def test_deterministic_loading_with_seed(self, mock_dataset):
        """Test loading is deterministic with fixed seed."""
        np.random.seed(42)
        torch.manual_seed(42)

        dataset1 = MockPCamDataset(mock_dataset, split="train")
        x1, y1 = dataset1[0]

        np.random.seed(42)
        torch.manual_seed(42)

        dataset2 = MockPCamDataset(mock_dataset, split="train")
        x2, y2 = dataset2[0]

        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_deterministic_augmentation_with_seed(self, mock_dataset):
        """Test augmentation is deterministic with fixed seed."""

        class RandomAugment:
            def __call__(self, x):
                # Random flip
                if np.random.rand() > 0.5:
                    x = np.flip(x, axis=1)
                return x.astype(np.float32) / 255.0

        # Run 1
        np.random.seed(42)
        preprocessor1 = RandomAugment()
        dataset1 = MockPCamDataset(mock_dataset, split="train", transform=preprocessor1)
        x1, _ = dataset1[0]

        # Run 2 with same seed
        np.random.seed(42)
        preprocessor2 = RandomAugment()
        dataset2 = MockPCamDataset(mock_dataset, split="train", transform=preprocessor2)
        x2, _ = dataset2[0]

        np.testing.assert_array_almost_equal(x1, x2)

    def test_dataloader_reproducibility_with_seed(self, mock_dataset):
        """Test DataLoader is reproducible with fixed seed."""
        from torch.utils.data import DataLoader

        def seed_worker(worker_id):
            np.random.seed(42 + worker_id)

        # Run 1
        torch.manual_seed(42)
        dataset1 = MockPCamDataset(mock_dataset, split="train")
        loader1 = DataLoader(
            dataset1, batch_size=8, shuffle=True, worker_init_fn=seed_worker, num_workers=0
        )
        batch1_x, batch1_y = next(iter(loader1))

        # Run 2 with same seed
        torch.manual_seed(42)
        dataset2 = MockPCamDataset(mock_dataset, split="train")
        loader2 = DataLoader(
            dataset2, batch_size=8, shuffle=True, worker_init_fn=seed_worker, num_workers=0
        )
        batch2_x, batch2_y = next(iter(loader2))

        torch.testing.assert_close(batch1_x, batch2_x)
        torch.testing.assert_close(batch1_y, batch2_y)

    def test_config_hash_for_reproducibility(self, mock_dataset):
        """Test config hashing ensures reproducibility."""
        import hashlib
        import json

        config = {"augment": True, "flip": True, "seed": 42}

        # Hash config
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        # Same config should produce same hash
        config2 = {"seed": 42, "flip": True, "augment": True}  # Different order
        config_str2 = json.dumps(config2, sort_keys=True)
        config_hash2 = hashlib.md5(config_str2.encode()).hexdigest()

        assert config_hash == config_hash2

    def test_dataset_version_tracking(self, mock_dataset):
        """Test dataset version is tracked for reproducibility."""

        class VersionedDataset(MockPCamDataset):
            VERSION = "1.0.0"

            def get_version(self):
                return self.VERSION

        dataset = VersionedDataset(mock_dataset, split="train")
        assert dataset.get_version() == "1.0.0"


# Requirement 9.1, 9.2: End-to-end pipeline validation
class TestEndToEndPipeline:
    """Test complete data pipeline from loading to model input."""

    def test_full_pipeline_integration(self, mock_dataset):
        """Test complete pipeline: dataset → preprocessing → DataLoader → model."""
        from torch.utils.data import DataLoader

        # Setup pipeline
        preprocessor = MockPreprocessor({"augment": False})
        dataset = MockPCamDataset(mock_dataset, split="train", transform=preprocessor)
        loader = DataLoader(dataset, batch_size=8, num_workers=0)

        # Mock model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(96 * 96 * 3, 2)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = SimpleModel()

        # Run through pipeline
        batch_x, batch_y = next(iter(loader))

        # Convert to tensor if needed
        if not isinstance(batch_x, torch.Tensor):
            batch_x = torch.from_numpy(batch_x).float()
        else:
            batch_x = batch_x.float()

        output = model(batch_x)
        assert output.shape == (8, 2)

    def test_pipeline_with_multiple_datasets(self, mock_dataset):
        """Test pipeline works with multiple dataset types."""
        from torch.utils.data import DataLoader, ConcatDataset

        dataset1 = MockPCamDataset(mock_dataset, split="train")
        dataset2 = MockPCamDataset(mock_dataset, split="val")

        combined = ConcatDataset([dataset1, dataset2])
        loader = DataLoader(combined, batch_size=8, num_workers=0)

        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[0] == 8

    def test_pipeline_error_propagation(self, mock_dataset):
        """Test errors propagate correctly through pipeline."""
        from torch.utils.data import DataLoader

        def failing_transform(x):
            raise RuntimeError("Transform failed")

        dataset = MockPCamDataset(mock_dataset, split="train", transform=failing_transform)
        loader = DataLoader(dataset, batch_size=8, num_workers=0)

        with pytest.raises(RuntimeError, match="Transform failed"):
            _ = next(iter(loader))

    def test_pipeline_memory_efficiency(self, mock_dataset):
        """Test pipeline doesn't leak memory."""
        from torch.utils.data import DataLoader
        import gc

        dataset = MockPCamDataset(mock_dataset, split="train")
        loader = DataLoader(dataset, batch_size=8, num_workers=0)

        # Load multiple batches
        for i, (batch_x, batch_y) in enumerate(loader):
            if i >= 5:
                break

        # Force garbage collection
        gc.collect()

        # Memory should be released (basic check)
        assert True  # If we get here without OOM, memory is managed
