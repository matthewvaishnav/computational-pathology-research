"""
Unit tests for task-aware batch samplers.

Tests cover:
- BalancedBatchSampler: Equal class representation, minority oversampling
- RegressionBatchSampler: Binned sampling for uniform target coverage
- SurvivalBatchSampler: Event-balanced sampling for survival analysis
"""

import pytest
import torch

from src.data.loaders.batch_samplers import (
    BalancedBatchSampler,
    RegressionBatchSampler,
    SurvivalBatchSampler,
)

# ============================================================================
# BalancedBatchSampler Tests
# ============================================================================


class TestBalancedBatchSamplerInitialization:
    """Test BalancedBatchSampler initialization and configuration validation."""

    def test_valid_initialization(self):
        """Test initialization with valid parameters."""
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        sampler = BalancedBatchSampler(labels, batch_size=6, shuffle=True)

        assert sampler.batch_size == 6
        assert sampler.shuffle is True
        assert sampler.num_classes == 3
        assert sampler.num_samples == 6

    def test_initialization_with_list(self):
        """Test initialization with Python list instead of tensor."""
        labels = [0, 0, 1, 1, 2, 2]
        sampler = BalancedBatchSampler(labels, batch_size=6, shuffle=False)

        assert sampler.num_classes == 3
        assert sampler.num_samples == 6

    def test_empty_labels_raises_error(self):
        """Test that empty labels raises ValueError."""
        with pytest.raises(ValueError, match="labels cannot be empty"):
            BalancedBatchSampler([], batch_size=32)

    def test_negative_batch_size_raises_error(self):
        """Test that negative batch_size raises ValueError."""
        labels = [0, 1, 2]
        with pytest.raises(ValueError, match="batch_size must be positive"):
            BalancedBatchSampler(labels, batch_size=-1)

    def test_zero_batch_size_raises_error(self):
        """Test that zero batch_size raises ValueError."""
        labels = [0, 1, 2]
        with pytest.raises(ValueError, match="batch_size must be positive"):
            BalancedBatchSampler(labels, batch_size=0)

    def test_default_parameters(self):
        """Test default parameter values."""
        labels = [0, 1, 2]
        sampler = BalancedBatchSampler(labels)

        assert sampler.batch_size == 32
        assert sampler.shuffle is True


class TestBalancedBatchSamplerBalancing:
    """Test class balancing functionality."""

    def test_balanced_dataset(self):
        """Test with perfectly balanced dataset."""
        # 10 samples per class, 3 classes
        labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
        sampler = BalancedBatchSampler(labels, batch_size=6, shuffle=False)

        # Each batch should have 2 samples per class
        assert sampler.samples_per_class == 2

        # Generate first batch
        batch = list(next(iter(sampler)))
        assert len(batch) == 6

        # Count class representation in batch
        batch_labels = labels[batch]
        for class_id in range(3):
            count = (batch_labels == class_id).sum().item()
            assert count == 2  # Equal representation

    def test_imbalanced_dataset(self):
        """Test with imbalanced dataset (minority class oversampling)."""
        # Imbalanced: 20 class-0, 10 class-1, 5 class-2
        labels = torch.tensor([0] * 20 + [1] * 10 + [2] * 5)
        sampler = BalancedBatchSampler(labels, batch_size=9, shuffle=False)

        # Each batch should have 3 samples per class
        assert sampler.samples_per_class == 3

        # Generate multiple batches and check class distribution
        batches = list(sampler)

        # Each batch should have approximately equal class representation
        for batch in batches[:3]:  # Check first 3 batches
            batch_labels = labels[batch]
            for class_id in range(3):
                count = (batch_labels == class_id).sum().item()
                # Should be close to samples_per_class (3)
                assert count >= 2 and count <= 4

    def test_minority_class_oversampling(self):
        """Test that minority classes are oversampled."""
        # Highly imbalanced: 100 class-0, 10 class-1
        labels = torch.tensor([0] * 100 + [1] * 10)
        sampler = BalancedBatchSampler(labels, batch_size=20, shuffle=False)

        # Collect all batches
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)

        # Count how many times each class appears
        all_labels = labels[all_indices]
        class_0_count = (all_labels == 0).sum().item()
        class_1_count = (all_labels == 1).sum().item()

        # Class 1 should appear multiple times (oversampled)
        # Since we have 10 class-1 samples but need many batches,
        # class-1 samples must be reused
        assert class_1_count > 10  # Oversampled

    def test_single_class(self):
        """Test with single class (no balancing needed)."""
        labels = torch.tensor([0] * 20)
        sampler = BalancedBatchSampler(labels, batch_size=5, shuffle=False)

        assert sampler.num_classes == 1

        batch = list(next(iter(sampler)))
        assert len(batch) == 5


class TestBalancedBatchSamplerBatchGeneration:
    """Test batch generation functionality."""

    def test_batch_size_correctness(self):
        """Test that generated batches have correct size."""
        labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
        sampler = BalancedBatchSampler(labels, batch_size=12, shuffle=False)

        for batch in sampler:
            assert len(batch) <= 12  # May be smaller for last batch

    def test_shuffle_produces_different_batches(self):
        """Test that shuffle=True produces different batches across epochs."""
        labels = torch.tensor([0] * 20 + [1] * 20 + [2] * 20)
        sampler = BalancedBatchSampler(labels, batch_size=12, shuffle=True)

        # Generate first batch from two epochs
        batch1 = list(next(iter(sampler)))
        batch2 = list(next(iter(sampler)))

        # Batches should be different (with high probability)
        assert batch1 != batch2

    def test_no_shuffle_produces_same_batches(self):
        """Test that shuffle=False produces same batches across epochs."""
        labels = torch.tensor([0] * 20 + [1] * 20 + [2] * 20)
        sampler = BalancedBatchSampler(labels, batch_size=12, shuffle=False)

        # Generate all batches from two epochs
        batches1 = list(sampler)
        batches2 = list(sampler)

        # Batches should be identical
        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2):
            assert b1 == b2

    def test_len_method(self):
        """Test __len__ method returns correct number of batches."""
        labels = torch.tensor([0] * 20 + [1] * 20 + [2] * 20)
        sampler = BalancedBatchSampler(labels, batch_size=12, shuffle=False)

        num_batches = len(sampler)
        actual_batches = len(list(sampler))

        assert num_batches == actual_batches

    def test_repr_method(self):
        """Test __repr__ method."""
        labels = torch.tensor([0] * 10 + [1] * 10)
        sampler = BalancedBatchSampler(labels, batch_size=8, shuffle=True)

        repr_str = repr(sampler)
        assert "BalancedBatchSampler" in repr_str
        assert "num_classes=2" in repr_str
        assert "batch_size=8" in repr_str
        assert "shuffle=True" in repr_str


# ============================================================================
# RegressionBatchSampler Tests
# ============================================================================


class TestRegressionBatchSamplerInitialization:
    """Test RegressionBatchSampler initialization and configuration validation."""

    def test_valid_initialization(self):
        """Test initialization with valid parameters."""
        targets = torch.tensor([0.1, 0.5, 0.9, 1.5, 2.0, 2.5])
        sampler = RegressionBatchSampler(targets, batch_size=6, num_bins=3)

        assert sampler.batch_size == 6
        assert sampler.num_bins <= 3  # May be less if bins are empty
        assert sampler.num_samples == 6

    def test_initialization_with_list(self):
        """Test initialization with Python list instead of tensor."""
        targets = [0.1, 0.5, 0.9, 1.5, 2.0, 2.5]
        sampler = RegressionBatchSampler(targets, batch_size=6, num_bins=3)

        assert sampler.num_samples == 6

    def test_empty_targets_raises_error(self):
        """Test that empty targets raises ValueError."""
        with pytest.raises(ValueError, match="targets cannot be empty"):
            RegressionBatchSampler([], batch_size=32)

    def test_negative_batch_size_raises_error(self):
        """Test that negative batch_size raises ValueError."""
        targets = [0.1, 0.5, 0.9]
        with pytest.raises(ValueError, match="batch_size must be positive"):
            RegressionBatchSampler(targets, batch_size=-1)

    def test_negative_num_bins_raises_error(self):
        """Test that negative num_bins raises ValueError."""
        targets = [0.1, 0.5, 0.9]
        with pytest.raises(ValueError, match="num_bins must be positive"):
            RegressionBatchSampler(targets, batch_size=3, num_bins=-1)

    def test_default_parameters(self):
        """Test default parameter values."""
        targets = [0.1, 0.5, 0.9]
        sampler = RegressionBatchSampler(targets)

        assert sampler.batch_size == 32
        assert sampler.num_bins <= 10  # Default, may be less if bins empty


class TestRegressionBatchSamplerBinning:
    """Test binning functionality."""

    def test_uniform_distribution(self):
        """Test with uniformly distributed targets."""
        # Targets uniformly distributed in [0, 1]
        targets = torch.linspace(0, 1, 100)
        sampler = RegressionBatchSampler(targets, batch_size=20, num_bins=5)

        # Should have 5 bins
        assert sampler.num_bins == 5

        # Each bin should have approximately 20 samples
        for bin_list in sampler.bin_indices:
            assert len(bin_list) >= 15 and len(bin_list) <= 25

    def test_bimodal_distribution(self):
        """Test with bimodal distribution."""
        # Two clusters: [0.0-0.1] and [0.9-1.0]
        targets = torch.cat([torch.linspace(0.0, 0.1, 50), torch.linspace(0.9, 1.0, 50)])
        sampler = RegressionBatchSampler(targets, batch_size=20, num_bins=10)

        # Should have bins, some may be empty
        assert sampler.num_bins >= 2
        assert sampler.num_bins <= 10

    def test_constant_targets(self):
        """Test with all targets having same value."""
        targets = torch.ones(50) * 0.5
        sampler = RegressionBatchSampler(targets, batch_size=10, num_bins=5)

        # Should collapse to 1 bin
        assert sampler.num_bins == 1
        assert len(sampler.bin_indices[0]) == 50

    def test_bin_coverage(self):
        """Test that bins cover the entire target range."""
        targets = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        sampler = RegressionBatchSampler(targets, batch_size=5, num_bins=2)

        # Collect all indices from all bins
        all_indices = []
        for bin_list in sampler.bin_indices:
            all_indices.extend(bin_list)

        # All samples should be assigned to some bin
        assert len(all_indices) == 5
        assert set(all_indices) == set(range(5))


class TestRegressionBatchSamplerBatchGeneration:
    """Test batch generation functionality."""

    def test_batch_size_correctness(self):
        """Test that generated batches have correct size."""
        targets = torch.linspace(0, 1, 100)
        sampler = RegressionBatchSampler(targets, batch_size=20, num_bins=5)

        for batch in sampler:
            assert len(batch) <= 20

    def test_target_diversity_in_batch(self):
        """Test that batches contain diverse target values."""
        # Create targets with clear bins
        targets = torch.cat(
            [
                torch.ones(20) * 0.1,  # Bin 1
                torch.ones(20) * 0.5,  # Bin 2
                torch.ones(20) * 0.9,  # Bin 3
            ]
        )
        sampler = RegressionBatchSampler(targets, batch_size=15, num_bins=3)

        # Generate first batch
        batch = list(next(iter(sampler)))
        batch_targets = targets[batch]

        # Batch should contain targets from multiple bins
        unique_targets = torch.unique(batch_targets)
        assert len(unique_targets) >= 2  # At least 2 different target values

    def test_len_method(self):
        """Test __len__ method returns correct number of batches."""
        targets = torch.linspace(0, 1, 100)
        sampler = RegressionBatchSampler(targets, batch_size=20, num_bins=5)

        num_batches = len(sampler)
        actual_batches = len(list(sampler))

        assert num_batches == actual_batches

    def test_repr_method(self):
        """Test __repr__ method."""
        targets = torch.linspace(0, 1, 50)
        sampler = RegressionBatchSampler(targets, batch_size=10, num_bins=5)

        repr_str = repr(sampler)
        assert "RegressionBatchSampler" in repr_str
        assert "batch_size=10" in repr_str


# ============================================================================
# SurvivalBatchSampler Tests
# ============================================================================


class TestSurvivalBatchSamplerInitialization:
    """Test SurvivalBatchSampler initialization and configuration validation."""

    def test_valid_initialization(self):
        """Test initialization with valid parameters."""
        times = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        events = torch.tensor([1, 1, 0, 1, 0, 0])
        sampler = SurvivalBatchSampler(times, events, batch_size=4)

        assert sampler.batch_size == 4
        assert sampler.num_samples == 6

    def test_initialization_with_lists(self):
        """Test initialization with Python lists instead of tensors."""
        times = [1.0, 2.0, 3.0, 4.0]
        events = [1, 1, 0, 0]
        sampler = SurvivalBatchSampler(times, events, batch_size=4)

        assert sampler.num_samples == 4

    def test_empty_times_raises_error(self):
        """Test that empty times raises ValueError."""
        with pytest.raises(ValueError, match="times cannot be empty"):
            SurvivalBatchSampler([], [1, 0], batch_size=2)

    def test_empty_events_raises_error(self):
        """Test that empty events raises ValueError."""
        with pytest.raises(ValueError, match="events cannot be empty"):
            SurvivalBatchSampler([1.0, 2.0], [], batch_size=2)

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched times and events lengths raises ValueError."""
        times = [1.0, 2.0, 3.0]
        events = [1, 0]
        with pytest.raises(ValueError, match="times and events must have same length"):
            SurvivalBatchSampler(times, events, batch_size=2)

    def test_negative_batch_size_raises_error(self):
        """Test that negative batch_size raises ValueError."""
        times = [1.0, 2.0]
        events = [1, 0]
        with pytest.raises(ValueError, match="batch_size must be positive"):
            SurvivalBatchSampler(times, events, batch_size=-1)

    def test_invalid_event_values_raises_error(self):
        """Test that events with values other than 0 or 1 raises ValueError."""
        times = [1.0, 2.0, 3.0]
        events = [1, 0, 2]  # Invalid: 2
        with pytest.raises(ValueError, match="events must contain only 0 or 1"):
            SurvivalBatchSampler(times, events, batch_size=3)

    def test_default_batch_size(self):
        """Test default batch_size parameter."""
        times = [1.0, 2.0]
        events = [1, 0]
        sampler = SurvivalBatchSampler(times, events)

        assert sampler.batch_size == 32


class TestSurvivalBatchSamplerEventBalancing:
    """Test event balancing functionality."""

    def test_balanced_events(self):
        """Test with balanced events and censored samples."""
        # 3 events, 3 censored
        times = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        events = torch.tensor([1, 1, 1, 0, 0, 0])
        sampler = SurvivalBatchSampler(times, events, batch_size=4)

        # Should split batch evenly: 2 events, 2 censored
        assert sampler.samples_per_event == 2
        assert sampler.samples_per_censored == 2

        # Generate first batch
        batch = list(next(iter(sampler)))
        batch_events = events[batch]

        # Count events and censored in batch
        num_events = (batch_events == 1).sum().item()
        num_censored = (batch_events == 0).sum().item()

        # Should be approximately balanced
        assert num_events >= 1 and num_events <= 3
        assert num_censored >= 1 and num_censored <= 3

    def test_only_events(self):
        """Test with only event samples (no censored)."""
        times = torch.tensor([1.0, 2.0, 3.0, 4.0])
        events = torch.tensor([1, 1, 1, 1])
        sampler = SurvivalBatchSampler(times, events, batch_size=2)

        assert sampler.samples_per_event == 2
        assert sampler.samples_per_censored == 0
        assert len(sampler.event_indices) == 4
        assert len(sampler.censored_indices) == 0

    def test_only_censored(self):
        """Test with only censored samples (no events)."""
        times = torch.tensor([1.0, 2.0, 3.0, 4.0])
        events = torch.tensor([0, 0, 0, 0])
        sampler = SurvivalBatchSampler(times, events, batch_size=2)

        assert sampler.samples_per_event == 0
        assert sampler.samples_per_censored == 2
        assert len(sampler.event_indices) == 0
        assert len(sampler.censored_indices) == 4

    def test_imbalanced_events(self):
        """Test with imbalanced events (more censored than events)."""
        # 2 events, 8 censored
        times = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        events = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        sampler = SurvivalBatchSampler(times, events, batch_size=6)

        # Should split evenly: 3 events, 3 censored
        assert sampler.samples_per_event == 3
        assert sampler.samples_per_censored == 3

        # Since we only have 2 events, they will be oversampled
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)

        all_events = events[all_indices]
        event_count = (all_events == 1).sum().item()

        # Events should be oversampled (appear more than 2 times)
        assert event_count > 2


class TestSurvivalBatchSamplerTemporalDiversity:
    """Test temporal diversity in batches."""

    def test_temporal_sorting(self):
        """Test that samples are sorted by time within event groups."""
        times = torch.tensor([5.0, 1.0, 3.0, 6.0, 2.0, 4.0])
        events = torch.tensor([1, 1, 1, 0, 0, 0])
        sampler = SurvivalBatchSampler(times, events, batch_size=6)

        # Event indices should be sorted by time
        event_times = times[sampler.event_indices]
        assert torch.all(event_times[:-1] <= event_times[1:])  # Ascending order

        # Censored indices should be sorted by time
        censored_times = times[sampler.censored_indices]
        assert torch.all(censored_times[:-1] <= censored_times[1:])  # Ascending order

    def test_temporal_diversity_in_batch(self):
        """Test that batches contain diverse survival times."""
        # Create samples with clear temporal separation
        times = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        events = torch.tensor([1, 1, 1, 0, 0, 0])
        sampler = SurvivalBatchSampler(times, events, batch_size=6)

        # Generate first batch
        batch = list(next(iter(sampler)))
        batch_times = times[batch]

        # Batch should contain diverse times (not all early or all late)
        time_range = batch_times.max() - batch_times.min()
        assert time_range > 5.0  # Significant temporal spread


class TestSurvivalBatchSamplerBatchGeneration:
    """Test batch generation functionality."""

    def test_batch_size_correctness(self):
        """Test that generated batches have correct size."""
        times = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        events = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
        sampler = SurvivalBatchSampler(times, events, batch_size=4)

        for batch in sampler:
            assert len(batch) <= 4

    def test_len_method(self):
        """Test __len__ method returns correct number of batches."""
        times = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        events = torch.tensor([1, 1, 1, 0, 0, 0])
        sampler = SurvivalBatchSampler(times, events, batch_size=4)

        num_batches = len(sampler)
        actual_batches = len(list(sampler))

        assert num_batches == actual_batches

    def test_repr_method(self):
        """Test __repr__ method."""
        times = torch.tensor([1.0, 2.0, 3.0, 4.0])
        events = torch.tensor([1, 1, 0, 0])
        sampler = SurvivalBatchSampler(times, events, batch_size=4)

        repr_str = repr(sampler)
        assert "SurvivalBatchSampler" in repr_str
        assert "num_events=2" in repr_str
        assert "num_censored=2" in repr_str
        assert "batch_size=4" in repr_str


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for all samplers."""

    def test_all_samplers_with_dataloader(self):
        """Test that all samplers work with PyTorch DataLoader."""
        # This is a basic integration test to ensure samplers are compatible
        # with PyTorch's DataLoader interface

        # Create dummy dataset
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return idx

        dataset = DummyDataset(100)

        # Test BalancedBatchSampler
        labels = torch.randint(0, 3, (100,))
        balanced_sampler = BalancedBatchSampler(labels, batch_size=16)
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=balanced_sampler)
        batches = list(loader)
        assert len(batches) > 0

        # Test RegressionBatchSampler
        targets = torch.rand(100)
        regression_sampler = RegressionBatchSampler(targets, batch_size=16, num_bins=5)
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=regression_sampler)
        batches = list(loader)
        assert len(batches) > 0

        # Test SurvivalBatchSampler
        times = torch.rand(100) * 10
        events = torch.randint(0, 2, (100,))
        survival_sampler = SurvivalBatchSampler(times, events, batch_size=16)
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=survival_sampler)
        batches = list(loader)
        assert len(batches) > 0
