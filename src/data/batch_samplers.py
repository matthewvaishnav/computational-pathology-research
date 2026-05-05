"""
Task-aware batch samplers for Multiple Instance Learning.

This module implements task-specific batch samplers from nnMIL (Stanford/NIH 2024)
that ensure balanced representation across different task types:
- BalancedBatchSampler: Equal class representation for classification
- RegressionBatchSampler: Uniform coverage of target range for regression
- SurvivalBatchSampler: Balanced event rates for survival analysis

These samplers address class imbalance and improve training stability by
ensuring each batch contains diverse samples across the target distribution.
"""

import logging
from typing import List

import torch
import torch.utils.data

logger = logging.getLogger(__name__)


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Balanced batch sampler for classification tasks.
    
    Ensures approximately equal representation of each class within batches
    by oversampling minority classes. This addresses class imbalance and
    improves training stability for imbalanced datasets.
    
    The sampler works by:
    1. Grouping samples by class label
    2. For each batch, sampling approximately batch_size/num_classes samples
       from each class
    3. Oversampling minority classes (with replacement) to match majority class
    4. Shuffling samples within each class if shuffle=True
    
    Args:
        labels: Array-like of class labels [N] where N is dataset size
        batch_size: Number of samples per batch (default: 32)
        shuffle: Whether to shuffle samples within each class (default: True)
    
    Raises:
        ValueError: If labels is empty
        ValueError: If batch_size is not positive
    
    Example:
        >>> labels = [0, 0, 0, 0, 1, 1, 2]  # Imbalanced: 4 class-0, 2 class-1, 1 class-2
        >>> sampler = BalancedBatchSampler(labels, batch_size=6, shuffle=False)
        >>> batch = list(next(iter(sampler)))
        >>> len(batch)
        6
        >>> # Each class appears approximately equally (2 samples per class)
    """
    
    def __init__(
        self,
        labels: torch.Tensor,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        """
        Initialize BalancedBatchSampler.
        
        Args:
            labels: Class labels [N]
            batch_size: Batch size (default: 32)
            shuffle: Shuffle within classes (default: True)
        """
        # Validate inputs
        if len(labels) == 0:
            raise ValueError("labels cannot be empty")
        
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        # Convert to tensor if needed
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by class
        self.class_indices = {}
        unique_labels = torch.unique(labels)
        
        for label in unique_labels:
            label_item = label.item()
            self.class_indices[label_item] = torch.where(labels == label)[0].tolist()
        
        self.num_classes = len(self.class_indices)
        self.num_samples = len(labels)
        
        # Calculate samples per class per batch
        self.samples_per_class = max(1, batch_size // self.num_classes)
        
        # Calculate number of batches
        # Each batch samples samples_per_class from each class
        max_class_size = max(len(indices) for indices in self.class_indices.values())
        self.num_batches = (max_class_size + self.samples_per_class - 1) // self.samples_per_class
        
        logger.debug(
            f"Initialized BalancedBatchSampler: "
            f"num_classes={self.num_classes}, "
            f"batch_size={batch_size}, "
            f"samples_per_class={self.samples_per_class}, "
            f"num_batches={self.num_batches}"
        )
    
    def __iter__(self):
        """
        Generate balanced batches.
        
        Yields:
            List of indices for each batch
        """
        # Create iterators for each class
        class_iterators = {}
        
        for label, indices in self.class_indices.items():
            if self.shuffle:
                # Shuffle indices within class
                perm = torch.randperm(len(indices)).tolist()
                shuffled_indices = [indices[i] for i in perm]
            else:
                shuffled_indices = indices.copy()
            
            # Create cycling iterator that repeats when exhausted (oversampling)
            class_iterators[label] = self._cycle_iterator(shuffled_indices)
        
        # Generate batches
        for _ in range(self.num_batches):
            batch = []
            
            # Sample from each class
            for label in sorted(self.class_indices.keys()):
                iterator = class_iterators[label]
                for _ in range(self.samples_per_class):
                    batch.append(next(iterator))
            
            # Trim to exact batch size if needed
            batch = batch[:self.batch_size]
            
            # Shuffle batch if requested
            if self.shuffle:
                perm = torch.randperm(len(batch)).tolist()
                batch = [batch[i] for i in perm]
            
            yield batch
    
    def _cycle_iterator(self, items: List[int]):
        """
        Create an iterator that cycles through items indefinitely.
        
        Args:
            items: List of items to cycle through
        
        Yields:
            Items from the list, cycling back to start when exhausted
        """
        while True:
            for item in items:
                yield item
    
    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches
    
    def __repr__(self) -> str:
        """String representation of the sampler."""
        return (
            f"BalancedBatchSampler("
            f"num_classes={self.num_classes}, "
            f"batch_size={self.batch_size}, "
            f"shuffle={self.shuffle})"
        )


class RegressionBatchSampler(torch.utils.data.Sampler):
    """
    Binned batch sampler for regression tasks.
    
    Divides the target range into bins and samples proportionally from each bin
    to achieve uniform coverage of the target distribution. This prevents the
    model from focusing only on common target values and ensures representation
    across the entire target range.
    
    The sampler works by:
    1. Dividing target range [min, max] into num_bins equal-width bins
    2. Assigning each sample to a bin based on its target value
    3. For each batch, sampling proportionally from each bin
    4. Oversampling bins with fewer samples to maintain balance
    
    Args:
        targets: Array-like of regression targets [N] where N is dataset size
        batch_size: Number of samples per batch (default: 32)
        num_bins: Number of bins for target discretization (default: 10)
    
    Raises:
        ValueError: If targets is empty
        ValueError: If batch_size is not positive
        ValueError: If num_bins is not positive
    
    Example:
        >>> targets = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]  # Bimodal distribution
        >>> sampler = RegressionBatchSampler(targets, batch_size=4, num_bins=2)
        >>> batch = list(next(iter(sampler)))
        >>> len(batch)
        4
        >>> # Batch contains samples from both low (0.1-0.5) and high (0.5-1.0) ranges
    """
    
    def __init__(
        self,
        targets: torch.Tensor,
        batch_size: int = 32,
        num_bins: int = 10
    ):
        """
        Initialize RegressionBatchSampler.
        
        Args:
            targets: Regression targets [N]
            batch_size: Batch size (default: 32)
            num_bins: Number of bins (default: 10)
        """
        # Validate inputs
        if len(targets) == 0:
            raise ValueError("targets cannot be empty")
        
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}")
        
        # Convert to tensor if needed
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.float32)
        
        self.targets = targets
        self.batch_size = batch_size
        self.num_bins = num_bins
        self.num_samples = len(targets)
        
        # Create bins
        min_target = targets.min().item()
        max_target = targets.max().item()
        
        # Handle edge case where all targets are the same
        if min_target == max_target:
            bin_edges = torch.tensor([min_target, min_target + 1e-6])
            self.num_bins = 1
        else:
            bin_edges = torch.linspace(min_target, max_target, num_bins + 1)
        
        # Assign samples to bins
        self.bin_indices = [[] for _ in range(self.num_bins)]
        
        for idx, target in enumerate(targets):
            # Find which bin this target belongs to
            # Use searchsorted to find the bin index
            bin_idx = torch.searchsorted(bin_edges[1:], target).item()
            bin_idx = min(bin_idx, self.num_bins - 1)  # Clamp to valid range
            self.bin_indices[bin_idx].append(idx)
        
        # Remove empty bins
        self.bin_indices = [bin_list for bin_list in self.bin_indices if len(bin_list) > 0]
        self.num_bins = len(self.bin_indices)
        
        # Calculate samples per bin per batch
        self.samples_per_bin = max(1, batch_size // self.num_bins)
        
        # Calculate number of batches
        max_bin_size = max(len(bin_list) for bin_list in self.bin_indices)
        self.num_batches = (max_bin_size + self.samples_per_bin - 1) // self.samples_per_bin
        
        logger.debug(
            f"Initialized RegressionBatchSampler: "
            f"num_bins={self.num_bins}, "
            f"batch_size={batch_size}, "
            f"samples_per_bin={self.samples_per_bin}, "
            f"target_range=[{min_target:.3f}, {max_target:.3f}]"
        )
    
    def __iter__(self):
        """
        Generate binned batches.
        
        Yields:
            List of indices for each batch
        """
        # Create iterators for each bin
        bin_iterators = []
        
        for bin_list in self.bin_indices:
            # Shuffle indices within bin
            perm = torch.randperm(len(bin_list)).tolist()
            shuffled_indices = [bin_list[i] for i in perm]
            
            # Create cycling iterator that repeats when exhausted
            bin_iterators.append(self._cycle_iterator(shuffled_indices))
        
        # Generate batches
        for _ in range(self.num_batches):
            batch = []
            
            # Sample from each bin
            for iterator in bin_iterators:
                for _ in range(self.samples_per_bin):
                    batch.append(next(iterator))
            
            # Trim to exact batch size if needed
            batch = batch[:self.batch_size]
            
            # Shuffle batch
            perm = torch.randperm(len(batch)).tolist()
            batch = [batch[i] for i in perm]
            
            yield batch
    
    def _cycle_iterator(self, items: List[int]):
        """
        Create an iterator that cycles through items indefinitely.
        
        Args:
            items: List of items to cycle through
        
        Yields:
            Items from the list, cycling back to start when exhausted
        """
        while True:
            for item in items:
                yield item
    
    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches
    
    def __repr__(self) -> str:
        """String representation of the sampler."""
        return (
            f"RegressionBatchSampler("
            f"num_bins={self.num_bins}, "
            f"batch_size={self.batch_size})"
        )


class SurvivalBatchSampler(torch.utils.data.Sampler):
    """
    Event-balanced batch sampler for survival analysis.
    
    Maintains balanced event rates and temporal distributions across batches.
    This ensures the model sees both censored and event samples, as well as
    both early and late events, in each batch.
    
    The sampler works by:
    1. Dividing samples into event (event=1) and censored (event=0) groups
    2. Within each group, sorting by survival time
    3. For each batch, sampling proportionally from both groups
    4. Ensuring temporal diversity by sampling across the time range
    
    Args:
        times: Array-like of survival times [N] where N is dataset size
        events: Array-like of event indicators [N] (1=event occurred, 0=censored)
        batch_size: Number of samples per batch (default: 32)
    
    Raises:
        ValueError: If times or events is empty
        ValueError: If times and events have different lengths
        ValueError: If batch_size is not positive
        ValueError: If events contains values other than 0 or 1
    
    Example:
        >>> times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        >>> events = [1, 1, 0, 1, 0, 0]  # 3 events, 3 censored
        >>> sampler = SurvivalBatchSampler(times, events, batch_size=4)
        >>> batch = list(next(iter(sampler)))
        >>> len(batch)
        4
        >>> # Batch contains mix of events and censored samples
    """
    
    def __init__(
        self,
        times: torch.Tensor,
        events: torch.Tensor,
        batch_size: int = 32
    ):
        """
        Initialize SurvivalBatchSampler.
        
        Args:
            times: Survival times [N]
            events: Event indicators [N] (1=event, 0=censored)
            batch_size: Batch size (default: 32)
        """
        # Validate inputs
        if len(times) == 0:
            raise ValueError("times cannot be empty")
        
        if len(events) == 0:
            raise ValueError("events cannot be empty")
        
        if len(times) != len(events):
            raise ValueError(
                f"times and events must have same length, "
                f"got {len(times)} and {len(events)}"
            )
        
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        # Convert to tensors if needed
        if not isinstance(times, torch.Tensor):
            times = torch.tensor(times, dtype=torch.float32)
        
        if not isinstance(events, torch.Tensor):
            events = torch.tensor(events, dtype=torch.long)
        
        # Validate events are binary
        unique_events = torch.unique(events)
        if not all(e in [0, 1] for e in unique_events.tolist()):
            raise ValueError(
                f"events must contain only 0 or 1, got {unique_events.tolist()}"
            )
        
        self.times = times
        self.events = events
        self.batch_size = batch_size
        self.num_samples = len(times)
        
        # Group indices by event status
        event_mask = events == 1
        censored_mask = events == 0
        
        event_indices = torch.where(event_mask)[0].tolist()
        censored_indices = torch.where(censored_mask)[0].tolist()
        
        # Sort by time within each group for temporal diversity
        if len(event_indices) > 0:
            event_times = times[event_indices]
            event_sort_idx = torch.argsort(event_times)
            self.event_indices = [event_indices[i] for i in event_sort_idx.tolist()]
        else:
            self.event_indices = []
        
        if len(censored_indices) > 0:
            censored_times = times[censored_indices]
            censored_sort_idx = torch.argsort(censored_times)
            self.censored_indices = [censored_indices[i] for i in censored_sort_idx.tolist()]
        else:
            self.censored_indices = []
        
        # Calculate samples per group per batch
        # Aim for balanced representation
        num_event = len(self.event_indices)
        num_censored = len(self.censored_indices)
        
        if num_event > 0 and num_censored > 0:
            # Both groups present - split batch evenly
            self.samples_per_event = batch_size // 2
            self.samples_per_censored = batch_size - self.samples_per_event
        elif num_event > 0:
            # Only events
            self.samples_per_event = batch_size
            self.samples_per_censored = 0
        else:
            # Only censored
            self.samples_per_event = 0
            self.samples_per_censored = batch_size
        
        # Calculate number of batches
        if self.samples_per_event > 0:
            event_batches = (len(self.event_indices) + self.samples_per_event - 1) // self.samples_per_event
        else:
            event_batches = 0
        
        if self.samples_per_censored > 0:
            censored_batches = (len(self.censored_indices) + self.samples_per_censored - 1) // self.samples_per_censored
        else:
            censored_batches = 0
        
        self.num_batches = max(event_batches, censored_batches)
        
        logger.debug(
            f"Initialized SurvivalBatchSampler: "
            f"num_events={num_event}, "
            f"num_censored={num_censored}, "
            f"batch_size={batch_size}, "
            f"samples_per_event={self.samples_per_event}, "
            f"samples_per_censored={self.samples_per_censored}"
        )
    
    def __iter__(self):
        """
        Generate event-balanced batches.
        
        Yields:
            List of indices for each batch
        """
        # Create iterators for each group
        if len(self.event_indices) > 0:
            # Shuffle event indices
            perm = torch.randperm(len(self.event_indices)).tolist()
            shuffled_event = [self.event_indices[i] for i in perm]
            event_iterator = self._cycle_iterator(shuffled_event)
        else:
            event_iterator = None
        
        if len(self.censored_indices) > 0:
            # Shuffle censored indices
            perm = torch.randperm(len(self.censored_indices)).tolist()
            shuffled_censored = [self.censored_indices[i] for i in perm]
            censored_iterator = self._cycle_iterator(shuffled_censored)
        else:
            censored_iterator = None
        
        # Generate batches
        for _ in range(self.num_batches):
            batch = []
            
            # Sample from event group
            if event_iterator is not None:
                for _ in range(self.samples_per_event):
                    batch.append(next(event_iterator))
            
            # Sample from censored group
            if censored_iterator is not None:
                for _ in range(self.samples_per_censored):
                    batch.append(next(censored_iterator))
            
            # Trim to exact batch size if needed
            batch = batch[:self.batch_size]
            
            # Shuffle batch
            perm = torch.randperm(len(batch)).tolist()
            batch = [batch[i] for i in perm]
            
            yield batch
    
    def _cycle_iterator(self, items: List[int]):
        """
        Create an iterator that cycles through items indefinitely.
        
        Args:
            items: List of items to cycle through
        
        Yields:
            Items from the list, cycling back to start when exhausted
        """
        while True:
            for item in items:
                yield item
    
    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches
    
    def __repr__(self) -> str:
        """String representation of the sampler."""
        return (
            f"SurvivalBatchSampler("
            f"num_events={len(self.event_indices)}, "
            f"num_censored={len(self.censored_indices)}, "
            f"batch_size={self.batch_size})"
        )
