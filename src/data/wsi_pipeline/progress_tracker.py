"""
Progress tracking and monitoring for WSI processing pipeline.

This module provides comprehensive progress tracking, ETA calculation,
and monitoring capabilities for batch processing operations.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for processing operations."""

    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Performance metrics
    patches_processed: int = 0
    total_processing_time: float = 0.0
    total_file_size_mb: float = 0.0

    # Error tracking
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.failed_items / self.total_items) * 100

    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Calculate elapsed time."""
        if self.start_time is None:
            return None

        end_time = self.end_time or datetime.now()
        return end_time - self.start_time

    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time per item."""
        if self.completed_items == 0:
            return 0.0
        return self.total_processing_time / self.completed_items

    @property
    def patches_per_second(self) -> float:
        """Calculate patches processed per second."""
        if self.total_processing_time == 0:
            return 0.0
        return self.patches_processed / self.total_processing_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "patches_processed": self.patches_processed,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.average_processing_time,
            "patches_per_second": self.patches_per_second,
            "total_file_size_mb": self.total_file_size_mb,
            "elapsed_time": str(self.elapsed_time) if self.elapsed_time else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_count": len(self.errors),
        }


class ProgressTracker:
    """
    Advanced progress tracker with ETA calculation and monitoring.

    Provides real-time progress tracking, ETA calculation, and performance
    monitoring for WSI processing operations.
    """

    def __init__(
        self,
        total_items: int,
        description: str = "Processing",
        show_progress_bar: bool = True,
        log_interval: int = 10,
    ):
        """
        Initialize progress tracker.

        Args:
            total_items: Total number of items to process
            description: Description for progress display
            show_progress_bar: Whether to show tqdm progress bar
            log_interval: Interval for logging progress (in items)
        """
        self.total_items = total_items
        self.description = description
        self.show_progress_bar = show_progress_bar
        self.log_interval = log_interval

        # Initialize statistics
        self.stats = ProcessingStats(total_items=total_items)

        # Progress tracking
        self.current_item = 0
        self.last_log_item = 0

        # Performance tracking
        self.item_start_times = {}
        self.recent_processing_times = []
        self.max_recent_times = 50  # Keep last 50 processing times for ETA

        # Progress bar
        self.progress_bar = None
        if self.show_progress_bar:
            self.progress_bar = tqdm(
                total=total_items,
                desc=description,
                unit="items",
                ncols=100,
            )

        logger.info(f"Started progress tracking: {description} ({total_items} items)")

    def start(self) -> None:
        """Start progress tracking."""
        self.stats.start_time = datetime.now()
        logger.info(f"Processing started at {self.stats.start_time}")

    def start_item(self, item_id: str) -> None:
        """
        Mark the start of processing an item.

        Args:
            item_id: Unique identifier for the item
        """
        self.item_start_times[item_id] = time.time()

    def complete_item(
        self,
        item_id: str,
        patches_processed: int = 0,
        file_size_mb: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """
        Mark an item as completed.

        Args:
            item_id: Unique identifier for the item
            patches_processed: Number of patches processed for this item
            file_size_mb: Output file size in MB
            error: Error message if item failed
        """
        # Calculate processing time
        processing_time = 0.0
        if item_id in self.item_start_times:
            processing_time = time.time() - self.item_start_times[item_id]
            del self.item_start_times[item_id]

            # Update recent processing times for ETA calculation
            self.recent_processing_times.append(processing_time)
            if len(self.recent_processing_times) > self.max_recent_times:
                self.recent_processing_times.pop(0)

        # Update statistics
        if error:
            self.stats.failed_items += 1
            self.stats.errors.append(f"{item_id}: {error}")
            logger.warning(f"Item {item_id} failed: {error}")
        else:
            self.stats.completed_items += 1
            self.stats.patches_processed += patches_processed
            self.stats.total_file_size_mb += file_size_mb
            logger.debug(f"Item {item_id} completed in {processing_time:.2f}s")

        self.stats.total_processing_time += processing_time
        self.current_item += 1

        # Update progress bar
        if self.progress_bar:
            self.progress_bar.update(1)
            self.progress_bar.set_postfix(
                {
                    "Success": f"{self.stats.success_rate:.1f}%",
                    "ETA": self.get_eta_string(),
                }
            )

        # Log progress at intervals
        if (self.current_item - self.last_log_item) >= self.log_interval:
            self.log_progress()
            self.last_log_item = self.current_item

    def get_eta(self) -> Optional[timedelta]:
        """
        Calculate estimated time to completion.

        Returns:
            Estimated time remaining or None if cannot calculate
        """
        if not self.recent_processing_times or self.current_item == 0:
            return None

        remaining_items = self.total_items - self.current_item
        if remaining_items <= 0:
            return timedelta(0)

        # Use average of recent processing times
        avg_time_per_item = sum(self.recent_processing_times) / len(self.recent_processing_times)
        eta_seconds = remaining_items * avg_time_per_item

        return timedelta(seconds=eta_seconds)

    def get_eta_string(self) -> str:
        """
        Get ETA as formatted string.

        Returns:
            Formatted ETA string
        """
        eta = self.get_eta()
        if eta is None:
            return "Unknown"

        total_seconds = int(eta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    def log_progress(self) -> None:
        """Log current progress statistics."""
        progress_pct = (self.current_item / self.total_items) * 100
        eta_str = self.get_eta_string()

        logger.info(
            f"Progress: {self.current_item}/{self.total_items} "
            f"({progress_pct:.1f}%) - "
            f"Success: {self.stats.success_rate:.1f}% - "
            f"ETA: {eta_str}"
        )

    def finish(self) -> ProcessingStats:
        """
        Finish progress tracking and return final statistics.

        Returns:
            Final processing statistics
        """
        self.stats.end_time = datetime.now()

        if self.progress_bar:
            self.progress_bar.close()

        # Log final statistics
        elapsed = self.stats.elapsed_time
        logger.info(
            f"Processing completed: {self.stats.completed_items}/{self.total_items} "
            f"successful ({self.stats.success_rate:.1f}%) in {elapsed}"
        )

        if self.stats.failed_items > 0:
            logger.warning(
                f"Failed items: {self.stats.failed_items} ({self.stats.failure_rate:.1f}%)"
            )

        if self.stats.patches_processed > 0:
            logger.info(
                f"Performance: {self.stats.patches_processed} patches processed "
                f"({self.stats.patches_per_second:.1f} patches/sec)"
            )

        return self.stats

    def get_current_stats(self) -> ProcessingStats:
        """
        Get current statistics without finishing.

        Returns:
            Current processing statistics
        """
        return self.stats


class BatchProgressMonitor:
    """
    Monitor for batch processing operations with multiple progress trackers.

    Manages multiple concurrent progress trackers and provides
    aggregate monitoring capabilities.
    """

    def __init__(self):
        """Initialize batch progress monitor."""
        self.trackers: Dict[str, ProgressTracker] = {}
        self.start_time = None

    def create_tracker(
        self, tracker_id: str, total_items: int, description: str = "Processing", **kwargs
    ) -> ProgressTracker:
        """
        Create a new progress tracker.

        Args:
            tracker_id: Unique identifier for the tracker
            total_items: Total number of items to process
            description: Description for progress display
            **kwargs: Additional arguments for ProgressTracker

        Returns:
            Created progress tracker
        """
        tracker = ProgressTracker(total_items=total_items, description=description, **kwargs)

        self.trackers[tracker_id] = tracker

        if self.start_time is None:
            self.start_time = datetime.now()

        return tracker

    def get_tracker(self, tracker_id: str) -> Optional[ProgressTracker]:
        """
        Get existing progress tracker.

        Args:
            tracker_id: Tracker identifier

        Returns:
            Progress tracker or None if not found
        """
        return self.trackers.get(tracker_id)

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all trackers.

        Returns:
            Aggregate statistics dictionary
        """
        total_items = sum(t.total_items for t in self.trackers.values())
        completed_items = sum(t.stats.completed_items for t in self.trackers.values())
        failed_items = sum(t.stats.failed_items for t in self.trackers.values())
        patches_processed = sum(t.stats.patches_processed for t in self.trackers.values())
        total_processing_time = sum(t.stats.total_processing_time for t in self.trackers.values())

        return {
            "total_trackers": len(self.trackers),
            "total_items": total_items,
            "completed_items": completed_items,
            "failed_items": failed_items,
            "success_rate": (completed_items / total_items * 100) if total_items > 0 else 0,
            "patches_processed": patches_processed,
            "patches_per_second": (
                (patches_processed / total_processing_time) if total_processing_time > 0 else 0
            ),
            "elapsed_time": str(datetime.now() - self.start_time) if self.start_time else None,
        }

    def log_aggregate_progress(self) -> None:
        """Log aggregate progress across all trackers."""
        stats = self.get_aggregate_stats()

        logger.info(
            f"Batch Progress: {stats['completed_items']}/{stats['total_items']} "
            f"({stats['success_rate']:.1f}%) - "
            f"{stats['patches_processed']} patches "
            f"({stats['patches_per_second']:.1f} patches/sec)"
        )

    def finish_all(self) -> Dict[str, ProcessingStats]:
        """
        Finish all trackers and return final statistics.

        Returns:
            Dictionary of final statistics for each tracker
        """
        results = {}

        for tracker_id, tracker in self.trackers.items():
            results[tracker_id] = tracker.finish()

        # Log final aggregate statistics
        aggregate_stats = self.get_aggregate_stats()
        logger.info(f"Batch processing completed: {aggregate_stats}")

        return results


# Example usage
if __name__ == "__main__":
    import random

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Test single progress tracker
    tracker = ProgressTracker(
        total_items=10,
        description="Test Processing",
        show_progress_bar=True,
        log_interval=3,
    )

    tracker.start()

    for i in range(10):
        item_id = f"item_{i}"
        tracker.start_item(item_id)

        # Simulate processing
        time.sleep(random.uniform(0.1, 0.5))

        # Simulate occasional failures
        error = "Test error" if random.random() < 0.1 else None

        tracker.complete_item(
            item_id=item_id,
            patches_processed=random.randint(50, 200),
            file_size_mb=random.uniform(1.0, 10.0),
            error=error,
        )

    final_stats = tracker.finish()
    print(f"Final stats: {final_stats.to_dict()}")
