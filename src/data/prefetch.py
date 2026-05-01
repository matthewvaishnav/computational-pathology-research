"""
Data prefetching utilities for improved data loading performance.

Implements background prefetching to overlap data loading with GPU computation,
reducing training bottlenecks from I/O operations.
"""

import queue
import threading
from typing import Iterator, Optional

import torch


class DataPrefetcher:
    """
    Prefetch data batches in background thread to overlap I/O with computation.
    
    Loads next batch while GPU processes current batch, hiding data loading latency.
    
    Usage:
        prefetcher = DataPrefetcher(dataloader, device='cuda')
        for batch in prefetcher:
            # Process batch - next batch loading in background
            outputs = model(batch)
    
    Args:
        loader: PyTorch DataLoader
        device: Target device for prefetched data
        prefetch_count: Number of batches to prefetch (default: 1)
    """

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        prefetch_count: int = 1,
    ):
        self.loader = loader
        self.device = device
        self.prefetch_count = prefetch_count
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self) -> Iterator:
        """Iterate over prefetched batches."""
        loader_iter = iter(self.loader)
        
        # Prefetch first batch
        try:
            next_batch = next(loader_iter)
            next_batch = self._move_to_device(next_batch)
        except StopIteration:
            return

        for batch in loader_iter:
            # Wait for current batch to finish transferring
            if self.stream is not None:
                torch.cuda.current_stream().wait_stream(self.stream)

            current_batch = next_batch

            # Start prefetching next batch in background
            next_batch = self._move_to_device(batch)

            yield current_batch

        # Yield last batch
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        yield next_batch

    def _move_to_device(self, batch):
        """Move batch to device asynchronously."""
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                batch = self._recursive_to_device(batch)
        else:
            batch = self._recursive_to_device(batch)
        return batch

    def _recursive_to_device(self, data):
        """Recursively move data structures to device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: self._recursive_to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._recursive_to_device(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._recursive_to_device(item) for item in data)
        else:
            return data

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)


class BackgroundPrefetcher:
    """
    Advanced prefetcher using background thread and queue.
    
    Maintains queue of prefetched batches for smoother data flow.
    Useful when data loading is highly variable or expensive.
    
    Args:
        loader: PyTorch DataLoader
        device: Target device
        queue_size: Size of prefetch queue (default: 2)
    """

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        queue_size: int = 2,
    ):
        self.loader = loader
        self.device = device
        self.queue_size = queue_size
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def _prefetch_worker(self):
        """Background worker that prefetches batches."""
        try:
            for batch in self.loader:
                if self.stop_event.is_set():
                    break

                # Move to device
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(self.device, non_blocking=True)
                        if isinstance(v, torch.Tensor)
                        else v
                        for k, v in batch.items()
                    }
                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device, non_blocking=True)

                # Add to queue (blocks if queue is full)
                self.queue.put(batch)

        except Exception as e:
            self.queue.put(e)
        finally:
            # Signal end of iteration
            self.queue.put(None)

    def __iter__(self) -> Iterator:
        """Iterate over prefetched batches."""
        # Start prefetch thread
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()

        # Yield batches from queue
        while True:
            batch = self.queue.get()

            # Check for end of iteration
            if batch is None:
                break

            # Check for errors
            if isinstance(batch, Exception):
                raise batch

            yield batch

        # Wait for thread to finish
        if self.thread is not None:
            self.thread.join()

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)

    def shutdown(self):
        """Stop prefetching and clean up resources."""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)


def create_optimized_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader with optimized settings for performance.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        num_workers: Number of worker processes (default: 4)
        pin_memory: Pin memory for faster GPU transfer (default: True)
        prefetch_factor: Batches to prefetch per worker (default: 2)
        persistent_workers: Keep workers alive between epochs (default: True)
        **kwargs: Additional DataLoader arguments
    
    Returns:
        Optimized DataLoader
    
    Performance tips:
        - num_workers: Set to 2-4x number of GPUs
        - pin_memory: Always True for GPU training
        - prefetch_factor: 2-4 for good overlap
        - persistent_workers: True to avoid worker respawn overhead
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        **kwargs,
    )
