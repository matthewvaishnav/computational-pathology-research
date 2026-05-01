"""
Safe Threading Utilities for HistoCore

Provides thread-safe patterns and utilities to prevent:
- Deadlocks from lock timeouts
- Memory exhaustion from unbounded queues
- Resource leaks from daemon threads
- Race conditions on shared state

Usage:
    from src.utils.safe_threading import (
        TimeoutLock, BoundedQueue, GracefulThread,
        ThreadSafeDict, ThreadSafeSet
    )
"""

import logging
import queue
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generic, Optional, Set, TypeVar

from src.exceptions import ThreadingError

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class TimeoutLock:
    """
    Lock with timeout to prevent deadlocks.
    
    Usage:
        lock = TimeoutLock(timeout=30.0)
        
        with lock:
            # Critical section
            pass
    
    Raises:
        TimeoutError: If lock cannot be acquired within timeout
    """
    
    def __init__(self, timeout: float = 30.0, name: str = "unnamed"):
        """
        Initialize timeout lock.
        
        Args:
            timeout: Maximum seconds to wait for lock acquisition
            name: Lock name for debugging
        """
        self._lock = threading.Lock()
        self.timeout = timeout
        self.name = name
        self._owner = None
        self._acquire_time = None
    
    def __enter__(self):
        """Acquire lock with timeout."""
        acquired = self._lock.acquire(timeout=self.timeout)
        
        if not acquired:
            raise TimeoutError(
                f"Failed to acquire lock '{self.name}' within {self.timeout}s"
            )
        
        self._owner = threading.current_thread().name
        self._acquire_time = time.time()
        
        logger.debug(f"Lock '{self.name}' acquired by {self._owner}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock."""
        hold_time = time.time() - self._acquire_time if self._acquire_time else 0
        
        if hold_time > 5.0:  # Warn if lock held > 5 seconds
            logger.warning(
                f"Lock '{self.name}' held for {hold_time:.2f}s by {self._owner}"
            )
        
        self._owner = None
        self._acquire_time = None
        self._lock.release()
        
        logger.debug(f"Lock '{self.name}' released")


class BoundedQueue(Generic[T]):
    """
    Queue with bounded size to prevent memory exhaustion.
    
    Features:
    - Configurable maxsize
    - Backpressure when full
    - Drop policy options (oldest, newest, block)
    
    Usage:
        queue = BoundedQueue(maxsize=1000, drop_policy='oldest')
        
        # Producer
        queue.put(item, timeout=1.0)
        
        # Consumer
        item = queue.get(timeout=1.0)
    """
    
    def __init__(
        self,
        maxsize: int = 1000,
        drop_policy: str = 'block',
        name: str = "unnamed"
    ):
        """
        Initialize bounded queue.
        
        Args:
            maxsize: Maximum queue size
            drop_policy: What to do when full ('block', 'oldest', 'newest')
            name: Queue name for debugging
        """
        if drop_policy not in ['block', 'oldest', 'newest']:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")
        
        self._queue = queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self.drop_policy = drop_policy
        self.name = name
        self._dropped_count = 0
        self._lock = threading.Lock()
    
    def put(self, item: T, timeout: Optional[float] = None) -> bool:
        """
        Put item in queue with drop policy.
        
        Args:
            item: Item to add
            timeout: Timeout in seconds (None = wait forever)
        
        Returns:
            True if item added, False if dropped
        """
        try:
            self._queue.put(item, timeout=timeout)
            return True
        
        except queue.Full:
            if self.drop_policy == 'block':
                # Re-raise to let caller handle
                raise
            
            elif self.drop_policy == 'oldest':
                # Drop oldest item and add new one
                with self._lock:
                    try:
                        self._queue.get_nowait()  # Remove oldest
                        self._queue.put_nowait(item)  # Add new
                        self._dropped_count += 1
                        
                        if self._dropped_count % 100 == 0:
                            logger.warning(
                                f"Queue '{self.name}' dropped {self._dropped_count} items"
                            )
                        
                        return True
                    except (queue.Empty, queue.Full):
                        return False
            
            elif self.drop_policy == 'newest':
                # Drop new item
                self._dropped_count += 1
                
                if self._dropped_count % 100 == 0:
                    logger.warning(
                        f"Queue '{self.name}' dropped {self._dropped_count} items"
                    )
                
                return False
    
    def get(self, timeout: Optional[float] = None) -> T:
        """
        Get item from queue.
        
        Args:
            timeout: Timeout in seconds (None = wait forever)
        
        Returns:
            Item from queue
        
        Raises:
            queue.Empty: If timeout expires
        """
        return self._queue.get(timeout=timeout)
    
    def qsize(self) -> int:
        """Get approximate queue size."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            'name': self.name,
            'size': self.qsize(),
            'maxsize': self.maxsize,
            'dropped_count': self._dropped_count,
            'drop_policy': self.drop_policy,
        }


class GracefulThread(threading.Thread):
    """
    Thread with graceful shutdown support.
    
    Features:
    - Non-daemon by default (allows cleanup)
    - Shutdown event for clean exit
    - Exception handling and logging
    - Resource cleanup on exit
    
    Usage:
        def worker(thread: GracefulThread):
            while not thread.should_stop():
                # Do work
                thread.wait_or_stop(interval=1.0)
        
        thread = GracefulThread(target=worker, name="worker")
        thread.start()
        
        # Later...
        thread.stop(timeout=5.0)
    """
    
    def __init__(
        self,
        target: Callable,
        name: str = "unnamed",
        daemon: bool = False,
        cleanup_callback: Optional[Callable] = None
    ):
        """
        Initialize graceful thread.
        
        Args:
            target: Function to run (receives thread as first arg)
            name: Thread name
            daemon: Whether thread is daemon (default False for graceful shutdown)
            cleanup_callback: Optional cleanup function called on exit
        """
        super().__init__(name=name, daemon=daemon)
        
        self._target = target
        self._shutdown_event = threading.Event()
        self._cleanup_callback = cleanup_callback
        self._exception = None
    
    def run(self):
        """Run target with exception handling and cleanup."""
        try:
            logger.info(f"Thread '{self.name}' started")
            self._target(self)
        
        except ThreadingError:
            self._exception = e
            raise
        except Exception as e:
            self._exception = e
            logger.error(f"Thread '{self.name}' error: {e}", exc_info=True)
            raise ThreadingError(f"Thread '{self.name}' failed: {e}") from e
        
        finally:
            # Cleanup
            if self._cleanup_callback:
                try:
                    self._cleanup_callback()
                except ThreadingError:
                    raise
                except Exception as e:
                    logger.error(f"Thread '{self.name}' cleanup error: {e}")
                    raise ThreadingError(f"Thread '{self.name}' cleanup failed: {e}") from e
            
            logger.info(f"Thread '{self.name}' stopped")
    
    def should_stop(self) -> bool:
        """Check if thread should stop."""
        return self._shutdown_event.is_set()
    
    def wait_or_stop(self, interval: float) -> bool:
        """
        Wait for interval or until stop requested.
        
        Args:
            interval: Seconds to wait
        
        Returns:
            True if stop requested, False if timeout
        """
        return self._shutdown_event.wait(interval)
    
    def stop(self, timeout: float = 5.0) -> bool:
        """
        Request thread to stop and wait for completion.
        
        Args:
            timeout: Maximum seconds to wait
        
        Returns:
            True if thread stopped, False if timeout
        """
        logger.info(f"Stopping thread '{self.name}'...")
        
        self._shutdown_event.set()
        self.join(timeout=timeout)
        
        if self.is_alive():
            logger.warning(f"Thread '{self.name}' did not stop within {timeout}s")
            return False
        
        if self._exception:
            logger.error(f"Thread '{self.name}' had exception: {self._exception}")
        
        return True


class ThreadSafeDict(Generic[K, V]):
    """
    Thread-safe dictionary wrapper.
    
    Usage:
        d = ThreadSafeDict()
        
        d['key'] = 'value'
        value = d['key']
        
        # Atomic operations
        d.update({'key1': 'val1', 'key2': 'val2'})
        
        # Safe iteration
        for key, value in d.items():
            print(key, value)
    """
    
    def __init__(self, name: str = "unnamed"):
        """Initialize thread-safe dictionary."""
        self._dict: Dict[K, V] = {}
        self._lock = threading.RLock()  # Reentrant lock
        self.name = name
    
    def __setitem__(self, key: K, value: V):
        """Set item."""
        with self._lock:
            self._dict[key] = value
    
    def __getitem__(self, key: K) -> V:
        """Get item."""
        with self._lock:
            return self._dict[key]
    
    def __delitem__(self, key: K):
        """Delete item."""
        with self._lock:
            del self._dict[key]
    
    def __contains__(self, key: K) -> bool:
        """Check if key exists."""
        with self._lock:
            return key in self._dict
    
    def __len__(self) -> int:
        """Get dictionary size."""
        with self._lock:
            return len(self._dict)
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get item with default."""
        with self._lock:
            return self._dict.get(key, default)
    
    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Remove and return item."""
        with self._lock:
            return self._dict.pop(key, default)
    
    def update(self, other: Dict[K, V]):
        """Update dictionary."""
        with self._lock:
            self._dict.update(other)
    
    def clear(self):
        """Clear dictionary."""
        with self._lock:
            self._dict.clear()
    
    def keys(self):
        """Get keys (returns copy)."""
        with self._lock:
            return list(self._dict.keys())
    
    def values(self):
        """Get values (returns copy)."""
        with self._lock:
            return list(self._dict.values())
    
    def items(self):
        """Get items (returns copy)."""
        with self._lock:
            return list(self._dict.items())
    
    @contextmanager
    def lock(self):
        """Context manager for batch operations."""
        with self._lock:
            yield self._dict


class ThreadSafeSet(Generic[T]):
    """
    Thread-safe set wrapper.
    
    Usage:
        s = ThreadSafeSet()
        
        s.add('item')
        s.remove('item')
        
        if 'item' in s:
            print('found')
        
        # Safe iteration
        for item in s:
            print(item)
    """
    
    def __init__(self, name: str = "unnamed"):
        """Initialize thread-safe set."""
        self._set: Set[T] = set()
        self._lock = threading.RLock()
        self.name = name
    
    def add(self, item: T):
        """Add item."""
        with self._lock:
            self._set.add(item)
    
    def remove(self, item: T):
        """Remove item."""
        with self._lock:
            self._set.remove(item)
    
    def discard(self, item: T):
        """Remove item if exists."""
        with self._lock:
            self._set.discard(item)
    
    def __contains__(self, item: T) -> bool:
        """Check if item exists."""
        with self._lock:
            return item in self._set
    
    def __len__(self) -> int:
        """Get set size."""
        with self._lock:
            return len(self._set)
    
    def __iter__(self):
        """Iterate over copy of set."""
        with self._lock:
            return iter(list(self._set))
    
    def clear(self):
        """Clear set."""
        with self._lock:
            self._set.clear()
    
    def copy(self) -> Set[T]:
        """Get copy of set."""
        with self._lock:
            return self._set.copy()
    
    @contextmanager
    def lock(self):
        """Context manager for batch operations."""
        with self._lock:
            yield self._set


# Convenience functions

def create_bounded_queue(
    maxsize: int = 1000,
    drop_policy: str = 'oldest',
    name: str = "unnamed"
) -> BoundedQueue:
    """
    Create bounded queue with sensible defaults.
    
    Args:
        maxsize: Maximum queue size
        drop_policy: Drop policy when full
        name: Queue name
    
    Returns:
        BoundedQueue instance
    """
    return BoundedQueue(maxsize=maxsize, drop_policy=drop_policy, name=name)


def create_graceful_thread(
    target: Callable,
    name: str = "unnamed",
    cleanup_callback: Optional[Callable] = None
) -> GracefulThread:
    """
    Create graceful thread with sensible defaults.
    
    Args:
        target: Function to run
        name: Thread name
        cleanup_callback: Optional cleanup function
    
    Returns:
        GracefulThread instance
    """
    return GracefulThread(
        target=target,
        name=name,
        daemon=False,  # Non-daemon for graceful shutdown
        cleanup_callback=cleanup_callback
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Bounded queue
    print("Example 1: Bounded Queue")
    q = BoundedQueue(maxsize=5, drop_policy='oldest', name='example')
    
    for i in range(10):
        q.put(i, timeout=0.1)
    
    print(f"Queue stats: {q.get_stats()}")
    
    # Example 2: Graceful thread
    print("\nExample 2: Graceful Thread")
    
    def worker(thread: GracefulThread):
        count = 0
        while not thread.should_stop():
            count += 1
            print(f"Working... {count}")
            if thread.wait_or_stop(1.0):
                break
    
    thread = create_graceful_thread(target=worker, name='worker')
    thread.start()
    
    time.sleep(3)
    thread.stop(timeout=5.0)
    
    # Example 3: Thread-safe collections
    print("\nExample 3: Thread-Safe Collections")
    
    d = ThreadSafeDict(name='example')
    d['key1'] = 'value1'
    d['key2'] = 'value2'
    
    print(f"Dict items: {d.items()}")
    
    s = ThreadSafeSet(name='example')
    s.add('item1')
    s.add('item2')
    
    print(f"Set items: {list(s)}")
