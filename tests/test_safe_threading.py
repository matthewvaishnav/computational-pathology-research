"""
Tests for safe threading utilities.

Tests thread-safe primitives, graceful shutdown, and error handling.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

from src.utils.safe_threading import (
    TimeoutLock,
    BoundedQueue,
    GracefulThread,
    ThreadSafeDict,
    ThreadSafeSet,
)
from src.exceptions import ThreadingError


class TestTimeoutLock:
    """Test TimeoutLock functionality."""

    def test_lock_acquire_and_release(self):
        """Test lock can be acquired and released."""
        lock = TimeoutLock(timeout=1.0, name="test_lock")
        
        with lock:
            assert lock._owner == threading.current_thread().name
        
        assert lock._owner is None

    def test_lock_timeout_raises_error(self):
        """Test lock timeout raises TimeoutError."""
        lock = TimeoutLock(timeout=0.1, name="test_lock")
        
        # Acquire lock in main thread
        lock._lock.acquire()
        
        # Try to acquire again (should timeout)
        with pytest.raises(TimeoutError):
            with lock:
                pass
        
        # Cleanup
        lock._lock.release()

    def test_lock_reentrant_fails(self):
        """Test lock is not reentrant (blocks on same thread)."""
        lock = TimeoutLock(timeout=0.1, name="test_lock")
        
        with lock:
            # Try to acquire again in same thread (should timeout)
            with pytest.raises(TimeoutError):
                with lock:
                    pass


class TestBoundedQueue:
    """Test BoundedQueue functionality."""

    def test_queue_put_and_get(self):
        """Test queue put and get operations."""
        queue = BoundedQueue(maxsize=10, name="test_queue")
        
        queue.put(1)
        queue.put(2)
        queue.put(3)
        
        assert queue.get() == 1
        assert queue.get() == 2
        assert queue.get() == 3

    def test_queue_block_policy_raises_full(self):
        """Test block policy raises queue.Full when full."""
        import queue as q
        
        queue = BoundedQueue(maxsize=2, drop_policy='block', name="test_queue")
        
        queue.put(1)
        queue.put(2)
        
        with pytest.raises(q.Full):
            queue.put(3, timeout=0.1)

    def test_queue_oldest_policy_drops_oldest(self):
        """Test oldest policy drops oldest item when full."""
        queue = BoundedQueue(maxsize=2, drop_policy='oldest', name="test_queue")
        
        queue.put(1, timeout=0.1)
        queue.put(2, timeout=0.1)
        queue.put(3, timeout=0.1)  # Should drop 1
        
        assert queue.get() == 2
        assert queue.get() == 3

    def test_queue_newest_policy_drops_newest(self):
        """Test newest policy drops newest item when full."""
        queue = BoundedQueue(maxsize=2, drop_policy='newest', name="test_queue")
        
        queue.put(1, timeout=0.1)
        queue.put(2, timeout=0.1)
        result = queue.put(3, timeout=0.1)  # Should drop 3
        
        assert result is False  # Item was dropped
        assert queue.get() == 1
        assert queue.get() == 2

    def test_queue_stats(self):
        """Test queue statistics."""
        queue = BoundedQueue(maxsize=10, drop_policy='newest', name="test_queue")
        
        queue.put(1)
        queue.put(2)
        
        stats = queue.get_stats()
        assert stats['name'] == 'test_queue'
        assert stats['maxsize'] == 10
        assert stats['drop_policy'] == 'newest'


class TestGracefulThread:
    """Test GracefulThread functionality."""

    def test_thread_runs_target(self):
        """Test thread runs target function."""
        result = []
        
        def worker(thread):
            result.append("executed")
        
        thread = GracefulThread(target=worker, name="test_thread")
        thread.start()
        thread.join(timeout=1.0)
        
        assert result == ["executed"]

    def test_thread_should_stop(self):
        """Test thread should_stop flag."""
        stop_count = []
        
        def worker(thread):
            while not thread.should_stop():
                stop_count.append(1)
                if thread.wait_or_stop(0.1):
                    break
        
        thread = GracefulThread(target=worker, name="test_thread")
        thread.start()
        
        time.sleep(0.3)  # Let it run a bit
        thread.stop(timeout=1.0)
        
        assert len(stop_count) > 0  # Thread ran at least once

    def test_thread_cleanup_callback(self):
        """Test thread cleanup callback is called."""
        cleanup_called = []
        
        def cleanup():
            cleanup_called.append(True)
        
        def worker(thread):
            pass
        
        thread = GracefulThread(
            target=worker,
            name="test_thread",
            cleanup_callback=cleanup
        )
        thread.start()
        thread.join(timeout=1.0)
        
        assert cleanup_called == [True]

    def test_thread_exception_handling(self):
        """Test thread exception is captured."""
        def failing_worker(thread):
            raise ValueError("Test error")
        
        thread = GracefulThread(target=failing_worker, name="test_thread")
        thread.start()
        thread.join(timeout=1.0)
        
        assert thread._exception is not None


class TestThreadSafeDict:
    """Test ThreadSafeDict functionality."""

    def test_dict_set_and_get(self):
        """Test dict set and get operations."""
        d = ThreadSafeDict(name="test_dict")
        
        d['key1'] = 'value1'
        d['key2'] = 'value2'
        
        assert d['key1'] == 'value1'
        assert d['key2'] == 'value2'

    def test_dict_contains(self):
        """Test dict contains check."""
        d = ThreadSafeDict(name="test_dict")
        
        d['key1'] = 'value1'
        
        assert 'key1' in d
        assert 'key2' not in d

    def test_dict_delete(self):
        """Test dict delete operation."""
        d = ThreadSafeDict(name="test_dict")
        
        d['key1'] = 'value1'
        del d['key1']
        
        assert 'key1' not in d

    def test_dict_update(self):
        """Test dict update operation."""
        d = ThreadSafeDict(name="test_dict")
        
        d.update({'key1': 'value1', 'key2': 'value2'})
        
        assert d['key1'] == 'value1'
        assert d['key2'] == 'value2'

    def test_dict_items(self):
        """Test dict items returns copy."""
        d = ThreadSafeDict(name="test_dict")
        
        d['key1'] = 'value1'
        d['key2'] = 'value2'
        
        items = d.items()
        assert len(items) == 2
        assert ('key1', 'value1') in items
        assert ('key2', 'value2') in items

    def test_dict_thread_safety(self):
        """Test dict is thread-safe under concurrent access."""
        d = ThreadSafeDict(name="test_dict")
        errors = []
        
        def writer(thread_id):
            try:
                for i in range(100):
                    d[f'key_{thread_id}_{i}'] = f'value_{thread_id}_{i}'
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(d) == 500  # 5 threads * 100 items


class TestThreadSafeSet:
    """Test ThreadSafeSet functionality."""

    def test_set_add_and_contains(self):
        """Test set add and contains operations."""
        s = ThreadSafeSet(name="test_set")
        
        s.add('item1')
        s.add('item2')
        
        assert 'item1' in s
        assert 'item2' in s
        assert 'item3' not in s

    def test_set_remove(self):
        """Test set remove operation."""
        s = ThreadSafeSet(name="test_set")
        
        s.add('item1')
        s.remove('item1')
        
        assert 'item1' not in s

    def test_set_discard(self):
        """Test set discard operation (no error if not exists)."""
        s = ThreadSafeSet(name="test_set")
        
        s.add('item1')
        s.discard('item1')
        s.discard('item2')  # Should not raise error
        
        assert 'item1' not in s

    def test_set_iteration(self):
        """Test set iteration returns copy."""
        s = ThreadSafeSet(name="test_set")
        
        s.add('item1')
        s.add('item2')
        s.add('item3')
        
        items = list(s)
        assert len(items) == 3
        assert 'item1' in items
        assert 'item2' in items
        assert 'item3' in items

    def test_set_thread_safety(self):
        """Test set is thread-safe under concurrent access."""
        s = ThreadSafeSet(name="test_set")
        errors = []
        
        def writer(thread_id):
            try:
                for i in range(100):
                    s.add(f'item_{thread_id}_{i}')
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(s) == 500  # 5 threads * 100 items


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
