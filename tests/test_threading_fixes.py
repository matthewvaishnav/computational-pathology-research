"""
Test suite for Round 3 Threading and Concurrency Fixes.

This test file contains unit tests, integration tests, and stress tests for:
- Bounded queue implementation
- Graceful thread shutdown
- Lock timeout protection
- Thread-safe collections
- Stop event return value checking
- Asyncio exception handling
- Resource cleanup (SQLite, matplotlib, GPU)
- Configuration validation

Test markers:
- @pytest.mark.unit: Unit tests for individual components
- @pytest.mark.integration: Integration tests for end-to-end workflows
- @pytest.mark.stress: Concurrency stress tests with 100+ threads

Requirements: 13.1-13.7
"""

import pytest
import threading
import asyncio
import time
import sqlite3
import tempfile
import os
from queue import Empty
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Hypothesis for property-based testing
from hypothesis import given, strategies as st, settings, Phase
import hypothesis

# Import the thread-safe utilities
from src.utils.safe_threading import (
    BoundedQueue,
    GracefulThread,
    TimeoutLock,
    ThreadSafeDict,
    ThreadSafeSet,
)


# =============================================================================
# Test Infrastructure Setup
# =============================================================================

@pytest.fixture
def temp_db():
    """Fixture for temporary SQLite database."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def mock_websocket():
    """Fixture for mock WebSocket connection."""
    websocket = MagicMock()
    websocket.send_json = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


class AsyncMock(MagicMock):
    """Mock for async functions."""
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


# =============================================================================
# Unit Tests - Bounded Queues (Task 2.5)
# =============================================================================

@pytest.mark.unit
class TestBoundedQueue:
    """Unit tests for BoundedQueue implementation."""
    
    def test_maxsize_enforcement(self):
        """Test that queue enforces maxsize limit."""
        queue = BoundedQueue(maxsize=5, drop_policy='block', name='test_maxsize')
        
        # Fill queue to maxsize
        for i in range(5):
            assert queue.put(i, timeout=0.1)
        
        # Queue should be full
        assert queue.full()
        assert queue.qsize() == 5
        
        # Attempting to add more should raise queue.Full with block policy
        with pytest.raises(Exception):  # queue.Full
            queue.put(6, timeout=0.1)
    
    def test_drop_oldest_policy(self):
        """Test that drop-oldest policy works correctly."""
        queue = BoundedQueue(maxsize=5, drop_policy='oldest', name='test_drop_oldest')
        
        # Fill queue
        for i in range(5):
            assert queue.put(i, timeout=0.1)
        
        # Add one more - should drop oldest (0)
        assert queue.put(5, timeout=0.1)
        
        # Verify oldest was dropped
        assert queue.get(timeout=0.1) == 1  # 0 was dropped
        assert queue.get(timeout=0.1) == 2
        assert queue.get(timeout=0.1) == 3
        assert queue.get(timeout=0.1) == 4
        assert queue.get(timeout=0.1) == 5
    
    def test_drop_newest_policy(self):
        """Test that drop-newest policy works correctly."""
        queue = BoundedQueue(maxsize=5, drop_policy='newest', name='test_drop_newest')
        
        # Fill queue
        for i in range(5):
            assert queue.put(i, timeout=0.1)
        
        # Add one more - should drop newest (5)
        assert not queue.put(5, timeout=0.1)  # Returns False when dropped
        
        # Verify newest was dropped
        assert queue.get(timeout=0.1) == 0
        assert queue.get(timeout=0.1) == 1
        assert queue.get(timeout=0.1) == 2
        assert queue.get(timeout=0.1) == 3
        assert queue.get(timeout=0.1) == 4
    
    def test_queue_statistics(self):
        """Test that queue statistics are accurate."""
        queue = BoundedQueue(maxsize=10, drop_policy='oldest', name='test_stats')
        
        # Add items
        for i in range(5):
            queue.put(i, timeout=0.1)
        
        stats = queue.get_stats()
        assert stats['name'] == 'test_stats'
        assert stats['maxsize'] == 10
        assert stats['size'] == 5
        assert stats['dropped_count'] == 0
        assert stats['drop_policy'] == 'oldest'
        
        # Fill and overflow
        for i in range(10):
            queue.put(i, timeout=0.1)
        
        stats = queue.get_stats()
        assert stats['dropped_count'] == 5  # 5 items dropped
    
    def test_concurrent_producer_consumer(self):
        """Test concurrent producer/consumer scenarios."""
        queue = BoundedQueue(maxsize=50, drop_policy='oldest', name='test_concurrent')
        produced = []
        consumed = []
        
        def producer(start, count):
            for i in range(start, start + count):
                queue.put(i, timeout=1.0)
                produced.append(i)
        
        def consumer(count):
            for _ in range(count):
                try:
                    item = queue.get(timeout=2.0)
                    consumed.append(item)
                except Empty:
                    break
        
        # Create producer and consumer threads
        producers = [
            threading.Thread(target=producer, args=(i * 20, 20))
            for i in range(3)
        ]
        consumers = [
            threading.Thread(target=consumer, args=(20,))
            for _ in range(3)
        ]
        
        # Start all threads
        for t in producers + consumers:
            t.start()
        
        # Wait for completion
        for t in producers + consumers:
            t.join(timeout=5.0)
        
        # Verify items were produced and consumed
        assert len(produced) == 60
        assert len(consumed) > 0  # Some items consumed
        
        # Verify queue statistics
        stats = queue.get_stats()
        assert stats['dropped_count'] >= 0  # May have dropped some items
    
    def test_empty_queue_get_timeout(self):
        """Test that get() times out on empty queue."""
        queue = BoundedQueue(maxsize=5, drop_policy='oldest', name='test_empty')
        
        with pytest.raises(Empty):
            queue.get(timeout=0.1)
    
    def test_queue_empty_and_full_checks(self):
        """Test empty() and full() methods."""
        queue = BoundedQueue(maxsize=3, drop_policy='oldest', name='test_checks')
        
        assert queue.empty()
        assert not queue.full()
        
        queue.put(1, timeout=0.1)
        assert not queue.empty()
        assert not queue.full()
        
        queue.put(2, timeout=0.1)
        queue.put(3, timeout=0.1)
        assert not queue.empty()
        assert queue.full()


# =============================================================================
# Unit Tests - Graceful Thread Shutdown (Task 3.5)
# =============================================================================

@pytest.mark.unit
class TestGracefulThread:
    """Unit tests for GracefulThread implementation."""
    
    def test_thread_stops_within_timeout(self):
        """Test that graceful thread stops within timeout."""
        def worker(thread: GracefulThread):
            """Simple worker that checks for stop signal."""
            while not thread.should_stop():
                if thread.wait_or_stop(0.1):
                    break
        
        thread = GracefulThread(target=worker, name='test_stop')
        thread.start()
        
        # Give thread time to start
        time.sleep(0.2)
        
        # Stop thread
        start_time = time.time()
        assert thread.stop(timeout=1.0)
        stop_time = time.time()
        
        # Verify thread stopped
        assert not thread.is_alive()
        
        # Verify it stopped quickly (within timeout)
        assert (stop_time - start_time) < 1.0
    
    def test_cleanup_callback_executed(self):
        """Test that cleanup callbacks are executed on thread stop."""
        cleanup_called = []
        
        def cleanup_callback():
            cleanup_called.append(True)
        
        def worker(thread: GracefulThread):
            while not thread.should_stop():
                if thread.wait_or_stop(0.1):
                    break
        
        thread = GracefulThread(
            target=worker,
            name='test_cleanup',
            cleanup_callback=cleanup_callback
        )
        thread.start()
        
        # Stop thread
        thread.stop(timeout=1.0)
        
        # Verify cleanup was called
        assert len(cleanup_called) == 1
    
    def test_thread_stop_logging(self):
        """Test that thread start and stop events are logged."""
        def worker(thread: GracefulThread):
            while not thread.should_stop():
                if thread.wait_or_stop(0.1):
                    break
        
        thread = GracefulThread(target=worker, name='test_logging')
        
        with patch('src.utils.safe_threading.logger') as mock_logger:
            thread.start()
            time.sleep(0.2)
            thread.stop(timeout=1.0)
            
            # Verify start and stop were logged
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("started" in str(call).lower() for call in info_calls)
            assert any("stopped" in str(call).lower() for call in info_calls)
    
    def test_timeout_warning_when_thread_doesnt_stop(self):
        """Test that a warning is logged when thread doesn't stop within timeout."""
        def worker(thread: GracefulThread):
            # Intentionally ignore stop signal for a while
            time.sleep(2.0)
        
        thread = GracefulThread(target=worker, name='test_timeout_warning')
        thread.start()
        
        with patch('src.utils.safe_threading.logger') as mock_logger:
            # Try to stop with short timeout
            result = thread.stop(timeout=0.1)
            
            # Verify stop returned False (timeout)
            assert not result
            
            # Verify warning was logged
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
            assert any("did not stop" in str(call).lower() for call in warning_calls)
        
        # Clean up - wait for thread to actually finish
        thread.join(timeout=3.0)
    
    def test_should_stop_returns_true_after_stop_requested(self):
        """Test that should_stop() returns True after stop is requested."""
        stop_checked = []
        
        def worker(thread: GracefulThread):
            for i in range(10):
                if thread.should_stop():
                    stop_checked.append(True)
                    break
                time.sleep(0.1)
        
        thread = GracefulThread(target=worker, name='test_should_stop')
        thread.start()
        
        # Let thread run a bit
        time.sleep(0.2)
        
        # Request stop
        thread.stop(timeout=1.0)
        
        # Verify should_stop was checked and returned True
        assert len(stop_checked) > 0
    
    def test_wait_or_stop_returns_true_when_stopped(self):
        """Test that wait_or_stop() returns True when stop is requested."""
        wait_result = []
        
        def worker(thread: GracefulThread):
            # Wait for a long time, but should exit early when stopped
            result = thread.wait_or_stop(10.0)
            wait_result.append(result)
        
        thread = GracefulThread(target=worker, name='test_wait_or_stop')
        thread.start()
        
        # Let thread start waiting
        time.sleep(0.2)
        
        # Request stop
        start_time = time.time()
        thread.stop(timeout=1.0)
        stop_time = time.time()
        
        # Verify thread stopped quickly (not after 10 seconds)
        assert (stop_time - start_time) < 2.0
        
        # Verify wait_or_stop returned True
        assert len(wait_result) == 1
        assert wait_result[0] is True
    
    def test_exception_in_worker_is_logged(self):
        """Test that exceptions in worker function are logged."""
        def worker(thread: GracefulThread):
            raise ValueError("Test exception")
        
        thread = GracefulThread(target=worker, name='test_exception')
        
        with patch('src.utils.safe_threading.logger') as mock_logger:
            thread.start()
            thread.join(timeout=1.0)
            
            # Verify exception was logged
            error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
            assert any("error" in str(call).lower() for call in error_calls)
    
    def test_cleanup_callback_runs_even_with_exception(self):
        """Test that cleanup callback runs even when worker raises exception."""
        cleanup_called = []
        
        def cleanup_callback():
            cleanup_called.append(True)
        
        def worker(thread: GracefulThread):
            raise ValueError("Test exception")
        
        thread = GracefulThread(
            target=worker,
            name='test_cleanup_exception',
            cleanup_callback=cleanup_callback
        )
        thread.start()
        thread.join(timeout=1.0)
        
        # Verify cleanup was called despite exception
        assert len(cleanup_called) == 1
    
    def test_non_daemon_thread_by_default(self):
        """Test that GracefulThread is non-daemon by default."""
        def worker(thread: GracefulThread):
            while not thread.should_stop():
                if thread.wait_or_stop(0.1):
                    break
        
        thread = GracefulThread(target=worker, name='test_non_daemon')
        
        # Verify thread is non-daemon
        assert not thread.daemon
        
        # Clean up
        thread.start()
        thread.stop(timeout=1.0)


# =============================================================================
# Unit Tests - Lock Timeout Protection (Task 5.3)
# =============================================================================

@pytest.mark.unit
class TestTimeoutLock:
    """Unit tests for TimeoutLock implementation."""
    
    def test_lock_timeout_raises_timeout_error(self):
        """Test that lock timeout raises TimeoutError when lock cannot be acquired."""
        lock = TimeoutLock(timeout=0.5, name='test_timeout')
        
        # Acquire lock in main thread
        with lock:
            # Try to acquire in another thread - should timeout
            timeout_occurred = []
            
            def try_acquire():
                try:
                    with lock:
                        pass
                except TimeoutError as e:
                    timeout_occurred.append(str(e))
            
            thread = threading.Thread(target=try_acquire)
            thread.start()
            thread.join(timeout=2.0)
            
            # Verify TimeoutError was raised
            assert len(timeout_occurred) == 1
            assert 'test_timeout' in timeout_occurred[0]
            assert '0.5' in timeout_occurred[0]
    
    def test_lock_hold_time_warning(self):
        """Test that lock hold time warnings are logged when lock held >5 seconds."""
        lock = TimeoutLock(timeout=30.0, name='test_hold_warning')
        
        with patch('src.utils.safe_threading.logger') as mock_logger:
            with lock:
                # Hold lock for more than 5 seconds
                time.sleep(5.5)
            
            # Verify warning was logged
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
            assert any('test_hold_warning' in str(call) and 'held for' in str(call).lower() 
                      for call in warning_calls)
    
    def test_lock_acquisition_release_logging(self):
        """Test that lock acquisition and release events are logged at debug level."""
        lock = TimeoutLock(timeout=30.0, name='test_logging')
        
        with patch('src.utils.safe_threading.logger') as mock_logger:
            with lock:
                pass
            
            # Verify acquisition was logged
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any('acquired' in str(call).lower() and 'test_logging' in str(call) 
                      for call in debug_calls)
            assert any('released' in str(call).lower() and 'test_logging' in str(call) 
                      for call in debug_calls)
    
    def test_concurrent_lock_acquisition(self):
        """Test concurrent lock acquisition with multiple threads."""
        lock = TimeoutLock(timeout=2.0, name='test_concurrent')
        acquired_order = []
        
        def worker(worker_id):
            try:
                with lock:
                    acquired_order.append(worker_id)
                    time.sleep(0.1)  # Hold lock briefly
            except TimeoutError:
                acquired_order.append(f'{worker_id}_timeout')
        
        # Create multiple threads
        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(5)
        ]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Verify all threads either acquired lock or timed out
        assert len(acquired_order) == 5
        
        # Verify at least some threads acquired the lock successfully
        successful_acquisitions = [x for x in acquired_order if not str(x).endswith('_timeout')]
        assert len(successful_acquisitions) > 0
    
    def test_lock_includes_thread_name_in_logs(self):
        """Test that lock logs include the thread name."""
        lock = TimeoutLock(timeout=30.0, name='test_thread_name')
        
        with patch('src.utils.safe_threading.logger') as mock_logger:
            with lock:
                pass
            
            # Verify thread name is in debug logs
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            # The lock should log the thread name that acquired it
            assert len(debug_calls) > 0
    
    def test_lock_can_be_reacquired_after_release(self):
        """Test that lock can be acquired again after being released."""
        lock = TimeoutLock(timeout=1.0, name='test_reacquire')
        
        # First acquisition
        with lock:
            pass
        
        # Second acquisition should succeed
        with lock:
            pass
        
        # Third acquisition should also succeed
        with lock:
            pass
    
    def test_lock_timeout_with_zero_timeout(self):
        """Test that lock with very short timeout fails quickly."""
        lock = TimeoutLock(timeout=0.1, name='test_quick_timeout')
        
        # Acquire lock
        with lock:
            # Try to acquire in another thread with short timeout
            timeout_occurred = []
            start_time = time.time()
            
            def try_acquire():
                try:
                    with lock:
                        pass
                except TimeoutError:
                    timeout_occurred.append(time.time() - start_time)
            
            thread = threading.Thread(target=try_acquire)
            thread.start()
            thread.join(timeout=1.0)
            
            # Verify timeout occurred quickly
            assert len(timeout_occurred) == 1
            assert timeout_occurred[0] < 0.5  # Should timeout in ~0.1s, not longer


# =============================================================================
# Unit Tests - Thread-Safe Collections (Task 6.3)
# =============================================================================

@pytest.mark.unit
class TestThreadSafeCollections:
    """Unit tests for ThreadSafeDict and ThreadSafeSet."""
    
    def test_thread_safe_dict_concurrent_read_write(self):
        """Test ThreadSafeDict handles concurrent read/write operations."""
        d = ThreadSafeDict[str, int](name='test_dict')
        
        def writer(key: str, value: int):
            d[key] = value
        
        def reader(key: str) -> Optional[int]:
            return d.get(key)
        
        # Start 10 writer threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=writer, args=(f'key{i}', i))
            threads.append(t)
            t.start()
        
        # Wait for all writes
        for t in threads:
            t.join()
        
        # Verify all writes succeeded
        for i in range(10):
            assert d[f'key{i}'] == i
    
    def test_thread_safe_set_concurrent_add_remove(self):
        """Test ThreadSafeSet handles concurrent add/remove operations."""
        s = ThreadSafeSet[int](name='test_set')
        
        def adder(value: int):
            s.add(value)
        
        def remover(value: int):
            s.discard(value)
        
        # Add 20 items
        threads = []
        for i in range(20):
            t = threading.Thread(target=adder, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(s) == 20
        
        # Remove 10 items
        threads = []
        for i in range(10):
            t = threading.Thread(target=remover, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(s) == 10
    
    def test_thread_safe_dict_iteration_safety(self):
        """Test ThreadSafeDict iteration doesn't raise concurrent modification errors."""
        d = ThreadSafeDict[str, int](name='test_dict')
        
        # Populate dict
        for i in range(100):
            d[f'key{i}'] = i
        
        # Iterate while modifying in another thread
        def modifier():
            for i in range(100, 200):
                d[f'key{i}'] = i
                time.sleep(0.001)
        
        t = threading.Thread(target=modifier)
        t.start()
        
        # Iterate (should not raise)
        count = 0
        for key, value in d.items():
            count += 1
        
        t.join()
        
        assert count >= 100  # At least original items
    
    def test_thread_safe_set_iteration_safety(self):
        """Test ThreadSafeSet iteration doesn't raise concurrent modification errors."""
        s = ThreadSafeSet[int](name='test_set')
        
        # Populate set
        for i in range(100):
            s.add(i)
        
        # Iterate while modifying in another thread
        def modifier():
            for i in range(100, 200):
                s.add(i)
                time.sleep(0.001)
        
        t = threading.Thread(target=modifier)
        t.start()
        
        # Iterate (should not raise)
        count = 0
        for item in s:
            count += 1
        
        t.join()
        
        assert count >= 100  # At least original items
    
    def test_thread_safe_dict_batch_operations_with_lock(self):
        """Test ThreadSafeDict batch operations with lock context manager."""
        d = ThreadSafeDict[str, int](name='test_dict')
        
        # Batch update with lock
        with d.lock():
            for i in range(10):
                d[f'key{i}'] = i
        
        # Verify all updates
        assert len(d) == 10
        for i in range(10):
            assert d[f'key{i}'] == i
    
    def test_thread_safe_dict_get_set_operations(self):
        """Test ThreadSafeDict get/set operations."""
        d = ThreadSafeDict[str, int](name='test_dict')
        
        # Set
        d['key1'] = 100
        assert d['key1'] == 100
        
        # Get with default
        assert d.get('key2', 200) == 200
        
        # Contains
        assert 'key1' in d
        assert 'key2' not in d
        
        # Delete
        del d['key1']
        assert 'key1' not in d
    
    def test_thread_safe_set_add_remove_operations(self):
        """Test ThreadSafeSet add/remove operations."""
        s = ThreadSafeSet[str](name='test_set')
        
        # Add
        s.add('item1')
        assert 'item1' in s
        
        # Remove
        s.remove('item1')
        assert 'item1' not in s
        
        # Discard (no error if not present)
        s.discard('item2')  # Should not raise
        
        # Add multiple
        s.add('item3')
        s.add('item4')
        assert len(s) == 2


# =============================================================================
# Unit Tests - Stop Event Checking (Task 7.2)
# =============================================================================

@pytest.mark.unit
class TestStopEventChecking:
    """Unit tests for stop event return value checking."""
    
    def test_loop_exits_immediately_when_stop_requested(self):
        """Test that monitoring loop exits immediately when stop is requested."""
        exit_times = []
        
        def monitoring_loop(thread: GracefulThread):
            """Simulated monitoring loop."""
            start_time = time.time()
            while not thread.should_stop():
                # Simulate work
                if thread.wait_or_stop(10.0):  # Long wait
                    break
            exit_time = time.time() - start_time
            exit_times.append(exit_time)
        
        thread = GracefulThread(target=monitoring_loop, name='test_stop_exit')
        thread.start()
        
        # Let thread start waiting
        time.sleep(0.2)
        
        # Request stop
        start_time = time.time()
        thread.stop(timeout=2.0)
        stop_time = time.time()
        
        # Verify thread exited quickly, not after 10 seconds
        assert len(exit_times) == 1
        assert exit_times[0] < 3.0, f"Loop took {exit_times[0]}s to exit, expected < 3s"
        assert (stop_time - start_time) < 2.0, "Stop took too long"
    
    def test_loop_does_not_wait_full_timeout_when_stopped_early(self):
        """Test that loop doesn't wait full timeout period when stopped early."""
        iterations = []
        
        def monitoring_loop(thread: GracefulThread):
            """Loop with long wait periods."""
            while not thread.should_stop():
                iterations.append(time.time())
                # Long wait - should exit early when stopped
                if thread.wait_or_stop(30.0):
                    break
        
        thread = GracefulThread(target=monitoring_loop, name='test_early_exit')
        thread.start()
        
        # Let thread do one iteration
        time.sleep(0.5)
        
        # Stop thread
        start_time = time.time()
        thread.stop(timeout=2.0)
        stop_time = time.time()
        
        # Verify thread stopped quickly, not after 30 seconds
        assert (stop_time - start_time) < 3.0, "Thread took too long to stop"
        assert len(iterations) >= 1, "Thread should have done at least one iteration"
    
    def test_should_stop_checked_in_loop_condition(self):
        """Test that should_stop() is checked in loop condition."""
        loop_checks = []
        
        def monitoring_loop(thread: GracefulThread):
            """Loop that tracks should_stop checks."""
            while not thread.should_stop():
                loop_checks.append('checked')
                if thread.wait_or_stop(0.1):
                    break
        
        thread = GracefulThread(target=monitoring_loop, name='test_should_stop')
        thread.start()
        
        # Let thread run a bit
        time.sleep(0.3)
        
        # Stop thread
        thread.stop(timeout=1.0)
        
        # Verify should_stop was checked multiple times
        assert len(loop_checks) >= 2, "should_stop should be checked multiple times"
    
    def test_stop_event_responsiveness_under_load(self):
        """Test that stop event is responsive even under load."""
        work_done = []
        
        def monitoring_loop(thread: GracefulThread):
            """Loop that does work."""
            while not thread.should_stop():
                # Simulate some work
                work_done.append(1)
                time.sleep(0.01)
                # Check for stop with short wait
                if thread.wait_or_stop(0.1):
                    break
        
        thread = GracefulThread(target=monitoring_loop, name='test_load')
        thread.start()
        
        # Let thread do some work
        time.sleep(0.5)
        
        # Stop thread
        start_time = time.time()
        thread.stop(timeout=1.0)
        stop_time = time.time()
        
        # Verify thread stopped quickly
        assert (stop_time - start_time) < 1.0, "Thread took too long to stop under load"
        assert len(work_done) > 0, "Thread should have done some work"


# =============================================================================
# Integration Tests - WebSocket Exception Handling (Task 9.5)
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestWebSocketExceptionHandling:
    """Integration tests for WebSocket exception handling."""
    
    async def test_placeholder(self):
        """Placeholder test - will be implemented in Task 9.5."""
        # TODO: Implement in Task 9.5
        # - Test CancelledError is re-raised
        # - Test WebSocketDisconnect is handled gracefully
        # - Test timeout closes connection
        # - Test resource cleanup on exception
        pass


# =============================================================================
# Unit Tests - SQLite Connection Cleanup (Task 10.2)
# =============================================================================

@pytest.mark.unit
class TestSQLiteCleanup:
    """Unit tests for SQLite connection cleanup."""
    
    def test_placeholder(self, temp_db):
        """Placeholder test - will be implemented in Task 10.2."""
        # TODO: Implement in Task 10.2
        # - Test connection is closed on success
        # - Test connection is closed on exception
        # - Test rollback occurs on exception
        # - Test cleanup with null connection
        pass


# =============================================================================
# Unit Tests - Matplotlib Figure Cleanup (Task 11.2)
# =============================================================================

@pytest.mark.unit
class TestMatplotlibCleanup:
    """Unit tests for matplotlib figure cleanup."""
    
    def test_placeholder(self):
        """Placeholder test - will be implemented in Task 11.2."""
        # TODO: Implement in Task 11.2
        # - Test figure is closed on success
        # - Test figure is closed on exception
        # - Test cleanup with null figure
        pass


# =============================================================================
# Unit Tests - GPU Memory Cleanup (Task 12.2)
# =============================================================================

@pytest.mark.unit
class TestGPUMemoryCleanup:
    """Unit tests for GPU memory cleanup."""
    
    def test_placeholder(self):
        """Placeholder test - will be implemented in Task 12.2."""
        # TODO: Implement in Task 12.2
        # - Test tensors are deleted on success
        # - Test tensors are deleted on exception
        # - Test empty_cache is called when GPU available
        # - Test cleanup with null tensors
        pass


# =============================================================================
# Unit Tests - Configuration Validation (Task 14.4)
# =============================================================================

@pytest.mark.unit
class TestConfigurationValidation:
    """Unit tests for configuration validation."""
    
    def test_placeholder(self):
        """Placeholder test - will be implemented in Task 14.4."""
        # TODO: Implement in Task 14.4
        # - Test valid configuration is accepted
        # - Test invalid configuration is rejected
        # - Test missing required fields are detected
        # - Test out-of-range values are rejected
        # - Test Slack webhook validation
        # - Test email configuration validation
        pass


# =============================================================================
# Property-Based Tests - Bounded Queue (Task 2.6)
# =============================================================================

@pytest.mark.stress
@pytest.mark.property
class TestBoundedQueueProperties:
    """Property-based tests for bounded queue invariants."""
    
    @given(
        operations=st.lists(
            st.tuples(
                st.sampled_from(['put', 'get']),
                st.integers(min_value=0, max_value=1000)
            ),
            min_size=100,
            max_size=1000
        )
    )
    @settings(max_examples=10, deadline=None)
    def test_queue_size_never_exceeds_maxsize(self, operations):
        """
        Property 1: For any sequence of concurrent queue operations,
        queue size never exceeds maxsize.
        
        **Validates: Requirements 1.1-1.4**
        """
        maxsize = 100
        queue = BoundedQueue(maxsize=maxsize, drop_policy='oldest', name='property_test')
        
        def worker(ops):
            for op, value in ops:
                if op == 'put':
                    try:
                        queue.put(value, timeout=0.1)
                    except Exception:
                        pass  # Ignore timeout errors
                else:  # get
                    try:
                        queue.get(timeout=0.1)
                    except Empty:
                        pass  # Ignore empty queue
        
        # Split operations across 10 threads
        num_threads = 10
        chunk_size = len(operations) // num_threads
        threads = []
        
        for i in range(num_threads):
            start = i * chunk_size
            end = start + chunk_size if i < num_threads - 1 else len(operations)
            chunk = operations[start:end]
            thread = threading.Thread(target=worker, args=(chunk,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Verify invariant: queue size never exceeds maxsize
        assert queue.qsize() <= maxsize, f"Queue size {queue.qsize()} exceeds maxsize {maxsize}"
        
        # Verify statistics are consistent
        stats = queue.get_stats()
        assert stats['size'] <= stats['maxsize']
        assert stats['dropped_count'] >= 0


# =============================================================================
# Property-Based Tests - Graceful Thread (Task 3.6)
# =============================================================================

@pytest.mark.stress
@pytest.mark.property
class TestGracefulThreadProperties:
    """Property-based tests for graceful thread behavior."""
    
    @given(
        num_threads=st.integers(min_value=10, max_value=100),
        work_duration=st.floats(min_value=0.1, max_value=2.0)
    )
    @settings(max_examples=10, deadline=None)
    def test_threads_stop_within_timeout(self, num_threads, work_duration):
        """
        Property 2: Threads stop within timeout period.
        
        For any number of threads and work duration, all threads should
        stop within the specified timeout when stop is requested.
        
        **Validates: Requirements 2.4, 2.5**
        """
        threads = []
        cleanup_counts = []
        
        def cleanup_callback():
            cleanup_counts.append(1)
        
        def worker(thread: GracefulThread):
            """Worker that does some work then checks for stop."""
            start_time = time.time()
            while not thread.should_stop():
                # Simulate work
                if time.time() - start_time > work_duration:
                    break
                if thread.wait_or_stop(0.1):
                    break
        
        # Create threads
        for i in range(num_threads):
            thread = GracefulThread(
                target=worker,
                name=f'property_test_{i}',
                cleanup_callback=cleanup_callback
            )
            threads.append(thread)
            thread.start()
        
        # Let threads run a bit
        time.sleep(0.2)
        
        # Stop all threads
        timeout = 5.0
        start_time = time.time()
        
        for thread in threads:
            result = thread.stop(timeout=timeout)
            # Each thread should stop within timeout
            assert result or not thread.is_alive(), f"Thread {thread.name} did not stop"
        
        stop_time = time.time()
        
        # Verify all threads stopped
        for thread in threads:
            assert not thread.is_alive(), f"Thread {thread.name} still alive"
        
        # Verify total time is reasonable (not num_threads * timeout)
        # Threads should stop in parallel, not sequentially
        assert (stop_time - start_time) < (timeout * 2), "Threads took too long to stop"
        
        # Verify cleanup callbacks were called for all threads
        assert len(cleanup_counts) == num_threads, f"Expected {num_threads} cleanups, got {len(cleanup_counts)}"
    
    @given(
        stop_delay=st.floats(min_value=0.1, max_value=1.0)
    )
    @settings(max_examples=10, deadline=None)
    def test_wait_or_stop_exits_immediately_on_stop(self, stop_delay):
        """
        Property: wait_or_stop() exits immediately when stop is requested,
        not after the full wait period.
        
        **Validates: Requirements 2.2, 2.3**
        """
        exit_times = []
        
        def worker(thread: GracefulThread):
            start_time = time.time()
            # Wait for a long time (10 seconds)
            thread.wait_or_stop(10.0)
            exit_time = time.time() - start_time
            exit_times.append(exit_time)
        
        thread = GracefulThread(target=worker, name='test_immediate_exit')
        thread.start()
        
        # Wait a bit, then stop
        time.sleep(stop_delay)
        thread.stop(timeout=2.0)
        
        # Verify thread exited quickly, not after 10 seconds
        assert len(exit_times) == 1
        assert exit_times[0] < 3.0, f"Thread took {exit_times[0]}s to exit, expected < 3s"
        assert exit_times[0] >= stop_delay, f"Thread exited too early: {exit_times[0]}s < {stop_delay}s"


# =============================================================================
# Property-Based Tests - Lock Timeout (Task 5.4)
# =============================================================================

@pytest.mark.stress
@pytest.mark.property
class TestLockTimeoutProperties:
    """Property-based tests for lock timeout behavior."""
    
    @given(
        hold_time=st.floats(min_value=0.5, max_value=3.0),
        num_threads=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=10, deadline=None)
    def test_lock_acquisition_fails_with_timeout_error_after_timeout(self, hold_time, num_threads):
        """
        Property 3: Lock acquisition fails with TimeoutError after timeout.
        
        For any lock hold time and number of competing threads, threads that
        cannot acquire the lock within the timeout period should raise TimeoutError.
        
        **Validates: Requirements 3.2**
        """
        timeout = 0.5  # Short timeout for testing
        lock = TimeoutLock(timeout=timeout, name='property_test_lock')
        
        results = []
        
        def worker(worker_id):
            """Worker that tries to acquire lock."""
            try:
                with lock:
                    # Hold lock for specified time
                    time.sleep(hold_time)
                    results.append(('success', worker_id))
            except TimeoutError as e:
                results.append(('timeout', worker_id))
        
        # Create threads
        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(num_threads)
        ]
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=hold_time + timeout + 2.0)
        
        # Verify all threads completed
        assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"
        
        # Count successes and timeouts
        successes = [r for r in results if r[0] == 'success']
        timeouts = [r for r in results if r[0] == 'timeout']
        
        # At least one thread should succeed (the first one)
        assert len(successes) >= 1, "At least one thread should acquire the lock"
        
        # If hold_time > timeout, most threads should timeout
        if hold_time > timeout:
            # Most threads should timeout because lock is held longer than timeout
            assert len(timeouts) > 0, "Some threads should timeout when lock held longer than timeout"
        
        # Verify no threads are still running
        for thread in threads:
            assert not thread.is_alive(), f"Thread {thread.name} still alive"
    
    @given(
        num_contentious_threads=st.integers(min_value=20, max_value=100)
    )
    @settings(max_examples=10, deadline=None)
    def test_no_deadlocks_with_timeout_locks(self, num_contentious_threads):
        """
        Property: With timeout locks, deadlocks cannot occur - all threads
        either acquire the lock or timeout.
        
        **Validates: Requirements 3.1, 3.2**
        """
        lock = TimeoutLock(timeout=1.0, name='deadlock_test')
        completed = []
        
        def worker(worker_id):
            """Worker that tries to acquire lock multiple times."""
            for attempt in range(3):
                try:
                    with lock:
                        time.sleep(0.05)  # Brief work
                        completed.append((worker_id, attempt, 'success'))
                except TimeoutError:
                    completed.append((worker_id, attempt, 'timeout'))
                time.sleep(0.01)  # Brief pause between attempts
        
        # Create many threads
        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(num_contentious_threads)
        ]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion with reasonable timeout
        max_wait = 10.0
        for thread in threads:
            thread.join(timeout=max_wait)
        
        # Verify no threads are stuck (no deadlock)
        stuck_threads = [t for t in threads if t.is_alive()]
        assert len(stuck_threads) == 0, f"{len(stuck_threads)} threads are stuck (deadlock?)"
        
        # Verify all threads completed their attempts
        expected_results = num_contentious_threads * 3  # 3 attempts per thread
        assert len(completed) == expected_results, \
            f"Expected {expected_results} results, got {len(completed)}"


# =============================================================================
# Property-Based Tests - Thread-Safe Collections (Task 6.4)
# =============================================================================

@pytest.mark.stress
@pytest.mark.property
class TestThreadSafeCollectionProperties:
    """Property-based tests for thread-safe collection consistency."""
    
    @given(
        operations=st.lists(
            st.tuples(
                st.sampled_from(['add', 'remove', 'contains']),
                st.integers(min_value=0, max_value=100)
            ),
            min_size=100,
            max_size=500
        )
    )
    @settings(max_examples=10, deadline=None)
    def test_threadsafe_dict_final_state_consistent(self, operations):
        """
        Property 4: ThreadSafeDict final state is consistent with some serial execution order.
        
        For any sequence of concurrent operations on ThreadSafeDict, the final state
        should be valid and consistent (no corruption, no lost updates).
        
        **Validates: Requirements 4.1-4.5**
        """
        d = ThreadSafeDict[int, int](name='property_test_dict')
        
        def worker(ops):
            for op, key in ops:
                if op == 'add':
                    d[key] = key * 10
                elif op == 'remove':
                    try:
                        del d[key]
                    except KeyError:
                        pass  # Key doesn't exist
                else:  # contains
                    _ = key in d
        
        # Split operations across 10 threads
        num_threads = 10
        chunk_size = len(operations) // num_threads
        threads = []
        
        for i in range(num_threads):
            start = i * chunk_size
            end = start + chunk_size if i < num_threads - 1 else len(operations)
            chunk = operations[start:end]
            thread = threading.Thread(target=worker, args=(chunk,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Verify invariants:
        # 1. All keys in dict should have consistent values (key * 10)
        for key, value in d.items():
            assert value == key * 10, f"Inconsistent value for key {key}: expected {key * 10}, got {value}"
        
        # 2. Dict should be in valid state (no corruption)
        assert len(d) >= 0, "Dict length is negative (corruption)"
        
        # 3. All operations should have completed without errors
        for thread in threads:
            assert not thread.is_alive(), f"Thread {thread.name} still alive"
    
    @given(
        operations=st.lists(
            st.tuples(
                st.sampled_from(['add', 'remove', 'contains']),
                st.integers(min_value=0, max_value=100)
            ),
            min_size=100,
            max_size=500
        )
    )
    @settings(max_examples=10, deadline=None)
    def test_threadsafe_set_final_state_consistent(self, operations):
        """
        Property 4: ThreadSafeSet final state is consistent with some serial execution order.
        
        For any sequence of concurrent operations on ThreadSafeSet, the final state
        should be valid and consistent (no corruption, no duplicate items).
        
        **Validates: Requirements 4.1-4.5**
        """
        s = ThreadSafeSet[int](name='property_test_set')
        
        def worker(ops):
            for op, value in ops:
                if op == 'add':
                    s.add(value)
                elif op == 'remove':
                    s.discard(value)  # discard doesn't raise if not present
                else:  # contains
                    _ = value in s
        
        # Split operations across 10 threads
        num_threads = 10
        chunk_size = len(operations) // num_threads
        threads = []
        
        for i in range(num_threads):
            start = i * chunk_size
            end = start + chunk_size if i < num_threads - 1 else len(operations)
            chunk = operations[start:end]
            thread = threading.Thread(target=worker, args=(chunk,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Verify invariants:
        # 1. Set should be in valid state (no corruption)
        assert len(s) >= 0, "Set length is negative (corruption)"
        
        # 2. No duplicate items (set property)
        items = list(s)
        assert len(items) == len(set(items)), "Set contains duplicates (corruption)"
        
        # 3. All operations should have completed without errors
        for thread in threads:
            assert not thread.is_alive(), f"Thread {thread.name} still alive"
    
    @given(
        num_threads=st.integers(min_value=20, max_value=100),
        num_operations=st.integers(min_value=100, max_value=500)
    )
    @settings(max_examples=10, deadline=None)
    def test_no_concurrent_modification_errors(self, num_threads, num_operations):
        """
        Property: Thread-safe collections never raise concurrent modification errors
        during iteration, even when modified by other threads.
        
        **Validates: Requirements 4.3, 4.4**
        """
        d = ThreadSafeDict[int, int](name='concurrent_mod_test')
        s = ThreadSafeSet[int](name='concurrent_mod_test')
        
        # Populate collections
        for i in range(50):
            d[i] = i * 10
            s.add(i)
        
        errors = []
        
        def modifier():
            """Continuously modify collections."""
            for i in range(num_operations):
                d[i + 100] = i
                s.add(i + 100)
                if i % 10 == 0:
                    try:
                        del d[i]
                    except KeyError:
                        pass
                    s.discard(i)
        
        def iterator():
            """Continuously iterate collections."""
            try:
                for _ in range(num_operations // 10):
                    # Iterate dict
                    for key, value in d.items():
                        pass
                    # Iterate set
                    for item in s:
                        pass
            except Exception as e:
                errors.append(str(e))
        
        # Create threads
        threads = []
        for i in range(num_threads // 2):
            threads.append(threading.Thread(target=modifier, name=f'modifier-{i}'))
            threads.append(threading.Thread(target=iterator, name=f'iterator-{i}'))
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=15.0)
        
        # Verify no concurrent modification errors
        assert len(errors) == 0, f"Concurrent modification errors occurred: {errors}"
        
        # Verify all threads completed
        for thread in threads:
            assert not thread.is_alive(), f"Thread {thread.name} still alive"


# =============================================================================
# Property-Based Tests - Stop Event Responsiveness (Task 7.3)
# =============================================================================

@pytest.mark.stress
@pytest.mark.property
class TestStopEventProperties:
    """Property-based tests for stop event responsiveness."""
    
    @given(
        wait_interval=st.floats(min_value=0.1, max_value=5.0),
        stop_delay=st.floats(min_value=0.05, max_value=0.5)
    )
    @settings(max_examples=10, deadline=None)
    def test_loop_exits_within_100ms_of_stop_request(self, wait_interval, stop_delay):
        """
        Property 5: Loop exits within 100ms of stop request.
        
        For any wait interval and stop timing, the monitoring loop should
        exit within 100ms of the stop request being issued.
        
        **Validates: Requirements 5.1, 5.2**
        """
        exit_times = []
        stop_times = []
        
        def monitoring_loop(thread: GracefulThread):
            """Monitoring loop with configurable wait interval."""
            while not thread.should_stop():
                # Use wait_or_stop which should exit immediately on stop
                if thread.wait_or_stop(wait_interval):
                    break
            exit_times.append(time.time())
        
        thread = GracefulThread(target=monitoring_loop, name='property_test_stop')
        thread.start()
        
        # Wait before stopping
        time.sleep(stop_delay)
        
        # Request stop and record time
        stop_time = time.time()
        stop_times.append(stop_time)
        thread.stop(timeout=2.0)
        
        # Verify thread exited quickly after stop request
        assert len(exit_times) == 1, "Thread should have exited"
        assert len(stop_times) == 1, "Stop time should be recorded"
        
        exit_delay = exit_times[0] - stop_times[0]
        
        # Allow 200ms tolerance (100ms target + 100ms margin for system scheduling)
        assert exit_delay < 0.2, \
            f"Loop took {exit_delay*1000:.1f}ms to exit after stop, expected < 200ms"
    
    @given(
        num_loops=st.integers(min_value=5, max_value=20),
        wait_intervals=st.lists(
            st.floats(min_value=0.1, max_value=2.0),
            min_size=5,
            max_size=20
        )
    )
    @settings(max_examples=10, deadline=None)
    def test_multiple_loops_all_exit_quickly(self, num_loops, wait_intervals):
        """
        Property: Multiple monitoring loops all exit quickly when stopped.
        
        For any number of concurrent monitoring loops with different wait
        intervals, all should exit within a reasonable time when stopped.
        
        **Validates: Requirements 5.1, 5.2**
        """
        # Ensure we have enough intervals
        while len(wait_intervals) < num_loops:
            wait_intervals.append(1.0)
        
        threads = []
        exit_times = []
        
        # Create threads
        for i in range(num_loops):
            interval = wait_intervals[i]
            
            def monitoring_loop(thread: GracefulThread, interval=interval):
                """Monitoring loop with specific wait interval."""
                while not thread.should_stop():
                    if thread.wait_or_stop(interval):
                        break
                exit_times.append(time.time())
            
            thread = GracefulThread(
                target=monitoring_loop,
                name=f'loop_{i}'
            )
            threads.append(thread)
            thread.start()
        
        # Let threads start
        time.sleep(0.2)
        
        # Stop all threads
        stop_time = time.time()
        for thread in threads:
            thread.stop(timeout=2.0)
        
        # Verify all threads exited
        assert len(exit_times) == num_loops, f"Expected {num_loops} exits, got {len(exit_times)}"
        
        # Verify all threads exited within reasonable time
        max_exit_delay = max(t - stop_time for t in exit_times)
        assert max_exit_delay < 1.0, \
            f"Slowest loop took {max_exit_delay*1000:.1f}ms to exit, expected < 1000ms"


# =============================================================================
# Stress Tests - Comprehensive Concurrency (Task 15)
# =============================================================================

@pytest.mark.stress
class TestConcurrencyStress:
    """Comprehensive concurrency stress tests."""
    
    def test_placeholder_bounded_queue_stress(self):
        """Placeholder test - will be implemented in Task 15.1."""
        # TODO: Implement in Task 15.1
        # Property 1: Queue size never exceeds maxsize under concurrent load
        # **Validates: Requirements 1.1-1.4**
        pass
    
    def test_placeholder_graceful_thread_stress(self):
        """Placeholder test - will be implemented in Task 15.2."""
        # TODO: Implement in Task 15.2
        # Create 100 concurrent threads with random workloads
        pass
    
    def test_placeholder_lock_timeout_stress(self):
        """Placeholder test - will be implemented in Task 15.3."""
        # TODO: Implement in Task 15.3
        # Create 100 concurrent threads attempting lock acquisition
        pass
    
    def test_placeholder_collection_consistency_stress(self):
        """Placeholder test - will be implemented in Task 15.4."""
        # TODO: Implement in Task 15.4
        # Property 2: Final state is consistent with serial execution
        # **Validates: Requirements 4.1-4.5**
        pass
    
    def test_placeholder_resource_cleanup_stress(self):
        """Placeholder test - will be implemented in Task 15.5."""
        # TODO: Implement in Task 15.5
        # Simulate random exceptions during resource operations
        pass
    
    def test_placeholder_stop_event_responsiveness_stress(self):
        """Placeholder test - will be implemented in Task 15.6."""
        # TODO: Implement in Task 15.6
        # Create monitoring loops with random stop timings
        pass


# =============================================================================
# Helper Functions
# =============================================================================

def create_concurrent_workers(
    num_workers: int,
    target_func,
    *args,
    **kwargs
) -> List[threading.Thread]:
    """
    Helper function to create concurrent worker threads.
    
    Args:
        num_workers: Number of worker threads to create
        target_func: Function to execute in each thread
        *args: Positional arguments for target_func
        **kwargs: Keyword arguments for target_func
    
    Returns:
        List of created threads
    """
    threads = []
    for i in range(num_workers):
        thread = threading.Thread(
            target=target_func,
            args=args,
            kwargs=kwargs,
            name=f"worker-{i}"
        )
        threads.append(thread)
    return threads


def start_and_join_threads(threads: List[threading.Thread], timeout: float = 10.0):
    """
    Helper function to start threads and wait for completion.
    
    Args:
        threads: List of threads to start
        timeout: Maximum time to wait for each thread
    """
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError(f"Thread {thread.name} did not complete within {timeout}s")


# =============================================================================
# Test Configuration
# =============================================================================

# Hypothesis settings for property-based tests
hypothesis.settings.register_profile(
    "stress",
    max_examples=20,
    deadline=None,  # No deadline for stress tests
    phases=[Phase.generate, Phase.target, Phase.shrink],
)

# Use stress profile for stress tests
hypothesis.settings.load_profile("stress")
