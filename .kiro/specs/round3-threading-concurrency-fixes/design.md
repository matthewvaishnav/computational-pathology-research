# Design Document: Round 3 Threading and Concurrency Fixes

## Overview

This design document specifies the technical approach for implementing Round 3 threading and concurrency fixes for the HistoCore medical AI system. The fixes address 13 identified threading and concurrency issues to achieve production-grade reliability for a medical AI system that must meet HIPAA, FDA 21 CFR Part 11, and ISO 13485 standards.

### Background

Following successful completion of Round 1 (20 security vulnerabilities) and Round 2 (15 reliability issues), Round 3 focuses on threading and concurrency hardening. The system currently has several threading vulnerabilities:

- **Unbounded queues** that can exhaust memory during high-throughput operations
- **Daemon threads** that are killed without cleanup during shutdown
- **Locks without timeouts** that can cause deadlocks
- **Shared collections without synchronization** that cause race conditions
- **Missing exception handling** in async code that leaks resources
- **Missing resource cleanup** for GPU memory, matplotlib figures, and database connections

### Goals

1. **Eliminate memory exhaustion risks** by replacing unbounded queues with bounded queues
2. **Enable graceful shutdown** by replacing daemon threads with graceful threads
3. **Prevent deadlocks** by adding timeouts to all locks
4. **Eliminate race conditions** by using thread-safe collections
5. **Improve async reliability** by adding proper exception handling and timeouts
6. **Prevent resource leaks** by adding try-finally cleanup for GPU, matplotlib, and database resources
7. **Validate configurations** by adding JSON schema validation
8. **Comprehensive testing** with concurrency stress tests

### Scope

**In Scope:**
- Bounded queue implementation for visualization, alert, and retraining queues
- Graceful thread shutdown for all background threads
- Lock timeout protection for model swap and A/B testing locks
- Thread-safe collections for shared state
- Asyncio exception handling and timeouts for WebSocket operations
- Resource cleanup for SQLite, matplotlib, and GPU memory
- Configuration validation with JSON schemas
- Concurrency stress testing

**Out of Scope:**
- Performance optimization (focus is on correctness and reliability)
- New features or functionality
- UI/UX changes
- Database schema changes

## Architecture

### High-Level Architecture

The threading and concurrency fixes follow a layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (ProgressiveVisualizer, ModelManagement, FailureHandler)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              Thread-Safe Utilities Layer                     │
│  (BoundedQueue, GracefulThread, TimeoutLock,                │
│   ThreadSafeDict, ThreadSafeSet)                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              Python Threading Primitives                     │
│  (threading.Lock, threading.Thread, queue.Queue)            │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Defense in Depth**: Multiple layers of protection (timeouts, bounds, cleanup)
2. **Fail-Safe Defaults**: Safe defaults that prevent common errors
3. **Explicit Resource Management**: Clear ownership and cleanup of resources
4. **Observable Behavior**: Comprehensive logging for debugging
5. **Backward Compatibility**: Changes are drop-in replacements

### Key Design Decisions

**Decision 1: Use Pre-Built Utilities**
- **Rationale**: `src/utils/safe_threading.py` provides battle-tested implementations
- **Alternative Considered**: Implement from scratch
- **Trade-off**: Less flexibility, but higher reliability and consistency

**Decision 2: Drop-Oldest Policy for Bounded Queues**
- **Rationale**: Visualization updates are time-sensitive; old updates are less valuable
- **Alternative Considered**: Block or drop-newest
- **Trade-off**: May lose some updates, but prevents memory exhaustion

**Decision 3: 30-Second Timeout for Locks**
- **Rationale**: Balances responsiveness with tolerance for slow operations
- **Alternative Considered**: 10s (too short), 60s (too long)
- **Trade-off**: May timeout legitimate slow operations

**Decision 4: Non-Daemon Threads by Default**
- **Rationale**: Enables graceful shutdown with resource cleanup
- **Alternative Considered**: Keep daemon threads
- **Trade-off**: Requires explicit shutdown logic

## Components and Interfaces

### Component 1: Bounded Queue System

**Purpose**: Replace unbounded queues with size-limited queues to prevent memory exhaustion.

**Interface**:
```python
class BoundedQueue(Generic[T]):
    def __init__(
        self,
        maxsize: int = 1000,
        drop_policy: str = 'oldest',
        name: str = "unnamed"
    ):
        """Initialize bounded queue."""
        
    def put(self, item: T, timeout: Optional[float] = None) -> bool:
        """Put item in queue. Returns True if added, False if dropped."""
        
    def get(self, timeout: Optional[float] = None) -> T:
        """Get item from queue."""
        
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
```

**Affected Files**:
- `src/streaming/progressive_visualizer.py:71` - visualization queue
- `src/streaming/model_management.py:217` - alert queue
- `src/streaming/model_management.py:332` - retraining queue

**Implementation Strategy**:
1. Import `BoundedQueue` from `src/utils/safe_threading`
2. Replace `Queue()` with `BoundedQueue(maxsize=1000, drop_policy='oldest', name='...')`
3. Update queue consumers to handle potential `Empty` exceptions
4. Add logging for dropped items (every 100 drops)

### Component 2: Graceful Thread System

**Purpose**: Replace daemon threads with graceful threads that support clean shutdown.

**Interface**:
```python
class GracefulThread(threading.Thread):
    def __init__(
        self,
        target: Callable,
        name: str = "unnamed",
        daemon: bool = False,
        cleanup_callback: Optional[Callable] = None
    ):
        """Initialize graceful thread."""
        
    def should_stop(self) -> bool:
        """Check if thread should stop."""
        
    def wait_or_stop(self, interval: float) -> bool:
        """Wait for interval or until stop requested."""
        
    def stop(self, timeout: float = 5.0) -> bool:
        """Request thread to stop and wait for completion."""
```

**Affected Files**:
- `src/streaming/progressive_visualizer.py:88` - visualization thread
- `src/federated/coordinator/failure_handler.py:267` - monitor thread
- `src/federated/production/monitoring.py:587` - monitoring thread
- `src/streaming/model_management.py:230` - drift monitor thread

**Implementation Strategy**:
1. Import `GracefulThread` from `src/utils/safe_threading`
2. Refactor target function to accept `thread: GracefulThread` as first parameter
3. Replace `while self.running:` with `while not thread.should_stop():`
4. Replace `time.sleep(interval)` with `if thread.wait_or_stop(interval): break`
5. Add cleanup callback for resource cleanup
6. Update shutdown logic to call `thread.stop(timeout=5.0)`

### Component 3: Lock Timeout System

**Purpose**: Add timeouts to all locks to prevent deadlocks.

**Interface**:
```python
class TimeoutLock:
    def __init__(self, timeout: float = 30.0, name: str = "unnamed"):
        """Initialize timeout lock."""
        
    def __enter__(self):
        """Acquire lock with timeout."""
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock."""
```

**Affected Files**:
- `src/streaming/model_manager.py:149` - model swap lock
- `src/streaming/model_manager.py:313` - A/B testing lock

**Implementation Strategy**:
1. Import `TimeoutLock` from `src/utils/safe_threading`
2. Replace `threading.Lock()` or `threading.RLock()` with `TimeoutLock(timeout=30.0, name='...')`
3. Wrap lock usage in try-except to handle `TimeoutError`
4. Log timeout errors with context

### Component 4: Thread-Safe Collections

**Purpose**: Replace shared dictionaries and sets with thread-safe versions.

**Interface**:
```python
class ThreadSafeDict(Generic[K, V]):
    def __init__(self, name: str = "unnamed"):
        """Initialize thread-safe dictionary."""
        
    def __setitem__(self, key: K, value: V): ...
    def __getitem__(self, key: K) -> V: ...
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]: ...
    def items(self) -> List[Tuple[K, V]]: ...
    
    @contextmanager
    def lock(self):
        """Context manager for batch operations."""

class ThreadSafeSet(Generic[T]):
    def __init__(self, name: str = "unnamed"):
        """Initialize thread-safe set."""
        
    def add(self, item: T): ...
    def remove(self, item: T): ...
    def __contains__(self, item: T) -> bool: ...
    def __iter__(self): ...
```

**Affected Files**:
- `src/federated/coordinator/failure_handler.py:82-85` - client state dictionaries and sets

**Implementation Strategy**:
1. Import `ThreadSafeDict` and `ThreadSafeSet` from `src/utils/safe_threading`
2. Replace `Dict[K, V] = {}` with `ThreadSafeDict[K, V](name='...')`
3. Replace `Set[T] = set()` with `ThreadSafeSet[T](name='...')`
4. Use `.lock()` context manager for batch operations
5. Update iteration to use copies (already handled by thread-safe collections)

### Component 5: Asyncio Exception Handling

**Purpose**: Properly handle asyncio exceptions to prevent resource leaks.

**Pattern**:
```python
try:
    await asyncio.wait_for(
        websocket.send_json(data),
        timeout=30.0
    )
except asyncio.CancelledError:
    logger.info("WebSocket cancelled")
    raise  # Re-raise for proper cleanup
except asyncio.TimeoutError:
    logger.error("WebSocket send timeout")
    await websocket.close()
except WebSocketDisconnect:
    logger.info("Client disconnected")
except Exception as e:
    logger.error(f"WebSocket error: {e}")
    raise
```

**Affected Files**:
- `src/streaming/interactive_showcase.py:516` - WebSocket exception handling
- All WebSocket operations throughout the codebase

**Implementation Strategy**:
1. Identify all `await websocket.*` operations
2. Wrap with `asyncio.wait_for(operation, timeout=30.0)`
3. Add exception handling hierarchy: `CancelledError` → `TimeoutError` → `WebSocketDisconnect` → `Exception`
4. Re-raise `CancelledError` after logging
5. Close WebSocket on timeout or error

### Component 6: Resource Cleanup Patterns

**Purpose**: Ensure resources are cleaned up in all code paths.

**Pattern 1: SQLite Connection Cleanup**
```python
conn = None
try:
    conn = sqlite3.connect(db_path)
    # ... database operations ...
except Exception as e:
    if conn:
        conn.rollback()
    logger.error(f"Database error: {e}")
    raise
finally:
    if conn:
        conn.close()
```

**Pattern 2: Matplotlib Figure Cleanup**
```python
fig = None
try:
    fig, ax = plt.subplots(figsize=(12, 10))
    # ... plotting ...
    plt.savefig(output_path)
finally:
    if fig is not None:
        plt.close(fig)
```

**Pattern 3: GPU Memory Cleanup**
```python
features = None
try:
    features = model(input_tensor)
    # ... processing ...
finally:
    if features is not None:
        del features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

**Affected Files**:
- All database operations
- `src/streaming/progressive_visualizer.py` - all plotting functions
- All inference loops

**Implementation Strategy**:
1. Identify resource allocation points
2. Initialize resource variable to `None` before try block
3. Add try-finally block around resource usage
4. Clean up resource in finally block with null check
5. Add appropriate error handling in except block

### Component 7: Configuration Validation

**Purpose**: Validate configuration files against JSON schemas at startup.

**Schema Structure**:
```python
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "monitoring_interval_seconds": {
            "type": "number",
            "minimum": 1,
            "maximum": 3600
        },
        "thresholds": {
            "type": "object",
            "properties": {
                "cpu_percent": {"type": "number", "minimum": 0, "maximum": 100},
                "memory_percent": {"type": "number", "minimum": 0, "maximum": 100},
                "disk_percent": {"type": "number", "minimum": 0, "maximum": 100}
            }
        },
        "alert_channels": {
            "type": "object",
            "properties": {
                "slack": {
                    "type": "object",
                    "properties": {
                        "webhook_url": {"type": "string", "pattern": "^https://hooks\\.slack\\.com/"}
                    },
                    "required": ["webhook_url"]
                },
                "email": {
                    "type": "object",
                    "properties": {
                        "smtp_server": {"type": "string"},
                        "username": {"type": "string"},
                        "password": {"type": "string"},
                        "from_email": {"type": "string", "format": "email"},
                        "to_emails": {"type": "array", "items": {"type": "string", "format": "email"}}
                    },
                    "required": ["smtp_server", "username", "password", "from_email", "to_emails"]
                }
            }
        }
    },
    "required": ["monitoring_interval_seconds"]
}
```

**Affected Files**:
- `src/federated/production/monitoring.py:540` - configuration loading

**Implementation Strategy**:
1. Import `jsonschema`
2. Define `CONFIG_SCHEMA` constant with complete schema
3. Add validation after loading config: `jsonschema.validate(user_config, CONFIG_SCHEMA)`
4. Catch `jsonschema.ValidationError` and raise `ValueError` with clear message
5. Add alert channel validation (Slack webhook test, email config check)

## Data Models

### Queue Statistics Model

```python
@dataclass
class QueueStats:
    """Statistics for a bounded queue."""
    name: str
    size: int
    maxsize: int
    dropped_count: int
    drop_policy: str
```

### Thread Status Model

```python
@dataclass
class ThreadStatus:
    """Status of a graceful thread."""
    name: str
    is_alive: bool
    should_stop: bool
    exception: Optional[Exception]
```

### Lock Metrics Model

```python
@dataclass
class LockMetrics:
    """Metrics for a timeout lock."""
    name: str
    owner: Optional[str]
    acquire_time: Optional[float]
    hold_time: float
```

### Configuration Schema Model

```python
@dataclass
class MonitoringConfig:
    """Validated monitoring configuration."""
    monitoring_interval_seconds: int
    thresholds: Dict[str, float]
    alert_channels: Dict[str, Dict[str, Any]]
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'MonitoringConfig':
        """Create from validated dictionary."""
        jsonschema.validate(config, CONFIG_SCHEMA)
        return cls(**config)
```

## Error Handling

### Error Categories

1. **Timeout Errors**: Lock acquisition timeout, async operation timeout
2. **Queue Errors**: Queue full (handled by drop policy)
3. **Thread Errors**: Thread failed to stop, thread exception
4. **Resource Errors**: Failed to cleanup resource
5. **Configuration Errors**: Invalid configuration, failed validation

### Error Handling Strategy

**Timeout Errors**:
```python
try:
    with timeout_lock:
        # Critical section
        pass
except TimeoutError as e:
    logger.error(f"Lock timeout: {e}")
    # Raise to caller - this is a serious error
    raise
```

**Queue Errors**:
```python
# Handled by drop policy - log warning every 100 drops
if dropped_count % 100 == 0:
    logger.warning(f"Queue '{name}' dropped {dropped_count} items")
```

**Thread Errors**:
```python
if not thread.stop(timeout=5.0):
    logger.warning(f"Thread '{thread.name}' did not stop within timeout")
    # Continue - thread will be garbage collected
```

**Resource Errors**:
```python
finally:
    try:
        if resource is not None:
            resource.cleanup()
    except Exception as e:
        logger.error(f"Failed to cleanup resource: {e}")
        # Don't raise - cleanup is best-effort
```

**Configuration Errors**:
```python
try:
    jsonschema.validate(config, CONFIG_SCHEMA)
except jsonschema.ValidationError as e:
    logger.error(f"Invalid configuration: {e}")
    raise ValueError(f"Configuration validation failed: {e}")
```

### Logging Strategy

**Log Levels**:
- **DEBUG**: Lock acquire/release, queue operations
- **INFO**: Thread start/stop, configuration loaded
- **WARNING**: Lock held >5s, queue dropped items, thread didn't stop
- **ERROR**: Lock timeout, async timeout, resource cleanup failed, configuration invalid

**Log Format**:
```python
logger.info(f"Thread '{thread_name}' started")
logger.warning(f"Lock '{lock_name}' held for {hold_time:.2f}s by {owner}")
logger.error(f"Failed to acquire lock '{lock_name}' within {timeout}s")
```

## Testing Strategy

### Unit Testing

**Test Categories**:
1. **Bounded Queue Tests**: Test maxsize enforcement, drop policies, statistics
2. **Graceful Thread Tests**: Test shutdown, cleanup callbacks, timeout
3. **Timeout Lock Tests**: Test timeout, hold time warnings, deadlock detection
4. **Thread-Safe Collection Tests**: Test concurrent access, iteration safety
5. **Resource Cleanup Tests**: Test cleanup in success and error paths

**Example Unit Tests**:
```python
def test_bounded_queue_drops_oldest():
    """Test that bounded queue drops oldest item when full."""
    queue = BoundedQueue(maxsize=5, drop_policy='oldest', name='test')
    
    # Fill queue
    for i in range(5):
        assert queue.put(i, timeout=0.1)
    
    # Add one more - should drop oldest (0)
    assert queue.put(5, timeout=0.1)
    
    # Verify oldest was dropped
    assert queue.get(timeout=0.1) == 1  # 0 was dropped
    
def test_graceful_thread_stops_within_timeout():
    """Test that graceful thread stops within timeout."""
    def worker(thread: GracefulThread):
        while not thread.should_stop():
            if thread.wait_or_stop(0.1):
                break
    
    thread = GracefulThread(target=worker, name='test')
    thread.start()
    
    # Stop thread
    assert thread.stop(timeout=1.0)
    assert not thread.is_alive()
    
def test_timeout_lock_raises_on_timeout():
    """Test that timeout lock raises TimeoutError."""
    lock = TimeoutLock(timeout=0.1, name='test')
    
    # Acquire lock in one thread
    with lock:
        # Try to acquire in another thread - should timeout
        def try_acquire():
            with pytest.raises(TimeoutError):
                with lock:
                    pass
        
        thread = threading.Thread(target=try_acquire)
        thread.start()
        thread.join()
```

### Integration Testing

**Test Scenarios**:
1. **WebSocket Disconnect Handling**: Test that WebSocket disconnects are handled gracefully
2. **Database Error Handling**: Test that database errors trigger rollback and cleanup
3. **Configuration Validation**: Test that invalid configurations are rejected

**Example Integration Tests**:
```python
@pytest.mark.asyncio
async def test_websocket_disconnect_cleanup():
    """Test that WebSocket disconnect triggers cleanup."""
    # Create mock WebSocket that disconnects
    websocket = MockWebSocket()
    websocket.disconnect_after(1.0)
    
    # Start streaming
    handler = WebSocketHandler(websocket)
    
    # Verify cleanup occurred
    await handler.stream_data()
    assert websocket.is_closed()
    assert handler.resources_cleaned_up()
```

### Concurrency Stress Testing

**Test Approach**: Property-based testing with Hypothesis to generate random concurrent operation sequences.

**Stress Test Categories**:
1. **Queue Stress Test**: 100 concurrent producers/consumers
2. **Thread Stress Test**: 100 concurrent thread start/stop operations
3. **Lock Stress Test**: 100 concurrent lock acquisitions
4. **Collection Stress Test**: 100 concurrent read/write operations

**Example Stress Tests**:
```python
from hypothesis import given, strategies as st
import hypothesis

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
@hypothesis.settings(max_examples=100)
def test_bounded_queue_concurrent_stress(operations):
    """
    Property: For any sequence of concurrent queue operations,
    queue size never exceeds maxsize.
    
    **Feature: round3-threading-concurrency-fixes, Property 1**
    """
    queue = BoundedQueue(maxsize=100, drop_policy='oldest', name='stress_test')
    
    def worker(ops):
        for op, value in ops:
            if op == 'put':
                queue.put(value, timeout=0.1)
            else:
                try:
                    queue.get(timeout=0.1)
                except Empty:
                    pass
    
    # Split operations across threads
    threads = []
    chunk_size = len(operations) // 10
    for i in range(10):
        chunk = operations[i*chunk_size:(i+1)*chunk_size]
        thread = threading.Thread(target=worker, args=(chunk,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Verify invariant
    assert queue.qsize() <= queue.maxsize

@given(
    operations=st.lists(
        st.tuples(
            st.sampled_from(['add', 'remove', 'contains']),
            st.text(min_size=1, max_size=10)
        ),
        min_size=100,
        max_size=1000
    )
)
@hypothesis.settings(max_examples=100)
def test_thread_safe_dict_concurrent_stress(operations):
    """
    Property: For any sequence of concurrent dictionary operations,
    final state is consistent with some serial execution order.
    
    **Feature: round3-threading-concurrency-fixes, Property 2**
    """
    d = ThreadSafeDict(name='stress_test')
    
    def worker(ops):
        for op, key in ops:
            if op == 'add':
                d[key] = key
            elif op == 'remove':
                try:
                    del d[key]
                except KeyError:
                    pass
            else:  # contains
                _ = key in d
    
    # Split operations across threads
    threads = []
    chunk_size = len(operations) // 10
    for i in range(10):
        chunk = operations[i*chunk_size:(i+1)*chunk_size]
        thread = threading.Thread(target=worker, args=(chunk,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Verify consistency: all keys in dict should be valid
    for key in d.keys():
        assert isinstance(key, str)
        assert len(key) >= 1
```

### Test Configuration

**Pytest Configuration**:
```ini
[pytest]
markers =
    unit: Unit tests
    integration: Integration tests
    stress: Concurrency stress tests
    slow: Slow-running tests

# Property-based test settings
hypothesis_profile = default
    max_examples = 100
    deadline = None  # No deadline for stress tests
```

**Test Execution**:
```bash
# Run all tests
pytest tests/test_threading_fixes.py -v

# Run only unit tests
pytest tests/test_threading_fixes.py -m unit -v

# Run only stress tests
pytest tests/test_threading_fixes.py -m stress -v

# Run with coverage
pytest tests/test_threading_fixes.py --cov=src/utils/safe_threading --cov-report=html
```

## Implementation Plan

### Phase 1: Bounded Queues (Day 1-2)
1. Update `src/streaming/progressive_visualizer.py:71`
2. Update `src/streaming/model_management.py:217`
3. Update `src/streaming/model_management.py:332`
4. Write unit tests for bounded queue behavior
5. Verify queue statistics logging

### Phase 2: Graceful Threads (Day 3-4)
1. Update `src/streaming/progressive_visualizer.py:88`
2. Update `src/federated/coordinator/failure_handler.py:267`
3. Update `src/federated/production/monitoring.py:587`
4. Update `src/streaming/model_management.py:230`
5. Write unit tests for graceful shutdown
6. Test with Ctrl+C shutdown

### Phase 3: Lock Timeouts (Day 5)
1. Update `src/streaming/model_manager.py:149`
2. Update `src/streaming/model_manager.py:313`
3. Write unit tests for timeout behavior
4. Add timeout error handling

### Phase 4: Thread-Safe Collections (Day 6)
1. Update `src/federated/coordinator/failure_handler.py:82-85`
2. Write unit tests for concurrent access
3. Run stress tests

### Phase 5: Async Fixes (Day 7-8)
1. Update `src/streaming/interactive_showcase.py:516`
2. Add timeouts to all WebSocket operations
3. Write integration tests for disconnect handling
4. Test timeout behavior

### Phase 6: Resource Cleanup (Day 9-10)
1. Add try-finally to all SQLite operations
2. Add try-finally to all matplotlib plotting
3. Add try-finally to all GPU inference loops
4. Write tests for cleanup in error paths

### Phase 7: Configuration Validation (Day 11)
1. Define JSON schema for monitoring config
2. Add validation to config loading
3. Add alert channel validation
4. Write tests for invalid configs

### Phase 8: Testing (Day 12-14)
1. Write comprehensive unit tests
2. Write integration tests
3. Write concurrency stress tests
4. Run full test suite
5. Fix any issues found

### Phase 9: Documentation and Review (Day 15)
1. Update implementation guide
2. Update API documentation
3. Code review
4. Final testing

## Deployment Strategy

### Rollout Plan

**Phase 1: Development Environment**
1. Deploy to development environment
2. Run full test suite
3. Manual testing of key workflows
4. Monitor logs for warnings/errors

**Phase 2: Staging Environment**
1. Deploy to staging environment
2. Run integration tests
3. Load testing with realistic workloads
4. Monitor resource usage (memory, threads, locks)

**Phase 3: Production Environment**
1. Deploy to production with feature flag
2. Enable for 10% of traffic
3. Monitor metrics and logs
4. Gradually increase to 100%

### Rollback Plan

**Trigger Conditions**:
- Increased error rate (>1% increase)
- Memory leaks detected
- Deadlocks detected
- Performance degradation (>10% slower)

**Rollback Steps**:
1. Disable feature flag
2. Revert to previous version
3. Investigate root cause
4. Fix and redeploy

### Monitoring

**Key Metrics**:
- Queue size and dropped item count
- Thread count and thread stop failures
- Lock acquisition time and timeout count
- Memory usage (heap, GPU)
- Error rate by category

**Alerts**:
- Queue dropped >1000 items in 1 minute
- Thread failed to stop within timeout
- Lock timeout occurred
- Memory usage >90%
- Error rate >1%

## Appendix

### File Modification Summary

| File | Lines | Changes |
|------|-------|---------|
| `src/streaming/progressive_visualizer.py` | 71, 88, 239-297 | Bounded queue, graceful thread, matplotlib cleanup |
| `src/streaming/model_management.py` | 217, 230, 332 | Bounded queues, graceful thread |
| `src/streaming/model_manager.py` | 149, 313 | Timeout locks |
| `src/federated/coordinator/failure_handler.py` | 82-85, 267, 285 | Thread-safe collections, graceful thread, stop event checking |
| `src/federated/production/monitoring.py` | 540, 587 | Config validation, graceful thread |
| `src/streaming/interactive_showcase.py` | 516 | Async exception handling |
| All WebSocket operations | Various | Async timeouts |
| All database operations | Various | SQLite cleanup |
| All inference loops | Various | GPU cleanup |

### Dependencies

**Required Packages**:
- `jsonschema>=4.0.0` - Configuration validation
- `hypothesis>=6.0.0` - Property-based testing (dev dependency)

**Python Version**: Python 3.8+

### References

- [Python Threading Documentation](https://docs.python.org/3/library/threading.html)
- [Python Asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [JSON Schema Documentation](https://json-schema.org/)
- `ROUND3_IMPLEMENTATION_GUIDE.md` - Implementation patterns
- `src/utils/safe_threading.py` - Thread-safe utilities

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-08  
**Status**: Ready for Review
