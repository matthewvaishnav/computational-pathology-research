# Implementation Plan: Round 3 Threading and Concurrency Fixes

## Overview

This implementation plan breaks down the Round 3 threading and concurrency fixes into actionable coding tasks. The fixes address 13 identified threading and concurrency issues across 8 files, implementing bounded queues, graceful thread shutdown, lock timeouts, thread-safe collections, async exception handling, resource cleanup, and configuration validation.

The tasks are organized to enable parallel execution where possible, with two agents working on independent components simultaneously. Each task references specific requirements and includes clear implementation steps.

## Tasks

- [x] 1. Set up testing infrastructure and dependencies
  - Install `jsonschema>=4.0.0` for configuration validation
  - Install `hypothesis>=6.0.0` as dev dependency for property-based testing
  - Configure pytest markers for unit, integration, and stress tests
  - Create `tests/test_threading_fixes.py` test file structure
  - _Requirements: 13.1-13.7_

- [x] 2. Implement bounded queue system (Agent 1)
  - [x] 2.1 Replace visualization queue with BoundedQueue
    - Update `src/streaming/progressive_visualizer.py:71`
    - Import `BoundedQueue` from `src/utils/safe_threading`
    - Replace `Queue()` with `BoundedQueue(maxsize=1000, drop_policy='oldest', name='visualization_queue')`
    - Add logging for dropped items (every 100 drops)
    - _Requirements: 1.1, 1.2, 1.5_
  
  - [x] 2.2 Replace alert queue with BoundedQueue
    - Update `src/streaming/model_management.py:217`
    - Import `BoundedQueue` from `src/utils/safe_threading`
    - Replace `Queue()` with `BoundedQueue(maxsize=1000, drop_policy='oldest', name='alert_queue')`
    - Add logging for dropped items (every 100 drops)
    - _Requirements: 1.1, 1.3, 1.5_
  
  - [x] 2.3 Replace retraining queue with BoundedQueue
    - Update `src/streaming/model_management.py:332`
    - Import `BoundedQueue` from `src/utils/safe_threading`
    - Replace `Queue()` with `BoundedQueue(maxsize=1000, drop_policy='oldest', name='retraining_queue')`
    - Add logging for dropped items (every 100 drops)
    - _Requirements: 1.1, 1.4, 1.5_
  
  - [x] 2.4 Add queue statistics monitoring
    - Implement `get_stats()` calls for all three queues
    - Add periodic logging of queue statistics (size, maxsize, dropped_count)
    - _Requirements: 1.6_
  
  - [x] 2.5 Write unit tests for bounded queues
    - Test maxsize enforcement
    - Test drop-oldest policy
    - Test queue statistics
    - Test concurrent producer/consumer scenarios
    - _Requirements: 1.1-1.6_
  
  - [x] 2.6 Write property test for bounded queue invariant
    - **Property 1: Queue size never exceeds maxsize**
    - **Validates: Requirements 1.1-1.4**
    - Use Hypothesis to generate random sequences of put/get operations
    - Verify queue size ≤ maxsize after all operations
    - Test with 100 concurrent threads
    - _Requirements: 13.1, 13.2_

- [x] 3. Implement graceful thread shutdown (Agent 2)
  - [x] 3.1 Replace visualization thread with GracefulThread
    - Update `src/streaming/progressive_visualizer.py:88`
    - Import `GracefulThread` from `src/utils/safe_threading`
    - Refactor target function to accept `thread: GracefulThread` parameter
    - Replace `while self.running:` with `while not thread.should_stop():`
    - Replace `time.sleep(interval)` with `if thread.wait_or_stop(interval): break`
    - Add cleanup callback for resource cleanup
    - Update shutdown logic to call `thread.stop(timeout=5.0)`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6_
  
  - [x] 3.2 Replace failure handler monitor thread with GracefulThread
    - Update `src/federated/coordinator/failure_handler.py:267`
    - Import `GracefulThread` from `src/utils/safe_threading`
    - Refactor target function to accept `thread: GracefulThread` parameter
    - Replace daemon thread logic with graceful shutdown
    - Add cleanup callback
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6_
  
  - [x] 3.3 Replace production monitoring thread with GracefulThread
    - Update `src/federated/production/monitoring.py:587`
    - Import `GracefulThread` from `src/utils/safe_threading`
    - Refactor target function to accept `thread: GracefulThread` parameter
    - Replace daemon thread logic with graceful shutdown
    - Add cleanup callback
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6_
  
  - [x] 3.4 Replace drift monitor thread with GracefulThread
    - Update `src/streaming/model_management.py:230`
    - Import `GracefulThread` from `src/utils/safe_threading`
    - Refactor target function to accept `thread: GracefulThread` parameter
    - Replace daemon thread logic with graceful shutdown
    - Add cleanup callback
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6_
  
  - [x] 3.5 Write unit tests for graceful thread shutdown
    - Test thread stops within timeout
    - Test cleanup callbacks are executed
    - Test thread stop logging
    - Test timeout warning when thread doesn't stop
    - _Requirements: 2.1-2.6_
  
  - [x] 3.6 Write property test for graceful thread behavior
    - **Property 2: Threads stop within timeout period**
    - **Validates: Requirements 2.4, 2.5**
    - Use Hypothesis to generate random thread workloads
    - Verify all threads stop within 5 seconds
    - Test with 100 concurrent threads
    - _Requirements: 13.3_

- [x] 4. Checkpoint - Verify bounded queues and graceful threads
  - Ensure all tests pass for bounded queues and graceful threads
  - Manually test shutdown behavior with Ctrl+C
  - Review logs for proper queue statistics and thread lifecycle events
  - Ask the user if questions arise

- [x] 5. Implement lock timeout protection (Agent 1)
  - [x] 5.1 Replace model swap lock with TimeoutLock
    - Update `src/streaming/model_manager.py:149`
    - Import `TimeoutLock` from `src/utils/safe_threading`
    - Replace `threading.Lock()` with `TimeoutLock(timeout=30.0, name='model_swap_lock')`
    - Wrap lock usage in try-except to handle `TimeoutError`
    - Log timeout errors with context
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [x] 5.2 Replace A/B testing lock with TimeoutLock
    - Update `src/streaming/model_manager.py:313`
    - Import `TimeoutLock` from `src/utils/safe_threading`
    - Replace `threading.Lock()` with `TimeoutLock(timeout=30.0, name='ab_testing_lock')`
    - Wrap lock usage in try-except to handle `TimeoutError`
    - Log timeout errors with context
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [x] 5.3 Write unit tests for lock timeouts
    - Test lock timeout raises TimeoutError
    - Test lock hold time warnings (>5 seconds)
    - Test lock acquisition/release logging
    - Test concurrent lock acquisition
    - _Requirements: 3.1-3.5_
  
  - [x] 5.4 Write property test for lock timeout behavior
    - **Property 3: Lock acquisition fails with TimeoutError after timeout**
    - **Validates: Requirements 3.2**
    - Use Hypothesis to generate random lock hold times
    - Verify TimeoutError is raised when timeout exceeded
    - Test with 100 concurrent threads attempting acquisition
    - _Requirements: 13.4_

- [x] 6. Implement thread-safe collections (Agent 2)
  - [x] 6.1 Replace client state dictionaries with ThreadSafeDict
    - Update `src/federated/coordinator/failure_handler.py:82-85`
    - Import `ThreadSafeDict` from `src/utils/safe_threading`
    - Replace `Dict[str, Any] = {}` with `ThreadSafeDict[str, Any](name='client_state')`
    - Replace `Dict[str, float] = {}` with `ThreadSafeDict[str, float](name='last_heartbeat')`
    - Update batch operations to use `.lock()` context manager
    - _Requirements: 4.1, 4.3, 4.4, 4.5_
  
  - [x] 6.2 Replace client state sets with ThreadSafeSet
    - Update `src/federated/coordinator/failure_handler.py:82-85`
    - Import `ThreadSafeSet` from `src/utils/safe_threading`
    - Replace `Set[str] = set()` with `ThreadSafeSet[str](name='failed_clients')`
    - Replace `Set[str] = set()` with `ThreadSafeSet[str](name='recovering_clients')`
    - Update iteration to use copies (handled automatically)
    - _Requirements: 4.2, 4.3, 4.4, 4.5_
  
  - [x] 6.3 Write unit tests for thread-safe collections
    - Test concurrent read/write operations
    - Test iteration safety (no concurrent modification errors)
    - Test batch operations with lock context manager
    - Test get/set/add/remove operations
    - _Requirements: 4.1-4.5_
  
  - [x] 6.4 Write property test for thread-safe collection consistency
    - **Property 4: Final state is consistent with some serial execution order**
    - **Validates: Requirements 4.1-4.5**
    - Use Hypothesis to generate random sequences of operations
    - Verify final state is valid (no corruption)
    - Test with 100 concurrent threads
    - _Requirements: 13.5_

- [x] 7. Implement stop event return value checking (Agent 1)
  - [x] 7.1 Update failure handler monitoring loop
    - Update `src/federated/coordinator/failure_handler.py:285`
    - Check return value of `stop_event.wait()`
    - Exit loop immediately if `stop_event.wait()` returns True
    - Add `stop_event.is_set()` check in loop condition
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [x] 7.2 Write unit tests for stop event checking
    - Test loop exits immediately when stop is requested
    - Test loop does not wait full timeout when stopped early
    - Test stop_event.is_set() is checked in loop condition
    - _Requirements: 5.1-5.4_
  
  - [x] 7.3 Write property test for stop event responsiveness
    - **Property 5: Loop exits within 100ms of stop request**
    - **Validates: Requirements 5.1, 5.2**
    - Use Hypothesis to generate random stop timings
    - Verify loop exits quickly after stop
    - _Requirements: 13.7_

- [x] 8. Checkpoint - Verify locks, collections, and stop events
  - Ensure all tests pass for locks, collections, and stop events
  - Run stress tests to verify concurrent behavior
  - Review logs for proper lock metrics and collection operations
  - Ask the user if questions arise

- [x] 9. Implement asyncio exception handling (Agent 2)
  - [x] 9.1 Add proper exception handling to WebSocket operations
    - Update `src/streaming/interactive_showcase.py:516`
    - Add exception handling hierarchy: `CancelledError` → `TimeoutError` → `WebSocketDisconnect` → `Exception`
    - Catch `asyncio.CancelledError` separately and re-raise after logging
    - Catch `WebSocketDisconnect` separately and log
    - Log all exception types with appropriate severity
    - Remove any bare except clauses
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [x] 9.2 Add timeouts to all WebSocket send operations
    - Identify all `await websocket.send_json()` calls
    - Wrap with `asyncio.wait_for(websocket.send_json(data), timeout=30.0)`
    - Add `asyncio.TimeoutError` handling
    - Close WebSocket on timeout
    - _Requirements: 7.1, 7.3, 7.4_
  
  - [x] 9.3 Add timeouts to all WebSocket receive operations
    - Identify all `await websocket.receive_text()` calls
    - Wrap with `asyncio.wait_for(websocket.receive_text(), timeout=30.0)`
    - Add `asyncio.TimeoutError` handling
    - Close WebSocket on timeout
    - _Requirements: 7.2, 7.3, 7.4_
  
  - [x] 9.4 Add timeouts to all async HTTP calls
    - Identify all async HTTP operations
    - Wrap with `asyncio.wait_for(operation, timeout=30.0)`
    - Add timeout error handling
    - _Requirements: 7.5_
  
  - [x] 9.5 Write integration tests for WebSocket exception handling
    - Test CancelledError is re-raised
    - Test WebSocketDisconnect is handled gracefully
    - Test timeout closes connection
    - Test resource cleanup on exception
    - _Requirements: 6.1-6.5, 7.1-7.5_

- [x] 10. Implement SQLite connection cleanup (Agent 1)
  - [x] 10.1 Add try-finally blocks to all SQLite operations
    - Identify all `sqlite3.connect()` calls throughout codebase
    - Initialize `conn = None` before try block
    - Add try-finally block around database operations
    - Add rollback in except block
    - Close connection in finally block with null check
    - Log database errors with operation context
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [x] 10.2 Write unit tests for SQLite cleanup
    - Test connection is closed on success
    - Test connection is closed on exception
    - Test rollback occurs on exception
    - Test cleanup with null connection
    - _Requirements: 8.1-8.5, 13.6_

- [x] 11. Implement matplotlib figure cleanup (Agent 2)
  - [x] 11.1 Add try-finally blocks to all matplotlib plotting functions
    - Update `src/streaming/progressive_visualizer.py:239-297` (all plotting functions)
    - Initialize `fig = None` before try block
    - Add try-finally block around plotting operations
    - Close figure in finally block with null check: `if fig is not None: plt.close(fig)`
    - Apply pattern to all functions that create matplotlib figures
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  
  - [x] 11.2 Write unit tests for matplotlib cleanup
    - Test figure is closed on success
    - Test figure is closed on exception
    - Test cleanup with null figure
    - _Requirements: 9.1-9.5, 13.6_

- [x] 12. Implement GPU memory cleanup (Agent 1)
  - [x] 12.1 Add try-finally blocks to all GPU inference loops
    - Identify all GPU tensor operations in inference code
    - Initialize tensor variables to `None` before try block
    - Add try-finally block around tensor operations
    - Delete tensor variables in finally block with null check
    - Call `torch.cuda.empty_cache()` after deleting tensors (if GPU available)
    - Apply pattern to all inference loops and model evaluation code
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  
  - [x] 12.2 Write unit tests for GPU memory cleanup
    - Test tensors are deleted on success
    - Test tensors are deleted on exception
    - Test empty_cache is called when GPU available
    - Test cleanup with null tensors
    - _Requirements: 10.1-10.5, 13.6_

- [x] 13. Checkpoint - Verify async handling and resource cleanup
  - Ensure all tests pass for async operations and resource cleanup
  - Manually test WebSocket disconnect scenarios
  - Monitor memory usage during inference to verify GPU cleanup
  - Review logs for proper exception handling
  - Ask the user if questions arise

- [x] 14. Implement configuration validation (Agent 2)
  - [x] 14.1 Define JSON schema for monitoring configuration
    - Update `src/federated/production/monitoring.py:540`
    - Define `CONFIG_SCHEMA` constant with complete schema
    - Include type constraints, minimum/maximum values, required fields
    - Add pattern validation for Slack webhook URLs
    - Add format validation for email addresses
    - _Requirements: 11.1, 11.4, 11.5_
  
  - [x] 14.2 Add configuration validation to config loading
    - Import `jsonschema`
    - Add validation after loading config: `jsonschema.validate(user_config, CONFIG_SCHEMA)`
    - Catch `jsonschema.ValidationError` and raise `ValueError` with clear message
    - Log configuration validation errors with specific field and constraint
    - _Requirements: 11.2, 11.3, 11.6_
  
  - [x] 14.3 Add alert channel validation
    - Validate Slack webhook URLs start with "https://hooks.slack.com/"
    - Test Slack webhooks by sending test message during configuration
    - Log warning with HTTP status code if webhook test fails
    - Validate email configurations contain all required fields
    - Validate to_emails is a list type
    - Raise ValueError with missing or invalid field if validation fails
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_
  
  - [x] 14.4 Write unit tests for configuration validation
    - Test valid configuration is accepted
    - Test invalid configuration is rejected
    - Test missing required fields are detected
    - Test out-of-range values are rejected
    - Test Slack webhook validation
    - Test email configuration validation
    - _Requirements: 11.1-11.6, 12.1-12.6_

- [x] 15. Implement comprehensive concurrency stress tests (Both Agents)
  - [x] 15.1 Write stress test for bounded queue concurrency
    - **Property 1: Queue size never exceeds maxsize under concurrent load**
    - **Validates: Requirements 1.1-1.4**
    - Create 100 concurrent producer/consumer threads
    - Generate random sequences of put/get operations with Hypothesis
    - Verify queue size ≤ maxsize throughout execution
    - Verify dropped item count increases monotonically
    - Run with max_examples=100
    - _Requirements: 13.1, 13.2_
  
  - [x] 15.2 Write stress test for graceful thread shutdown
    - Create 100 concurrent threads with random workloads
    - Request stop and verify all threads stop within timeout
    - Verify cleanup callbacks are executed
    - Test with various stop timings
    - _Requirements: 13.3_
  
  - [x] 15.3 Write stress test for lock timeout behavior
    - Create 100 concurrent threads attempting lock acquisition
    - Simulate deadlock scenarios
    - Verify TimeoutError is raised appropriately
    - Verify no actual deadlocks occur
    - _Requirements: 13.4_
  
  - [x] 15.4 Write stress test for thread-safe collection consistency
    - **Property 2: Final state is consistent with serial execution**
    - **Validates: Requirements 4.1-4.5**
    - Create 100 concurrent threads performing random operations
    - Generate random sequences of add/remove/contains operations with Hypothesis
    - Verify final state is valid and consistent
    - Verify no concurrent modification errors
    - Run with max_examples=100
    - _Requirements: 13.5_
  
  - [x] 15.5 Write stress test for resource cleanup under exceptions
    - Simulate random exceptions during resource operations
    - Verify cleanup occurs in all code paths
    - Test SQLite, matplotlib, and GPU cleanup
    - Verify no resource leaks
    - _Requirements: 13.6_
  
  - [x] 15.6 Write stress test for stop event responsiveness
    - Create monitoring loops with random stop timings
    - Verify loops exit immediately when stop is requested
    - Verify loops don't wait full timeout period
    - Test with various interval values
    - _Requirements: 13.7_

- [x] 16. Final integration testing and validation
  - [x] 16.1 Run full test suite
    - Run all unit tests: `pytest tests/test_threading_fixes.py -m unit -v`
    - Run all integration tests: `pytest tests/test_threading_fixes.py -m integration -v`
    - Run all stress tests: `pytest tests/test_threading_fixes.py -m stress -v`
    - Verify all tests pass
    - _Requirements: 13.1-13.7_
  
  - [x] 16.2 Run test suite with coverage
    - Run with coverage: `pytest tests/test_threading_fixes.py --cov=src --cov-report=html`
    - Verify coverage >90% for modified files
    - Review coverage report for gaps
    - _Requirements: 13.1-13.7_
  
  - [x] 16.3 Manual testing of key workflows
    - Test system shutdown with Ctrl+C (verify graceful shutdown)
    - Test high-throughput scenarios (verify bounded queues)
    - Test WebSocket disconnect scenarios (verify async handling)
    - Monitor resource usage during long-running operations
    - _Requirements: 1.1-13.7_
  
  - [x] 16.4 Review logs and metrics
    - Verify queue statistics are logged correctly
    - Verify thread lifecycle events are logged
    - Verify lock metrics are logged
    - Verify exception handling logs are appropriate
    - _Requirements: 1.5, 1.6, 2.6, 3.3, 3.4, 6.4, 8.4, 11.6_

- [x] 17. Final checkpoint - Complete implementation
  - Ensure all tests pass with >90% coverage
  - Verify all 13 requirements are fully implemented
  - Review code for consistency and best practices
  - Update documentation if needed
  - Ask the user if questions arise or if ready for deployment

## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP
- Tasks are organized to enable parallel execution: Agent 1 focuses on queues, locks, SQLite, and GPU cleanup; Agent 2 focuses on threads, collections, async handling, matplotlib, and configuration
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties using Hypothesis
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflows
- Stress tests validate behavior under high concurrency (100 threads)
- All resource cleanup follows try-finally pattern to ensure cleanup in all code paths
- All async operations have 30-second timeouts to prevent hung connections
- All locks have 30-second timeouts to prevent deadlocks
- All queues have maxsize=1000 with drop-oldest policy to prevent memory exhaustion
- Configuration validation uses JSON schemas to catch errors at startup

## Parallel Execution Strategy

**Agent 1 Tasks**: 2 (Bounded Queues), 5 (Lock Timeouts), 7 (Stop Events), 10 (SQLite Cleanup), 12 (GPU Cleanup)

**Agent 2 Tasks**: 3 (Graceful Threads), 6 (Thread-Safe Collections), 9 (Async Handling), 11 (Matplotlib Cleanup), 14 (Configuration Validation)

**Shared Tasks**: 1 (Setup), 15 (Stress Tests), 16 (Integration Testing), 17 (Final Checkpoint)

This organization minimizes conflicts and enables efficient parallel development.
