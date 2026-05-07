"""
Property-based tests for retry logic in the Competitor Benchmark System.

Feature: competitor-benchmark-system
Property 4: Exponential Backoff Retry Pattern

**Validates: Requirement 8.2**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from experiments.benchmark_system.error_handler import ErrorHandler, ErrorContext, ErrorCategory


class TestExponentialBackoffProperties:
    """
    Property-based tests for exponential backoff retry pattern.
    
    Property 4: Exponential Backoff Retry Pattern
    
    For any transient failure requiring retry, the retry delays SHALL follow
    an exponential backoff pattern where delay(n) = base_delay * 2^n for
    retry attempt n.
    
    **Validates: Requirement 8.2**
    """
    
    @given(
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        retry_count=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=100)
    def test_exponential_backoff_formula(self, base_delay, retry_count):
        """
        Property: Delay follows exponential pattern delay(n) = base_delay * 2^n.
        
        For any base delay and retry count, the calculated backoff delay
        should follow the exponential formula (before capping).
        """
        handler = ErrorHandler(base_delay=base_delay, max_delay=float('inf'))
        
        # Calculate delay
        delay = handler.calculate_backoff_delay(retry_count)
        
        # Expected delay: base_delay * 2^retry_count
        expected_delay = base_delay * (2 ** retry_count)
        
        # Should match exponential formula
        assert abs(delay - expected_delay) < 1e-6, (
            f"Delay should follow exponential pattern: "
            f"expected {expected_delay:.6f}, got {delay:.6f}"
        )
    
    @given(
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        max_delay=st.floats(min_value=1.0, max_value=100.0),
        retry_count=st.integers(min_value=0, max_value=20)
    )
    @settings(max_examples=100)
    def test_backoff_respects_max_delay(self, base_delay, max_delay, retry_count):
        """
        Property: Delay is capped at max_delay.
        
        For any base delay, max delay, and retry count, the calculated
        backoff delay should never exceed max_delay.
        """
        # Ensure max_delay >= base_delay for valid test
        assume(max_delay >= base_delay)
        
        handler = ErrorHandler(base_delay=base_delay, max_delay=max_delay)
        
        # Calculate delay
        delay = handler.calculate_backoff_delay(retry_count)
        
        # Should not exceed max_delay
        assert delay <= max_delay, (
            f"Delay should be capped at max_delay: "
            f"expected <= {max_delay:.6f}, got {delay:.6f}"
        )
    
    @given(
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        retry_count=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=100)
    def test_backoff_is_non_negative(self, base_delay, retry_count):
        """
        Property: Delay is always non-negative.
        
        For any base delay and retry count, the calculated backoff delay
        should always be non-negative.
        """
        handler = ErrorHandler(base_delay=base_delay)
        
        # Calculate delay
        delay = handler.calculate_backoff_delay(retry_count)
        
        # Should be non-negative
        assert delay >= 0.0, f"Delay should be non-negative, got {delay:.6f}"
    
    @given(
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        retry_count_1=st.integers(min_value=0, max_value=10),
        retry_count_2=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=100)
    def test_backoff_increases_monotonically(
        self,
        base_delay,
        retry_count_1,
        retry_count_2
    ):
        """
        Property: Delay increases monotonically with retry count.
        
        For any base delay and two retry counts where count_1 < count_2,
        the delay for count_2 should be >= delay for count_1.
        """
        # Ensure retry_count_1 < retry_count_2
        assume(retry_count_1 < retry_count_2)
        
        handler = ErrorHandler(base_delay=base_delay, max_delay=float('inf'))
        
        # Calculate delays
        delay_1 = handler.calculate_backoff_delay(retry_count_1)
        delay_2 = handler.calculate_backoff_delay(retry_count_2)
        
        # delay_2 should be >= delay_1
        assert delay_2 >= delay_1, (
            f"Delay should increase monotonically: "
            f"delay({retry_count_1}) = {delay_1:.6f}, "
            f"delay({retry_count_2}) = {delay_2:.6f}"
        )
    
    @given(
        base_delay=st.floats(min_value=0.1, max_value=10.0)
    )
    @settings(max_examples=100)
    def test_backoff_first_retry_equals_base_delay(self, base_delay):
        """
        Property: First retry (n=0) has delay equal to base_delay.
        
        For any base delay, the delay for the first retry attempt (n=0)
        should equal the base delay (since 2^0 = 1).
        """
        handler = ErrorHandler(base_delay=base_delay, max_delay=float('inf'))
        
        # Calculate delay for first retry (n=0)
        delay = handler.calculate_backoff_delay(0)
        
        # Should equal base_delay
        assert abs(delay - base_delay) < 1e-6, (
            f"First retry delay should equal base_delay: "
            f"expected {base_delay:.6f}, got {delay:.6f}"
        )
    
    @given(
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        retry_count=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_backoff_doubles_each_retry(self, base_delay, retry_count):
        """
        Property: Delay doubles with each retry attempt.
        
        For any base delay and retry count n > 0, the delay for retry n
        should be approximately double the delay for retry n-1.
        """
        handler = ErrorHandler(base_delay=base_delay, max_delay=float('inf'))
        
        # Calculate delays for consecutive retries
        delay_n_minus_1 = handler.calculate_backoff_delay(retry_count - 1)
        delay_n = handler.calculate_backoff_delay(retry_count)
        
        # delay_n should be approximately 2 * delay_n_minus_1
        expected_ratio = 2.0
        actual_ratio = delay_n / delay_n_minus_1 if delay_n_minus_1 > 0 else 0
        
        assert abs(actual_ratio - expected_ratio) < 1e-6, (
            f"Delay should double each retry: "
            f"delay({retry_count - 1}) = {delay_n_minus_1:.6f}, "
            f"delay({retry_count}) = {delay_n:.6f}, "
            f"ratio = {actual_ratio:.6f} (expected {expected_ratio:.6f})"
        )
    
    @given(
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        max_delay=st.floats(min_value=1.0, max_value=100.0)
    )
    @settings(max_examples=100)
    def test_backoff_eventually_reaches_max_delay(self, base_delay, max_delay):
        """
        Property: Delay eventually reaches and stays at max_delay.
        
        For any base delay and max delay, there exists a retry count N
        such that for all n >= N, delay(n) = max_delay.
        """
        # Ensure max_delay >= base_delay for valid test
        assume(max_delay >= base_delay)
        
        handler = ErrorHandler(base_delay=base_delay, max_delay=max_delay)
        
        # Find the retry count where delay reaches max_delay
        # Exponential: base_delay * 2^n >= max_delay
        # Solve: n >= log2(max_delay / base_delay)
        import math
        n_threshold = math.ceil(math.log2(max_delay / base_delay))
        
        # Test that delay reaches max_delay at or before threshold
        delay_at_threshold = handler.calculate_backoff_delay(n_threshold)
        assert abs(delay_at_threshold - max_delay) < 1e-6, (
            f"Delay should reach max_delay at retry {n_threshold}: "
            f"expected {max_delay:.6f}, got {delay_at_threshold:.6f}"
        )
        
        # Test that delay stays at max_delay for higher retry counts
        delay_after_threshold = handler.calculate_backoff_delay(n_threshold + 5)
        assert abs(delay_after_threshold - max_delay) < 1e-6, (
            f"Delay should stay at max_delay after threshold: "
            f"expected {max_delay:.6f}, got {delay_after_threshold:.6f}"
        )
    
    @given(
        base_delay=st.floats(min_value=0.01, max_value=0.1),  # Smaller delays for faster tests
        max_retries=st.integers(min_value=1, max_value=3)  # Fewer retries for faster tests
    )
    @settings(max_examples=50, deadline=None)  # Disable deadline since we actually sleep
    def test_retry_with_backoff_respects_max_retries(self, base_delay, max_retries):
        """
        Property: Retry logic respects max_retries limit.
        
        For any base delay and max retries, the retry_with_backoff method
        should attempt at most (max_retries + 1) executions.
        """
        handler = ErrorHandler(base_delay=base_delay, max_retries=max_retries)
        
        # Create a context
        context = ErrorContext(
            framework_name="TestFramework",
            error=Exception("Test error"),
            error_category=ErrorCategory.RUNTIME
        )
        
        # Create an operation that always fails
        attempt_count = [0]
        
        def failing_operation():
            attempt_count[0] += 1
            raise Exception("Always fails")
        
        # Retry the operation
        result = handler.retry_with_backoff(failing_operation, context)
        
        # Should have attempted max_retries + 1 times
        assert result.attempts == max_retries + 1, (
            f"Should attempt {max_retries + 1} times, "
            f"but attempted {result.attempts} times"
        )
        assert attempt_count[0] == max_retries + 1, (
            f"Operation should be called {max_retries + 1} times, "
            f"but was called {attempt_count[0]} times"
        )
        assert not result.success, "Result should indicate failure"
        assert result.final_error is not None, "Should have final error"
    
    @given(
        base_delay=st.floats(min_value=0.01, max_value=0.1),  # Smaller delays for faster tests
        max_retries=st.integers(min_value=1, max_value=3),  # Fewer retries for faster tests
        success_on_attempt=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=50, deadline=None)  # Disable deadline since we actually sleep
    def test_retry_with_backoff_succeeds_on_nth_attempt(
        self,
        base_delay,
        max_retries,
        success_on_attempt
    ):
        """
        Property: Retry succeeds when operation succeeds on nth attempt.
        
        For any base delay, max retries, and success attempt n <= max_retries,
        if the operation succeeds on attempt n, retry_with_backoff should
        return success with attempts = n.
        """
        # Ensure success_on_attempt is within max_retries
        assume(success_on_attempt <= max_retries + 1)
        
        handler = ErrorHandler(base_delay=base_delay, max_retries=max_retries)
        
        # Create a context
        context = ErrorContext(
            framework_name="TestFramework",
            error=Exception("Test error"),
            error_category=ErrorCategory.RUNTIME
        )
        
        # Create an operation that succeeds on nth attempt
        attempt_count = [0]
        
        def eventually_succeeds():
            attempt_count[0] += 1
            if attempt_count[0] < success_on_attempt:
                raise Exception(f"Fail on attempt {attempt_count[0]}")
            return "Success"
        
        # Retry the operation
        result = handler.retry_with_backoff(eventually_succeeds, context)
        
        # Should succeed on the specified attempt
        assert result.success, f"Should succeed on attempt {success_on_attempt}"
        assert result.attempts == success_on_attempt, (
            f"Should report {success_on_attempt} attempts, "
            f"but reported {result.attempts}"
        )
        assert result.final_error is None, "Should have no final error on success"
    
    @given(
        base_delay=st.floats(min_value=0.01, max_value=0.1),  # Smaller delays for faster tests
        max_retries=st.integers(min_value=1, max_value=3)  # Fewer retries for faster tests
    )
    @settings(max_examples=50, deadline=None)  # Disable deadline since we actually sleep
    def test_retry_with_backoff_accumulates_delay(self, base_delay, max_retries):
        """
        Property: Total delay accumulates across retry attempts.
        
        For any base delay and max retries, if all attempts fail,
        the total delay should be approximately the sum of exponential delays.
        """
        handler = ErrorHandler(base_delay=base_delay, max_retries=max_retries)
        
        # Create a context
        context = ErrorContext(
            framework_name="TestFramework",
            error=Exception("Test error"),
            error_category=ErrorCategory.RUNTIME
        )
        
        # Create an operation that always fails
        def failing_operation():
            raise Exception("Always fails")
        
        # Calculate expected total delay
        expected_total_delay = sum(
            handler.calculate_backoff_delay(i)
            for i in range(max_retries)
        )
        
        # Retry the operation
        result = handler.retry_with_backoff(failing_operation, context)
        
        # Total delay should be approximately the sum of exponential delays
        # Allow some tolerance for timing variations
        assert abs(result.total_delay_seconds - expected_total_delay) < 0.5, (
            f"Total delay should be approximately {expected_total_delay:.3f}s, "
            f"but was {result.total_delay_seconds:.3f}s"
        )
