"""
Error handling and recovery logic for the Competitor Benchmark System.

This module provides error classification, retry logic with exponential backoff,
and recovery strategies for various failure modes during benchmark execution.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors that can occur during benchmarking."""
    
    INSTALLATION = "installation"  # Framework installation failures
    CONFIGURATION = "configuration"  # Invalid task specifications
    RUNTIME = "runtime"  # Training execution errors
    TIMEOUT = "timeout"  # Exceeded time limits
    RESOURCE = "resource"  # GPU/memory issues
    DATA = "data"  # Data loading or corruption issues


class RecoveryAction(Enum):
    """Actions to take when an error occurs."""
    
    RETRY = "retry"  # Retry the operation
    RETRY_WITH_PATCH = "retry_with_patch"  # Apply patch and retry
    SKIP_FRAMEWORK = "skip_framework"  # Skip this framework, continue with others
    HALT_BENCHMARK = "halt_benchmark"  # Stop entire benchmark
    LOG_AND_CONTINUE = "log_and_continue"  # Log error, continue with next framework
    SAVE_PARTIAL_AND_CONTINUE = "save_partial_and_continue"  # Save partial results


@dataclass
class ErrorContext:
    """Context information for error handling decisions."""
    
    framework_name: str
    error: Exception
    error_category: ErrorCategory
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetryResult:
    """Result of a retry operation."""
    
    success: bool
    attempts: int
    total_delay_seconds: float
    final_error: Optional[Exception] = None


class ErrorHandler:
    """
    Handles errors during benchmark execution with classification and recovery.
    
    Implements:
    - Error classification (recoverable vs fatal)
    - Retry logic with exponential backoff
    - Recovery action determination
    - Error logging and reporting
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.7
    """
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_retries: int = 3,
        max_delay: float = 60.0
    ):
        """
        Initialize error handler.
        
        Args:
            base_delay: Base delay in seconds for exponential backoff
            max_retries: Maximum number of retry attempts
            max_delay: Maximum delay between retries (cap for exponential growth)
        """
        self.base_delay = base_delay
        self.max_retries = max_retries
        self.max_delay = max_delay
        self.error_log: list[ErrorContext] = []
    
    def classify_error(self, error: Exception) -> ErrorCategory:
        """
        Classify an error into a category for recovery decision.
        
        Args:
            error: The exception that occurred
            
        Returns:
            ErrorCategory indicating the type of error
            
        Requirement: 8.3 (Error classification)
        """
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Installation errors
        if any(keyword in error_msg for keyword in [
            "dependency", "install", "import", "module not found", "no module named"
        ]):
            return ErrorCategory.INSTALLATION
        
        # Configuration errors
        if any(keyword in error_msg for keyword in [
            "invalid", "configuration", "parameter", "argument"
        ]) or error_type in ["ValueError", "TypeError", "KeyError"]:
            return ErrorCategory.CONFIGURATION
        
        # Timeout errors
        if "timeout" in error_msg or error_type == "TimeoutError":
            return ErrorCategory.TIMEOUT
        
        # Resource errors (GPU/memory)
        if any(keyword in error_msg for keyword in [
            "out of memory", "cuda", "gpu", "memory", "oom"
        ]):
            return ErrorCategory.RESOURCE
        
        # Data errors
        if any(keyword in error_msg for keyword in [
            "data", "corrupt", "file not found", "io error"
        ]):
            return ErrorCategory.DATA
        
        # Default to runtime error
        return ErrorCategory.RUNTIME
    
    def is_recoverable(self, error_category: ErrorCategory) -> bool:
        """
        Determine if an error category is recoverable.
        
        Args:
            error_category: The category of error
            
        Returns:
            True if error is recoverable, False if fatal
            
        Requirement: 8.3 (Error classification - recoverable vs fatal)
        """
        # Recoverable errors
        recoverable = {
            ErrorCategory.RESOURCE,  # GPU might become available
            ErrorCategory.TIMEOUT,  # Can retry with different timeout
            ErrorCategory.RUNTIME,  # Transient runtime issues
        }
        
        # Fatal errors
        fatal = {
            ErrorCategory.CONFIGURATION,  # Invalid config won't fix itself
            ErrorCategory.INSTALLATION,  # Installation issues need manual fix
            ErrorCategory.DATA,  # Data corruption is persistent
        }
        
        return error_category in recoverable
    
    def handle_error(
        self,
        error: Exception,
        context: ErrorContext
    ) -> RecoveryAction:
        """
        Determine recovery action based on error type and context.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            
        Returns:
            RecoveryAction indicating what to do next
            
        Requirements: 8.1 (Error isolation), 8.3 (Error classification),
                     8.4 (Fatal error handling)
        """
        # Classify the error
        error_category = self.classify_error(error)
        context.error_category = error_category
        
        # Log the error
        self.error_log.append(context)
        logger.error(
            f"Error in framework {context.framework_name}: "
            f"{error_category.value} - {str(error)}"
        )
        
        # Determine recovery action based on category
        if error_category == ErrorCategory.INSTALLATION:
            if context.retry_count < context.max_retries:
                logger.info(
                    f"Installation error, attempting retry with patch "
                    f"(attempt {context.retry_count + 1}/{context.max_retries})"
                )
                return RecoveryAction.RETRY_WITH_PATCH
            else:
                logger.warning(
                    f"Installation failed after {context.max_retries} attempts, "
                    f"skipping framework {context.framework_name}"
                )
                return RecoveryAction.SKIP_FRAMEWORK
        
        elif error_category == ErrorCategory.CONFIGURATION:
            logger.error(
                f"Configuration error in {context.framework_name}, "
                f"halting benchmark"
            )
            return RecoveryAction.HALT_BENCHMARK
        
        elif error_category == ErrorCategory.TIMEOUT:
            logger.warning(
                f"Timeout in {context.framework_name}, "
                f"saving partial results and continuing"
            )
            return RecoveryAction.SAVE_PARTIAL_AND_CONTINUE
        
        elif error_category == ErrorCategory.RESOURCE:
            if context.retry_count < context.max_retries:
                logger.info(
                    f"Resource error, will retry after backoff "
                    f"(attempt {context.retry_count + 1}/{context.max_retries})"
                )
                return RecoveryAction.RETRY
            else:
                logger.warning(
                    f"Resource error persists after {context.max_retries} attempts, "
                    f"continuing with next framework"
                )
                return RecoveryAction.LOG_AND_CONTINUE
        
        elif error_category == ErrorCategory.DATA:
            logger.error(
                f"Data error in {context.framework_name}, "
                f"continuing with next framework"
            )
            return RecoveryAction.LOG_AND_CONTINUE
        
        else:  # RUNTIME
            if context.retry_count < context.max_retries:
                logger.info(
                    f"Runtime error, will retry "
                    f"(attempt {context.retry_count + 1}/{context.max_retries})"
                )
                return RecoveryAction.RETRY
            else:
                logger.warning(
                    f"Runtime error persists, continuing with next framework"
                )
                return RecoveryAction.LOG_AND_CONTINUE
    
    def calculate_backoff_delay(self, retry_count: int) -> float:
        """
        Calculate exponential backoff delay for retry attempt.
        
        Args:
            retry_count: Current retry attempt number (0-indexed)
            
        Returns:
            Delay in seconds before next retry
            
        Requirement: 8.2 (Exponential backoff: delay(n) = base_delay * 2^n)
        """
        # Exponential backoff: delay(n) = base_delay * 2^n
        delay = self.base_delay * (2 ** retry_count)
        
        # Cap at max_delay to prevent excessive waits
        return min(delay, self.max_delay)
    
    def retry_with_backoff(
        self,
        operation,
        context: ErrorContext,
        *args,
        **kwargs
    ) -> RetryResult:
        """
        Retry an operation with exponential backoff.
        
        Args:
            operation: Callable to retry
            context: Error context for logging
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            RetryResult with success status and metadata
            
        Requirement: 8.2 (Retry logic with exponential backoff)
        """
        total_delay = 0.0
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = operation(*args, **kwargs)
                logger.info(
                    f"Operation succeeded on attempt {attempt + 1} "
                    f"for {context.framework_name}"
                )
                return RetryResult(
                    success=True,
                    attempts=attempt + 1,
                    total_delay_seconds=total_delay
                )
            
            except Exception as e:
                last_error = e
                context.retry_count = attempt
                
                if attempt < self.max_retries:
                    delay = self.calculate_backoff_delay(attempt)
                    total_delay += delay
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {context.framework_name}: "
                        f"{str(e)}. Retrying in {delay:.1f}s..."
                    )
                    
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed "
                        f"for {context.framework_name}"
                    )
        
        return RetryResult(
            success=False,
            attempts=self.max_retries + 1,
            total_delay_seconds=total_delay,
            final_error=last_error
        )
    
    def enforce_timeout(
        self,
        operation,
        timeout_seconds: float,
        context: ErrorContext,
        *args,
        **kwargs
    ):
        """
        Enforce timeout for hanging tasks.
        
        Args:
            operation: Callable to execute with timeout
            timeout_seconds: Maximum execution time
            context: Error context for logging
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of operation if successful
            
        Raises:
            TimeoutError: If operation exceeds timeout
            
        Requirement: 8.4 (Timeout enforcement for hanging tasks)
        """
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(
                f"Operation timed out after {timeout_seconds}s "
                f"for {context.framework_name}"
            )
        
        # Set up timeout (Unix-like systems)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = operation(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            except TimeoutError:
                signal.alarm(0)  # Cancel alarm
                logger.error(
                    f"Timeout enforced for {context.framework_name} "
                    f"after {timeout_seconds}s"
                )
                raise
        else:
            # Fallback for Windows (no SIGALRM)
            # Use threading.Timer or just execute without timeout
            logger.warning(
                "Timeout enforcement not available on this platform, "
                "executing without timeout"
            )
            return operation(*args, **kwargs)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Generate summary of all errors encountered.
        
        Returns:
            Dictionary with error statistics and details
            
        Requirement: 8.7 (Error logging and reporting)
        """
        if not self.error_log:
            return {
                "total_errors": 0,
                "by_category": {},
                "by_framework": {},
                "errors": []
            }
        
        # Count by category
        by_category = {}
        for ctx in self.error_log:
            category = ctx.error_category.value
            by_category[category] = by_category.get(category, 0) + 1
        
        # Count by framework
        by_framework = {}
        for ctx in self.error_log:
            framework = ctx.framework_name
            by_framework[framework] = by_framework.get(framework, 0) + 1
        
        # Detailed error list
        errors = [
            {
                "framework": ctx.framework_name,
                "category": ctx.error_category.value,
                "error": str(ctx.error),
                "retry_count": ctx.retry_count,
                "metadata": ctx.metadata
            }
            for ctx in self.error_log
        ]
        
        return {
            "total_errors": len(self.error_log),
            "by_category": by_category,
            "by_framework": by_framework,
            "errors": errors
        }
