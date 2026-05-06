"""
Common utility functions for repeated patterns across the codebase.

This module provides reusable utility functions to reduce code duplication
and standardize common operations like logging, error handling, and validation.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


def log_operation(operation_name: str, logger_instance: Optional[logging.Logger] = None):
    """
    Decorator to log operation start, completion, and errors.
    
    Args:
        operation_name: Name of the operation for logging
        logger_instance: Logger to use (defaults to module logger)
    
    Example:
        @log_operation("model training")
        def train_model():
            # training logic
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            log = logger_instance or logger
            start_time = time.time()
            
            log.info(f"Starting {operation_name}")
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                log.info(f"Completed {operation_name} in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                log.error(f"Failed {operation_name} after {elapsed:.2f}s: {e}")
                raise
        return wrapper
    return decorator


@contextmanager
def log_context(operation_name: str, logger_instance: Optional[logging.Logger] = None):
    """
    Context manager for logging operation lifecycle.
    
    Args:
        operation_name: Name of the operation
        logger_instance: Logger to use
    
    Example:
        with log_context("database transaction"):
            # database operations
            pass
    """
    log = logger_instance or logger
    start_time = time.time()
    
    log.info(f"Starting {operation_name}")
    try:
        yield
        elapsed = time.time() - start_time
        log.info(f"Completed {operation_name} in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"Failed {operation_name} after {elapsed:.2f}s: {e}")
        raise


def safe_execute(
    func: Callable[..., T], 
    *args, 
    default: Optional[T] = None,
    error_message: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None,
    **kwargs
) -> Optional[T]:
    """
    Safely execute a function with error handling and logging.
    
    Args:
        func: Function to execute
        *args: Positional arguments for function
        default: Default value to return on error
        error_message: Custom error message
        logger_instance: Logger to use
        **kwargs: Keyword arguments for function
    
    Returns:
        Function result or default value on error
    
    Example:
        result = safe_execute(risky_function, arg1, arg2, default=[], 
                            error_message="Failed to process data")
    """
    log = logger_instance or logger
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        message = error_message or f"Failed to execute {func.__name__}"
        log.error(f"{message}: {e}")
        return default


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        exceptions: Tuple of exceptions to catch and retry on
    
    Example:
        @retry_on_failure(max_retries=3, delay=1.0)
        def unreliable_network_call():
            # network operation
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {current_delay}s")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
            # This should never be reached, but just in case
            raise last_exception
        return wrapper
    return decorator


def validate_required_fields(data: Dict[str, Any], required_fields: list) -> Dict[str, Any]:
    """
    Validate that required fields are present in data dictionary.
    
    Args:
        data: Data dictionary to validate
        required_fields: List of required field names
    
    Returns:
        Validated data dictionary
    
    Raises:
        ValueError: If required fields are missing
    
    Example:
        validated_data = validate_required_fields(
            user_data, 
            ['username', 'email', 'password']
        )
    """
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    return data


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe filesystem usage.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed filename length
    
    Returns:
        Sanitized filename
    
    Example:
        safe_name = sanitize_filename("user input/file<name>.txt")
        # Returns: "user_input_file_name_.txt"
    """
    import re
    
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    
    # Trim whitespace and dots from ends
    sanitized = sanitized.strip(' .')
    
    # Ensure not empty
    if not sanitized:
        sanitized = 'unnamed_file'
    
    # Truncate if too long
    if len(sanitized) > max_length:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        max_name_length = max_length - len(ext) - 1 if ext else max_length
        sanitized = name[:max_name_length] + ('.' + ext if ext else '')
    
    return sanitized


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Number of bytes
    
    Returns:
        Formatted string (e.g., "1.5 GB")
    
    Example:
        size_str = format_bytes(1536000000)  # Returns "1.4 GB"
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string
    
    Example:
        duration_str = format_duration(3661)  # Returns "1h 1m 1s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    if hours < 24:
        return f"{hours}h {remaining_minutes}m"
    
    days = hours // 24
    remaining_hours = hours % 24
    
    return f"{days}d {remaining_hours}h"


def chunk_list(lst: list, chunk_size: int):
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
    
    Yields:
        Chunks of the original list
    
    Example:
        for chunk in chunk_list(range(10), 3):
            print(chunk)  # [0, 1, 2], [3, 4, 5], [6, 7, 8], [9]
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
    
    Returns:
        Merged dictionary
    
    Example:
        merged = deep_merge_dicts(
            {'a': {'b': 1, 'c': 2}}, 
            {'a': {'c': 3, 'd': 4}}
        )
        # Returns: {'a': {'b': 1, 'c': 3, 'd': 4}}
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


class ProgressTracker:
    """
    Simple progress tracking utility.
    
    Example:
        tracker = ProgressTracker(total=100, description="Processing")
        for i in range(100):
            # do work
            tracker.update(1)
    """
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, increment: int = 1):
        """Update progress by increment."""
        self.current = min(self.current + increment, self.total)
        self._log_progress()
    
    def _log_progress(self):
        """Log current progress."""
        if self.total == 0:
            return
            
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = format_duration(eta)
        else:
            eta_str = "unknown"
        
        logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - ETA: {eta_str}")


def memoize(func: Callable[..., T]) -> Callable[..., T]:
    """
    Simple memoization decorator for caching function results.
    
    Args:
        func: Function to memoize
    
    Returns:
        Memoized function
    
    Example:
        @memoize
        def expensive_computation(x):
            # expensive operation
            return result
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    return wrapper


def handle_errors(default_return: Any = None, log_level: str = "error", 
                 reraise: bool = False, error_message: Optional[str] = None):
    """
    Enhanced error handling decorator with configurable behavior.
    
    Args:
        default_return: Value to return on error (if not reraising)
        log_level: Logging level for errors ("debug", "info", "warning", "error", "critical")
        reraise: Whether to reraise the exception after logging
        error_message: Custom error message template
    
    Example:
        @handle_errors(default_return=[], log_level="warning", reraise=False)
        def get_data():
            # might fail
            return risky_operation()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error at specified level
                log_func = getattr(logger, log_level.lower(), logger.error)
                message = error_message or f"Error in {func.__name__}: {e}"
                log_func(message)
                
                # Reraise if requested
                if reraise:
                    raise
                
                # Return default value
                return default_return
        return wrapper
    return decorator


@contextmanager
def error_context(operation: str, cleanup_func: Optional[Callable] = None, 
                 reraise: bool = True):
    """
    Context manager for error handling with optional cleanup.
    
    Args:
        operation: Name of the operation for logging
        cleanup_func: Optional cleanup function to call on error
        reraise: Whether to reraise exceptions after cleanup
    
    Example:
        with error_context("file processing", cleanup_func=close_files):
            process_file()
    """
    try:
        yield
    except Exception as e:
        logger.error(f"Error during {operation}: {e}")
        
        # Run cleanup if provided
        if cleanup_func:
            try:
                cleanup_func()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup for {operation}: {cleanup_error}")
        
        # Reraise if requested
        if reraise:
            raise