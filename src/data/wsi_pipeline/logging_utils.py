"""
Logging utilities for WSI processing pipeline.

This module provides standardized logging configuration and utilities
for the WSI processing pipeline components.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class WSIPipelineFormatter(logging.Formatter):
    """
    Custom formatter for WSI pipeline logging.

    Provides colored output and structured formatting for better readability.
    """

    # Color codes for different log levels
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(self, use_colors: bool = True):
        """
        Initialize formatter.

        Args:
            use_colors: Whether to use colored output
        """
        self.use_colors = use_colors and sys.stderr.isatty()

        # Format string with component name
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record):
        """Format log record with colors if enabled."""
        # Get the formatted message
        formatted = super().format(record)

        if self.use_colors:
            level_color = self.COLORS.get(record.levelname, "")
            reset_color = self.COLORS["RESET"]

            # Color the level name
            formatted = formatted.replace(
                f" - {record.levelname} - ", f" - {level_color}{record.levelname}{reset_color} - "
            )

        return formatted


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Path] = None,
    use_colors: bool = True,
    component_filter: Optional[str] = None,
) -> None:
    """
    Setup standardized logging for WSI pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        use_colors: Whether to use colored console output
        component_filter: Optional component name filter (e.g., 'wsi_pipeline')

    Example:
        >>> setup_logging(level='DEBUG', log_file='wsi_processing.log')
        >>> logger = logging.getLogger('data.wsi_pipeline.reader')
        >>> logger.info("Processing slide.svs")
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(WSIPipelineFormatter(use_colors=use_colors))
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(WSIPipelineFormatter(use_colors=False))
        root_logger.addHandler(file_handler)

    # Set component-specific logging levels
    if component_filter:
        # Reduce noise from other libraries
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("h5py").setLevel(logging.WARNING)

        # Keep WSI pipeline components at specified level
        logging.getLogger(component_filter).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with standardized naming.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggingContext:
    """
    Context manager for temporary logging configuration.

    Allows temporary changes to logging level or output for specific operations.
    """

    def __init__(
        self,
        level: Optional[Union[str, int]] = None,
        logger_name: Optional[str] = None,
        suppress_output: bool = False,
    ):
        """
        Initialize logging context.

        Args:
            level: Temporary logging level
            logger_name: Specific logger to modify (None for root)
            suppress_output: Whether to suppress all output
        """
        self.level = level
        self.logger_name = logger_name
        self.suppress_output = suppress_output
        self.original_level = None
        self.original_handlers = None

    def __enter__(self):
        """Enter context and modify logging."""
        # Get target logger
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
        else:
            logger = logging.getLogger()

        # Store original configuration
        self.original_level = logger.level
        self.original_handlers = logger.handlers.copy()

        # Apply temporary configuration
        if self.level is not None:
            if isinstance(self.level, str):
                level = getattr(logging, self.level.upper())
            else:
                level = self.level
            logger.setLevel(level)

        if self.suppress_output:
            logger.handlers.clear()

        return logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore logging."""
        # Get target logger
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
        else:
            logger = logging.getLogger()

        # Restore original configuration
        logger.setLevel(self.original_level)
        logger.handlers = self.original_handlers


def log_processing_stats(
    logger: logging.Logger,
    slide_id: str,
    num_patches: int,
    processing_time: float,
    file_size_mb: Optional[float] = None,
) -> None:
    """
    Log standardized processing statistics.

    Args:
        logger: Logger instance
        slide_id: Slide identifier
        num_patches: Number of patches processed
        processing_time: Processing time in seconds
        file_size_mb: Optional output file size in MB
    """
    patches_per_sec = num_patches / processing_time if processing_time > 0 else 0

    stats_msg = (
        f"Processed {slide_id}: {num_patches} patches in {processing_time:.2f}s "
        f"({patches_per_sec:.1f} patches/sec)"
    )

    if file_size_mb is not None:
        stats_msg += f", output: {file_size_mb:.2f}MB"

    logger.info(stats_msg)


def log_memory_usage(logger: logging.Logger, component: str) -> None:
    """
    Log current memory usage for a component.

    Args:
        logger: Logger instance
        component: Component name
    """
    try:
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.debug(f"{component} memory usage: {memory_mb:.1f}MB")
    except ImportError:
        logger.debug(f"{component} memory monitoring unavailable (psutil not installed)")
    except Exception as e:
        logger.debug(f"{component} memory monitoring failed: {e}")


def create_processing_log_file(
    base_dir: Union[str, Path],
    slide_id: str,
    timestamp: Optional[datetime] = None,
) -> Path:
    """
    Create a log file path for processing a specific slide.

    Args:
        base_dir: Base directory for log files
        slide_id: Slide identifier
        timestamp: Optional timestamp (uses current time if None)

    Returns:
        Path to log file
    """
    if timestamp is None:
        timestamp = datetime.now()

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    log_filename = f"{slide_id}_{timestamp_str}.log"

    return base_dir / log_filename


# Example usage and testing
if __name__ == "__main__":
    # Test logging setup
    setup_logging(level="DEBUG", use_colors=True)

    logger = get_logger(__name__)

    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test logging context
    with LoggingContext(level="ERROR"):
        logger.info("This won't be shown")
        logger.error("This will be shown")

    logger.info("Back to normal logging")

    # Test processing stats
    log_processing_stats(logger, "test_slide", 1000, 45.2, 15.6)

    # Test memory logging
    log_memory_usage(logger, "TestComponent")
