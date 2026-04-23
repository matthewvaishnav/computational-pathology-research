"""
Exception classes for WSI processing pipeline.

This module defines custom exceptions for different error categories
in the WSI processing pipeline.
"""


class WSIProcessingError(Exception):
    """Base exception for WSI processing errors."""

    pass


class FileFormatError(WSIProcessingError):
    """Raised when file format is unsupported or corrupted.
    
    Examples:
        - Unsupported file extension
        - Corrupted WSI file
        - Missing required metadata
    """

    pass


class ResourceError(WSIProcessingError):
    """Raised when system resources are insufficient.
    
    Examples:
        - Insufficient disk space
        - GPU memory exhaustion
        - CPU memory exhaustion
    """

    pass


class ProcessingError(WSIProcessingError):
    """Raised when processing step fails.
    
    Examples:
        - OpenSlide read failure
        - Feature extraction failure
        - HDF5 write failure
    """

    pass
