"""
Secure Temporary File Utilities

Provides secure temporary file and directory creation to prevent race conditions.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional


def create_secure_temp_file(
    suffix: str = "", prefix: str = "histocore_", dir: Optional[str] = None
) -> Path:
    """Create secure temporary file.

    Uses mkstemp() which creates file with 0600 permissions (owner read/write only).

    Args:
        suffix: File suffix
        prefix: File prefix
        dir: Directory for temp file (default: system temp)

    Returns:
        Path to secure temporary file
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
    os.close(fd)  # Close file descriptor, caller will open as needed
    return Path(path)


def create_secure_temp_dir(
    suffix: str = "", prefix: str = "histocore_", dir: Optional[str] = None
) -> Path:
    """Create secure temporary directory.

    Uses mkdtemp() which creates directory with 0700 permissions (owner only).

    Args:
        suffix: Directory suffix
        prefix: Directory prefix
        dir: Parent directory (default: system temp)

    Returns:
        Path to secure temporary directory
    """
    path = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    return Path(path)


def get_secure_temp_path(suffix: str = "", prefix: str = "histocore_") -> Path:
    """Get secure temporary file path without creating file.

    WARNING: This has a race condition window. Prefer create_secure_temp_file().
    Only use when you need the path before creating the file.

    Args:
        suffix: File suffix
        prefix: File prefix

    Returns:
        Path to temporary file (not yet created)
    """
    # Use NamedTemporaryFile with delete=True to get a unique name
    with tempfile.NamedTemporaryFile(suffix=suffix, prefix=prefix, delete=True) as tmp:
        return Path(tmp.name)
