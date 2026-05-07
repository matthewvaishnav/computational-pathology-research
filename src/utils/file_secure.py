"""
Secure File Operations

Provides secure file writing with proper permissions.
"""

import os
from pathlib import Path
from typing import Union


def write_file_secure(filepath: Union[str, Path], content: Union[str, bytes], mode: int = 0o600):
    """Write file with secure permissions.
    
    Args:
        filepath: Path to file
        content: Content to write
        mode: File permissions (default: 0o600 - owner read/write only)
    """
    filepath = Path(filepath)
    
    # Create parent directories if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write file
    if isinstance(content, bytes):
        filepath.write_bytes(content)
    else:
        filepath.write_text(content)
    
    # Set secure permissions
    os.chmod(filepath, mode)


def write_config_file(filepath: Union[str, Path], content: str):
    """Write configuration file with restricted permissions.
    
    Args:
        filepath: Path to config file
        content: Configuration content
    """
    write_file_secure(filepath, content, mode=0o600)


def write_key_file(filepath: Union[str, Path], content: bytes):
    """Write cryptographic key file with restricted permissions.
    
    Args:
        filepath: Path to key file
        content: Key content
    """
    write_file_secure(filepath, content, mode=0o400)  # Read-only


def write_log_file(filepath: Union[str, Path], content: str):
    """Write log file with appropriate permissions.
    
    Args:
        filepath: Path to log file
        content: Log content
    """
    write_file_secure(filepath, content, mode=0o640)  # Owner rw, group r
