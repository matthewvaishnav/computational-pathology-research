"""
Environment Variable Validation

Validates and sanitizes environment variables for security.
"""

import os
import re
from typing import Optional


def get_env_secure(
    key: str,
    default: Optional[str] = None,
    required: bool = False,
    pattern: Optional[str] = None
) -> Optional[str]:
    """Get environment variable with validation.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        required: Raise error if not set
        pattern: Regex pattern to validate value
        
    Returns:
        Environment variable value
        
    Raises:
        ValueError: If required and not set, or doesn't match pattern
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable not set: {key}")
    
    if value and pattern:
        if not re.match(pattern, value):
            raise ValueError(f"Environment variable {key} doesn't match pattern: {pattern}")
    
    return value


def get_env_int(key: str, default: Optional[int] = None, min_val: Optional[int] = None, max_val: Optional[int] = None) -> Optional[int]:
    """Get integer environment variable with validation.
    
    Args:
        key: Environment variable name
        default: Default value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Integer value
        
    Raises:
        ValueError: If not a valid integer or out of range
    """
    value = os.getenv(key)
    
    if value is None:
        return default
    
    try:
        int_val = int(value)
    except ValueError:
        raise ValueError(f"Environment variable {key} must be an integer, got: {value}")
    
    if min_val is not None and int_val < min_val:
        raise ValueError(f"Environment variable {key} must be >= {min_val}, got: {int_val}")
    
    if max_val is not None and int_val > max_val:
        raise ValueError(f"Environment variable {key} must be <= {max_val}, got: {int_val}")
    
    return int_val


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable.
    
    Accepts: true/false, yes/no, 1/0 (case insensitive)
    
    Args:
        key: Environment variable name
        default: Default value
        
    Returns:
        Boolean value
    """
    value = os.getenv(key)
    
    if value is None:
        return default
    
    value_lower = value.lower()
    
    if value_lower in ('true', 'yes', '1', 'on'):
        return True
    elif value_lower in ('false', 'no', '0', 'off'):
        return False
    else:
        raise ValueError(f"Environment variable {key} must be boolean, got: {value}")


def validate_env_path(key: str, must_exist: bool = False) -> Optional[str]:
    """Validate environment variable is a valid path.
    
    Args:
        key: Environment variable name
        must_exist: Whether path must exist
        
    Returns:
        Path string
        
    Raises:
        ValueError: If path invalid or doesn't exist when required
    """
    from pathlib import Path
    
    value = os.getenv(key)
    
    if value is None:
        return None
    
    # Check for path traversal attempts
    if '..' in value:
        raise ValueError(f"Path traversal detected in {key}: {value}")
    
    path = Path(value)
    
    if must_exist and not path.exists():
        raise ValueError(f"Path in {key} does not exist: {value}")
    
    return value
