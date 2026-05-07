"""
JSON Schema Validation

Validates JSON data against schemas to prevent injection and malformed data.
"""

import json
from typing import Any, Dict


def validate_json_size(data: str, max_size_mb: float = 10.0) -> None:
    """Validate JSON size to prevent memory exhaustion.
    
    Args:
        data: JSON string
        max_size_mb: Maximum size in MB
        
    Raises:
        ValueError: If JSON too large
    """
    size_mb = len(data.encode('utf-8')) / (1024 * 1024)
    
    if size_mb > max_size_mb:
        raise ValueError(f"JSON too large: {size_mb:.2f}MB (max: {max_size_mb}MB)")


def validate_json_depth(obj: Any, max_depth: int = 20, current_depth: int = 0) -> None:
    """Validate JSON nesting depth to prevent stack overflow.
    
    Args:
        obj: JSON object
        max_depth: Maximum nesting depth
        current_depth: Current depth (internal)
        
    Raises:
        ValueError: If nesting too deep
    """
    if current_depth > max_depth:
        raise ValueError(f"JSON nesting too deep: {current_depth} (max: {max_depth})")
    
    if isinstance(obj, dict):
        for value in obj.values():
            validate_json_depth(value, max_depth, current_depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            validate_json_depth(item, max_depth, current_depth + 1)


def safe_json_loads(data: str, max_size_mb: float = 10.0, max_depth: int = 20) -> Any:
    """Safely load JSON with size and depth validation.
    
    Args:
        data: JSON string
        max_size_mb: Maximum size in MB
        max_depth: Maximum nesting depth
        
    Returns:
        Parsed JSON object
        
    Raises:
        ValueError: If JSON invalid, too large, or too deep
        json.JSONDecodeError: If JSON malformed
    """
    # Validate size
    validate_json_size(data, max_size_mb)
    
    # Parse JSON
    obj = json.loads(data)
    
    # Validate depth
    validate_json_depth(obj, max_depth)
    
    return obj


def validate_json_keys(obj: Dict, allowed_keys: set) -> None:
    """Validate JSON only contains allowed keys.
    
    Args:
        obj: JSON object
        allowed_keys: Set of allowed key names
        
    Raises:
        ValueError: If unexpected keys found
    """
    if not isinstance(obj, dict):
        return
    
    unexpected = set(obj.keys()) - allowed_keys
    
    if unexpected:
        raise ValueError(f"Unexpected JSON keys: {unexpected}")
