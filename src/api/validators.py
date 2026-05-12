"""
API Input Validators

Centralized validation functions for API request parameters.
"""

import re
from typing import Optional

from fastapi import HTTPException


def validate_limit(limit: int, min_val: int = 1, max_val: int = 1000) -> None:
    """Validate pagination limit parameter.
    
    Args:
        limit: The limit value to validate
        min_val: Minimum allowed value (default: 1)
        max_val: Maximum allowed value (default: 1000)
        
    Raises:
        HTTPException: If limit is out of bounds
    """
    if limit < min_val or limit > max_val:
        raise HTTPException(
            status_code=400,
            detail=f"Limit must be between {min_val} and {max_val}"
        )


def validate_email(email: str) -> None:
    """Validate email format.
    
    Args:
        email: Email address to validate
        
    Raises:
        HTTPException: If email format is invalid
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise HTTPException(status_code=400, detail="Invalid email format")


def validate_password(password: str, min_length: int = 8) -> None:
    """Validate password strength.
    
    Args:
        password: Password to validate
        min_length: Minimum password length (default: 8)
        
    Raises:
        HTTPException: If password doesn't meet requirements
    """
    if len(password) < min_length:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {min_length} characters"
        )
    
    # Check for at least one uppercase, one lowercase, one digit
    if not re.search(r'[A-Z]', password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least one uppercase letter"
        )
    if not re.search(r'[a-z]', password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least one lowercase letter"
        )
    if not re.search(r'\d', password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least one digit"
        )


def validate_file_upload(
    file_content: bytes,
    filename: str,
    allowed_extensions: Optional[set] = None,
    max_size: int = 100 * 1024 * 1024  # 100MB
) -> tuple[str, str]:
    """Validate uploaded file.
    
    Args:
        file_content: File content bytes
        filename: Original filename
        allowed_extensions: Set of allowed file extensions
        max_size: Maximum file size in bytes
        
    Returns:
        Tuple of (detected_mime_type, safe_filename)
        
    Raises:
        HTTPException: If file validation fails
    """
    import magic
    from pathlib import Path
    
    # Check file size
    if len(file_content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
        )
    
    # Detect MIME type
    try:
        detected_mime = magic.from_buffer(file_content, mime=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not detect file type")
    
    # Validate extension
    if allowed_extensions:
        ext = Path(filename).suffix.lower().lstrip('.')
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"
            )
    
    # Sanitize filename
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    return detected_mime, safe_filename
