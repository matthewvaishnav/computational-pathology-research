"""
Input Validators Module

Centralized validation functions for API endpoints.
Provides consistent validation logic across all routers.
"""

import re
from typing import Tuple
from fastapi import HTTPException

# Try to import magic, fall back to basic validation if not available
try:
    import magic
    MAGIC_AVAILABLE = True
except (ImportError, OSError):
    MAGIC_AVAILABLE = False


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email is valid, False otherwise
        
    Raises:
        HTTPException: If email format is invalid
    """
    if not email or not isinstance(email, str):
        raise HTTPException(status_code=400, detail="Email is required")
    
    # Practical email validation - allows most common valid formats
    # Prevents obvious issues like consecutive dots, missing @ etc.
    email_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._%+-]*[a-zA-Z0-9]@[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,}$'
    
    # Allow single character local parts
    if len(email.split('@')[0]) == 1:
        email_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._%+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        raise HTTPException(status_code=400, detail="Invalid email format")
    
    # Additional checks for consecutive dots
    if '..' in email:
        raise HTTPException(status_code=400, detail="Invalid email format")
    
    # Additional checks
    if len(email) > 254:  # RFC 5321 limit
        raise HTTPException(status_code=400, detail="Email address too long")
    
    local_part, domain = email.rsplit('@', 1)
    if len(local_part) > 64:  # RFC 5321 limit
        raise HTTPException(status_code=400, detail="Email local part too long")
    
    return True


def validate_password(password: str) -> bool:
    """
    Validate password strength.
    
    Requirements:
    - At least 8 characters long
    - Contains at least one uppercase letter
    - Contains at least one lowercase letter
    - Contains at least one digit
    - Contains at least one special character
    
    Args:
        password: Password to validate
        
    Returns:
        True if password meets requirements
        
    Raises:
        HTTPException: If password doesn't meet requirements
    """
    if not password or not isinstance(password, str):
        raise HTTPException(status_code=400, detail="Password is required")
    
    if len(password) < 8:
        raise HTTPException(
            status_code=400, 
            detail="Password must be at least 8 characters long"
        )
    
    if len(password) > 128:  # Prevent DoS via extremely long passwords
        raise HTTPException(
            status_code=400, 
            detail="Password must be less than 128 characters"
        )
    
    # Check for required character types
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    missing_requirements = []
    if not has_upper:
        missing_requirements.append("uppercase letter")
    if not has_lower:
        missing_requirements.append("lowercase letter")
    if not has_digit:
        missing_requirements.append("digit")
    if not has_special:
        missing_requirements.append("special character")
    
    if missing_requirements:
        raise HTTPException(
            status_code=400,
            detail=f"Password must contain at least one: {', '.join(missing_requirements)}"
        )
    
    return True


def validate_file_upload(file_content: bytes, filename: str) -> Tuple[str, str]:
    """
    Validate uploaded file content and determine file type.
    
    Performs security checks:
    - File size limits
    - Magic byte validation (prevents file type spoofing)
    - Allowed file type verification
    - Malicious content detection
    
    Args:
        file_content: Raw file content bytes
        filename: Original filename
        
    Returns:
        Tuple of (detected_mime_type, safe_filename)
        
    Raises:
        HTTPException: If file validation fails
    """
    if not file_content:
        raise HTTPException(status_code=400, detail="File content is required")
    
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # File size limits
    max_size = 100 * 1024 * 1024  # 100MB
    if len(file_content) > max_size:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
        )
    
    if len(file_content) < 10:  # Minimum viable file size
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupted")
    
    # Detect actual file type using magic bytes
    if MAGIC_AVAILABLE:
        try:
            detected_mime = magic.from_buffer(file_content, mime=True)
        except Exception:
            detected_mime = "application/octet-stream"
    else:
        # Fallback: detect based on file signatures
        detected_mime = _detect_mime_from_signature(file_content)
    
    # Allowed MIME types for medical imaging
    allowed_mimes = {
        'image/png',
        'image/jpeg', 
        'image/tiff',
        'image/bmp',
        'application/dicom',
        'application/octet-stream'  # For DICOM files that may not be detected
    }
    
    if detected_mime not in allowed_mimes:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{detected_mime}' not supported. "
                   f"Allowed types: {', '.join(sorted(allowed_mimes))}"
        )
    
    # Sanitize filename
    safe_filename = _sanitize_filename(filename)
    
    # Additional security checks for specific file types
    if detected_mime.startswith('image/'):
        _validate_image_content(file_content)
    elif detected_mime == 'application/dicom':
        _validate_dicom_content(file_content)
    elif detected_mime == 'application/octet-stream':
        # For octet-stream, do additional validation to ensure it's a valid file type
        _validate_unknown_content(file_content, filename)
    
    return detected_mime, safe_filename


def _detect_mime_from_signature(file_content: bytes) -> str:
    """
    Detect MIME type from file signature when python-magic is not available.
    
    Args:
        file_content: File content bytes
        
    Returns:
        Detected MIME type
    """
    if len(file_content) < 4:
        return "application/octet-stream"
    
    # Check common file signatures
    if file_content.startswith(b'\x89PNG\r\n\x1a\n'):
        return "image/png"
    elif file_content.startswith(b'\xff\xd8\xff'):
        return "image/jpeg"
    elif file_content.startswith(b'II*\x00') or file_content.startswith(b'MM\x00*'):
        return "image/tiff"
    elif file_content.startswith(b'BM'):
        return "image/bmp"
    elif len(file_content) > 132 and file_content[128:132] == b'DICM':
        return "application/dicom"
    elif file_content.startswith(b'\x08\x00') or file_content.startswith(b'\x00\x08'):
        # Possible DICOM without preamble
        return "application/dicom"
    else:
        return "application/octet-stream"


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and other attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem operations
    """
    # Remove path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure filename is not empty
    if not filename:
        filename = "uploaded_file"
    
    # Limit filename length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_len = 255 - len(ext) - 1 if ext else 255
        filename = name[:max_name_len] + ('.' + ext if ext else '')
    
    return filename


def _validate_image_content(file_content: bytes) -> None:
    """
    Validate image file content for security issues.
    
    Args:
        file_content: Image file content
        
    Raises:
        HTTPException: If image validation fails
    """
    # Check for common image file signatures
    image_signatures = {
        b'\x89PNG\r\n\x1a\n': 'PNG',
        b'\xff\xd8\xff': 'JPEG',
        b'II*\x00': 'TIFF (little-endian)',
        b'MM\x00*': 'TIFF (big-endian)',
        b'BM': 'BMP'
    }
    
    # Verify file starts with valid image signature
    valid_signature = False
    for signature in image_signatures:
        if file_content.startswith(signature):
            valid_signature = True
            break
    
    if not valid_signature:
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid image file"
        )


def _validate_dicom_content(file_content: bytes) -> None:
    """
    Validate DICOM file content.
    
    Args:
        file_content: DICOM file content
        
    Raises:
        HTTPException: If DICOM validation fails
    """
    # DICOM files should have 'DICM' at offset 128
    if len(file_content) > 132:
        if file_content[128:132] != b'DICM':
            # Some DICOM files don't have the preamble, check for DICOM tags
            if not (file_content.startswith(b'\x08\x00') or file_content.startswith(b'\x00\x08')):
                raise HTTPException(
                    status_code=400,
                    detail="File does not appear to be a valid DICOM file"
                )
    else:
        raise HTTPException(
            status_code=400,
            detail="File too small to be a valid DICOM file"
        )


def _validate_unknown_content(file_content: bytes, filename: str) -> None:
    """
    Validate files detected as application/octet-stream.
    
    Args:
        file_content: File content
        filename: Original filename
        
    Raises:
        HTTPException: If file validation fails
    """
    # Check if it might be a valid image or DICOM based on extension
    filename_lower = filename.lower()
    
    if filename_lower.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')):
        # Should be an image, validate image signatures
        _validate_image_content(file_content)
    elif filename_lower.endswith(('.dcm', '.dicom')):
        # Should be DICOM, validate DICOM structure
        _validate_dicom_content(file_content)
    else:
        # Unknown file type with octet-stream detection - reject for security
        raise HTTPException(
            status_code=400,
            detail=f"File type could not be determined. Supported extensions: .png, .jpg, .jpeg, .tiff, .bmp, .dcm"
        )