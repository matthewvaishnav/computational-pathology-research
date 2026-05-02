#!/usr/bin/env python3
"""
Security Module for Medical AI Platform

Provides JWT authentication, password hashing, file validation,
rate limiting, and other security utilities.
"""

import hashlib
import io
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import magic
from fastapi import HTTPException, Request
from jose import JWTError, jwt
from passlib.context import CryptContext
from PIL import Image
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION_USE_ENV_VAR")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# File upload configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/tiff", "image/bmp"]
ALLOWED_DICOM_TYPES = ["application/dicom"]

# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# Failed login tracking
failed_login_attempts: Dict[str, Tuple[int, float]] = {}
LOCKOUT_THRESHOLD = 5
LOCKOUT_DURATION = 900  # 15 minutes in seconds


# ============================================================================
# Password Hashing
# ============================================================================


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password to verify against

    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


# ============================================================================
# JWT Token Management
# ============================================================================


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.

    Args:
        data: Data to encode in the token (should include 'sub' for user ID)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "iat": datetime.utcnow()})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT access token.

    Args:
        token: JWT token to decode

    Returns:
        Decoded token payload if valid, None otherwise

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ============================================================================
# Rate Limiting and Brute Force Protection
# ============================================================================


def check_account_lockout(username: str) -> None:
    """Check if account is locked due to failed login attempts.

    Args:
        username: Username to check

    Raises:
        HTTPException: If account is locked
    """
    if username in failed_login_attempts:
        attempts, lockout_until = failed_login_attempts[username]

        if time.time() < lockout_until:
            remaining_seconds = int(lockout_until - time.time())
            logger.warning(f"Account locked for user {username}: {remaining_seconds}s remaining")
            raise HTTPException(
                status_code=429,
                detail=f"Account locked due to too many failed attempts. Try again in {remaining_seconds} seconds.",
            )


def record_failed_login(username: str) -> None:
    """Record a failed login attempt and lock account if threshold exceeded.

    Args:
        username: Username that failed login

    Raises:
        HTTPException: If account is now locked
    """
    current_time = time.time()

    if username in failed_login_attempts:
        attempts, _ = failed_login_attempts[username]
        attempts += 1
    else:
        attempts = 1

    if attempts >= LOCKOUT_THRESHOLD:
        lockout_until = current_time + LOCKOUT_DURATION
        failed_login_attempts[username] = (attempts, lockout_until)
        logger.warning(f"Account locked for user {username} after {attempts} failed attempts")
        raise HTTPException(
            status_code=429,
            detail=f"Account locked due to too many failed attempts. Try again in {LOCKOUT_DURATION // 60} minutes.",
        )
    else:
        failed_login_attempts[username] = (attempts, 0)
        logger.info(f"Failed login attempt {attempts}/{LOCKOUT_THRESHOLD} for user {username}")


def clear_failed_login(username: str) -> None:
    """Clear failed login attempts for a user after successful login.

    Args:
        username: Username to clear
    """
    if username in failed_login_attempts:
        del failed_login_attempts[username]
        logger.info(f"Cleared failed login attempts for user {username}")


# ============================================================================
# File Upload Validation
# ============================================================================


def secure_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal attacks.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem use
    """
    # Normalize unicode characters
    filename = unicodedata.normalize("NFKD", filename)

    # Remove any path components
    filename = os.path.basename(filename)

    # Remove any non-alphanumeric characters except dots, dashes, and underscores
    filename = re.sub(r"[^\w\s.-]", "", filename)

    # Replace spaces with underscores
    filename = filename.replace(" ", "_")

    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext

    # Ensure filename is not empty
    if not filename or filename == ".":
        filename = "unnamed_file"

    return filename


def validate_file_size(file_content: bytes, max_size: int = MAX_FILE_SIZE) -> None:
    """Validate file size is within limits.

    Args:
        file_content: File content as bytes
        max_size: Maximum allowed file size in bytes

    Raises:
        HTTPException: If file is too large
    """
    file_size = len(file_content)

    if file_size > max_size:
        logger.warning(f"File too large: {file_size} bytes (max: {max_size})")
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_size // (1024 * 1024)}MB",
        )


def validate_image_magic_bytes(file_content: bytes) -> str:
    """Validate file is actually an image using magic bytes.

    Args:
        file_content: File content as bytes

    Returns:
        Detected MIME type

    Raises:
        HTTPException: If file is not a valid image
    """
    try:
        # Detect actual file type using magic bytes
        detected_type = magic.from_buffer(file_content, mime=True)

        if detected_type not in ALLOWED_IMAGE_TYPES:
            logger.warning(f"Invalid image type detected: {detected_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}",
            )

        return detected_type

    except Exception as e:
        logger.error(f"Error detecting file type: {e}")
        raise HTTPException(status_code=400, detail="Unable to validate file type")


def validate_image_integrity(file_content: bytes) -> None:
    """Validate image can be opened and is not corrupted.

    Args:
        file_content: File content as bytes

    Raises:
        HTTPException: If image is corrupted or cannot be opened
    """
    try:
        # Try to open image with PIL
        image = Image.open(io.BytesIO(file_content))

        # Verify image by loading it
        image.verify()

        # Re-open for additional checks (verify() closes the file)
        image = Image.open(io.BytesIO(file_content))

        # Check image dimensions are reasonable
        width, height = image.size
        if width < 1 or height < 1 or width > 100000 or height > 100000:
            raise HTTPException(
                status_code=400, detail="Image dimensions are invalid or unreasonable"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Image integrity validation failed: {e}")
        raise HTTPException(status_code=400, detail="Corrupted or invalid image file")


def validate_dicom_file(file_content: bytes) -> None:
    """Validate DICOM file format.

    Args:
        file_content: File content as bytes

    Raises:
        HTTPException: If file is not a valid DICOM file
    """
    try:
        # Check DICOM magic bytes (DICM at offset 128)
        if len(file_content) < 132:
            raise HTTPException(status_code=400, detail="File too small to be a valid DICOM file")

        magic_bytes = file_content[128:132]
        if magic_bytes != b"DICM":
            raise HTTPException(status_code=400, detail="Invalid DICOM file format")

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"DICOM validation failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid DICOM file")


def scan_for_malware(file_content: bytes) -> bool:
    """Scan file content for malware signatures.

    Note: This is a placeholder. In production, integrate with ClamAV or similar.

    Args:
        file_content: File content as bytes

    Returns:
        True if file is clean, False if malware detected
    """
    # TODO: Integrate with ClamAV or similar antivirus solution
    # For now, perform basic checks

    # Check for common malware signatures
    malware_signatures = [
        b"<script",  # JavaScript injection
        b"<?php",  # PHP code injection
        b"eval(",  # Code evaluation
        b"exec(",  # Command execution
        b"system(",  # System command execution
    ]

    for signature in malware_signatures:
        if signature in file_content:
            logger.warning(f"Potential malware signature detected: {signature}")
            return False

    return True


def validate_uploaded_image(
    file_content: bytes, filename: str, content_type: str
) -> Tuple[str, str]:
    """Comprehensive validation for uploaded image files.

    Args:
        file_content: File content as bytes
        filename: Original filename
        content_type: Content-Type header from upload

    Returns:
        Tuple of (sanitized_filename, detected_mime_type)

    Raises:
        HTTPException: If validation fails
    """
    # 1. Validate file size
    validate_file_size(file_content)

    # 2. Sanitize filename
    safe_filename = secure_filename(filename)

    # 3. Validate magic bytes (actual file type)
    detected_type = validate_image_magic_bytes(file_content)

    # 4. Validate image integrity
    validate_image_integrity(file_content)

    # 5. Scan for malware
    if not scan_for_malware(file_content):
        raise HTTPException(status_code=400, detail="File failed security scan")

    logger.info(f"Image validation passed: {safe_filename} ({detected_type})")
    return safe_filename, detected_type


# ============================================================================
# Input Sanitization
# ============================================================================


def sanitize_for_log(value: str) -> str:
    """Sanitize string for safe logging (prevent log injection).

    Args:
        value: String to sanitize

    Returns:
        Sanitized string safe for logging
    """
    if not isinstance(value, str):
        value = str(value)

    # Replace newlines and carriage returns
    value = value.replace("\n", "\\n").replace("\r", "\\r")

    # Remove ANSI escape codes
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    value = ansi_escape.sub("", value)

    # Limit length
    if len(value) > 1000:
        value = value[:997] + "..."

    return value


def sanitize_sql_identifier(identifier: str) -> str:
    """Sanitize SQL identifier (table/column name).

    Args:
        identifier: SQL identifier to sanitize

    Returns:
        Sanitized identifier

    Raises:
        ValueError: If identifier is invalid
    """
    # Only allow alphanumeric and underscore
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier}")

    # Prevent SQL keywords
    sql_keywords = [
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "EXEC",
        "EXECUTE",
    ]
    if identifier.upper() in sql_keywords:
        raise ValueError(f"SQL keyword not allowed as identifier: {identifier}")

    return identifier


# ============================================================================
# Security Headers
# ============================================================================


def get_security_headers() -> Dict[str, str]:
    """Get security headers for HTTP responses.

    Returns:
        Dictionary of security headers
    """
    return {
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }


# ============================================================================
# Environment Configuration Validation
# ============================================================================


def validate_security_configuration() -> None:
    """Validate security configuration on startup.

    Raises:
        RuntimeError: If critical security configuration is missing
    """
    errors = []

    # Check JWT secret key
    if SECRET_KEY == "CHANGE_THIS_IN_PRODUCTION_USE_ENV_VAR":
        errors.append("JWT_SECRET_KEY environment variable not set - using insecure default")

    # Check CORS origins
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
    if not allowed_origins or allowed_origins == "*":
        errors.append("ALLOWED_ORIGINS environment variable not set or set to '*' - CORS is insecure")

    # Check environment
    environment = os.getenv("ENVIRONMENT", "development")
    if environment == "production" and errors:
        raise RuntimeError(
            f"Critical security configuration errors in production: {'; '.join(errors)}"
        )
    elif errors:
        for error in errors:
            logger.warning(f"Security configuration warning: {error}")


# ============================================================================
# Audit Logging
# ============================================================================


def log_security_event(
    event_type: str,
    username: Optional[str] = None,
    ip_address: Optional[str] = None,
    details: Optional[str] = None,
    success: bool = True,
) -> None:
    """Log security-relevant events for audit trail.

    Args:
        event_type: Type of security event (e.g., 'login', 'access_denied', 'file_upload')
        username: Username associated with event
        ip_address: IP address of client
        details: Additional details about the event
        success: Whether the event was successful
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "username": username,
        "ip_address": ip_address,
        "details": sanitize_for_log(details) if details else None,
        "success": success,
    }

    # Log to application logger
    if success:
        logger.info(f"Security event: {log_entry}")
    else:
        logger.warning(f"Security event (failed): {log_entry}")

    # TODO: Send to centralized audit logging system (e.g., Elasticsearch, Splunk)
