#!/usr/bin/env python3
"""
Security Module for Medical AI Platform

Provides JWT authentication, password hashing, file validation,
rate limiting, and other security utilities.
"""

import io
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

try:
    import magic

    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    magic = None

try:
    import pyclamd

    CLAMAV_AVAILABLE = True
except ImportError:
    CLAMAV_AVAILABLE = False
    pyclamd = None

from fastapi import HTTPException, Request

try:
    from jose import JWTError, jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    JWTError = Exception
    jwt = None

try:
    from passlib.context import CryptContext

    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False
    CryptContext = None

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
from slowapi import Limiter
from slowapi.util import get_remote_address  # Keep for reference but use custom function

logger = logging.getLogger(__name__)

# Password hashing configuration
if PASSLIB_AVAILABLE and CryptContext:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
else:
    pwd_context = None

# JWT configuration
# JWT configuration - MUST be set via environment variable for production
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    # Fail fast in production
    if os.getenv("ENVIRONMENT", "development").lower() == "production":
        raise RuntimeError(
            "JWT_SECRET_KEY environment variable must be set in production. "
            "Generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    # For development/testing only - generate temporary key
    import secrets

    SECRET_KEY = secrets.token_urlsafe(32)
    logger.warning(
        "Generated temporary JWT secret key for development. "
        "Set JWT_SECRET_KEY environment variable for production."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# File upload configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/tiff", "image/bmp"]
ALLOWED_DICOM_TYPES = ["application/dicom"]


# Rate limiting configuration
# Security: Use custom key function to prevent X-Forwarded-For spoofing
def get_client_ip(request: Request) -> str:
    """Get client IP with X-Forwarded-For validation.

    Only trusts X-Forwarded-For if request comes from trusted proxy.
    Otherwise uses direct connection IP to prevent rate limit bypass.
    """
    import ipaddress

    # Get direct connection IP
    client_host = request.client.host if request.client else "unknown"

    # Trusted proxy IPs (configure for your infrastructure)
    trusted_proxies_str = os.getenv("TRUSTED_PROXIES", "")
    trusted_proxies = set()

    # Validate and parse trusted proxy IPs
    for ip_str in trusted_proxies_str.split(","):
        ip_str = ip_str.strip()
        if not ip_str:
            continue
        try:
            # Validate IP address format
            ipaddress.ip_address(ip_str)
            trusted_proxies.add(ip_str)
        except ValueError:
            logger.warning(f"Invalid trusted proxy IP address: {ip_str}")

    # Only trust X-Forwarded-For from known proxies
    if client_host in trusted_proxies:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take first IP (original client)
            return forwarded.split(",")[0].strip()

    # Use direct connection IP
    return client_host


limiter = Limiter(key_func=get_client_ip, default_limits=["30/minute"])  # Reduced from 100/minute

# CSRF Protection Note:
# For web clients (non-API), implement CSRF protection using:
# - fastapi-csrf-protect library
# - Double-submit cookie pattern
# - SameSite cookie attribute (already set in session cookies)
# API clients using JWT Bearer tokens are not vulnerable to CSRF
# but any cookie-based authentication MUST implement CSRF protection.

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
    if not PASSLIB_AVAILABLE or not pwd_context:
        raise HTTPException(status_code=500, detail="Password hashing not available")
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password to verify against

    Returns:
        True if password matches, False otherwise
    """
    if not PASSLIB_AVAILABLE or not pwd_context:
        return False  # Fail safely
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

    if not JWT_AVAILABLE or not jwt:
        raise HTTPException(status_code=500, detail="JWT functionality not available")

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
    if not JWT_AVAILABLE or not jwt:
        raise HTTPException(status_code=500, detail="JWT functionality not available")

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

    # Remove any path components (prevent directory traversal)
    filename = os.path.basename(filename)

    # Additional path traversal protection
    if ".." in filename or "/" in filename or "\\" in filename:
        logger.warning(f"Path traversal attempt detected in filename: {filename}")
        filename = filename.replace("..", "").replace("/", "").replace("\\", "")

    # Remove any non-alphanumeric characters except dots, dashes, and underscores
    filename = re.sub(r"[^\w\s.-]", "", filename)

    # Replace spaces with underscores
    filename = filename.replace(" ", "_")

    # Prevent hidden files
    if filename.startswith("."):
        filename = "file" + filename

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
        # Detect actual file type using magic bytes if available
        if MAGIC_AVAILABLE and magic:
            detected_type = magic.from_buffer(file_content, mime=True)
        else:
            # Fallback: try to detect from file header
            if file_content.startswith(b"\xff\xd8\xff"):
                detected_type = "image/jpeg"
            elif file_content.startswith(b"\x89PNG\r\n\x1a\n"):
                detected_type = "image/png"
            elif file_content.startswith(b"GIF87a") or file_content.startswith(b"GIF89a"):
                detected_type = "image/gif"
            elif file_content.startswith(b"RIFF") and b"WEBP" in file_content[:12]:
                detected_type = "image/webp"
            else:
                detected_type = "application/octet-stream"

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
    """Scan file content for malware signatures using ClamAV.

    Attempts to connect to ClamAV daemon (clamd) for malware scanning.
    Falls back to basic signature checks if ClamAV unavailable.

    Args:
        file_content: File content as bytes

    Returns:
        True if file is clean, False if malware detected
    """
    # Try ClamAV integration first if available
    if CLAMAV_AVAILABLE and pyclamd:
        try:
            # Try Unix socket first (common on Linux)
            cd = pyclamd.ClamdUnixSocket()
            if not cd.ping():
                # Fall back to network socket (common on Windows/Docker)
                cd = pyclamd.ClamdNetworkSocket()
                if not cd.ping():
                    raise ConnectionError("ClamAV daemon not available")

            # Scan file content
            scan_result = cd.scan_stream(file_content)

            if scan_result is None:
                # Clean file
                return True
            else:
                # Malware detected
                logger.warning(f"ClamAV detected malware: {scan_result}")
                return False

        except (ConnectionError, Exception) as e:
            logger.warning(f"ClamAV not available ({e}), falling back to basic signature checks")
            # Fall through to basic checks
    else:
        logger.info("ClamAV not available, using basic signature checks")

    # Fallback: Basic signature checks
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

    # Check CORS origins
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
    if not allowed_origins or allowed_origins == "*":
        errors.append(
            "ALLOWED_ORIGINS environment variable not set or set to '*' - CORS is insecure"
        )

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

    Logs to application logger and optionally sends to centralized
    audit logging system (Elasticsearch) if configured.

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

    # Send to centralized audit logging system if configured
    _send_to_audit_system(log_entry)

    # Send to IDS for intrusion detection
    _send_to_ids(event_type, username, ip_address, details, success)


def _send_to_audit_system(log_entry: Dict) -> None:
    """Send audit log entry to centralized logging system.

    Supports Elasticsearch, Splunk, or custom SIEM endpoints.
    Fails gracefully if system unavailable.

    Args:
        log_entry: Audit log entry dictionary
    """
    # Check for Elasticsearch configuration
    es_url = os.getenv("ELASTICSEARCH_URL")
    es_index = os.getenv("ELASTICSEARCH_AUDIT_INDEX", "security-audit")
    es_api_key = os.getenv("ELASTICSEARCH_API_KEY")

    if es_url:
        try:
            import requests

            headers = {"Content-Type": "application/json"}
            if es_api_key:
                headers["Authorization"] = f"ApiKey {es_api_key}"

            # Send to Elasticsearch
            response = requests.post(
                f"{es_url}/{es_index}/_doc", json=log_entry, headers=headers, timeout=5
            )
            response.raise_for_status()

        except ImportError:
            logger.debug("requests library not available for Elasticsearch integration")
        except Exception as e:
            # Don't fail application if audit system unavailable
            logger.debug(f"Failed to send audit log to Elasticsearch: {e}")

    # Check for Splunk HEC configuration
    splunk_url = os.getenv("SPLUNK_HEC_URL")
    splunk_token = os.getenv("SPLUNK_HEC_TOKEN")

    if splunk_url and splunk_token:
        try:
            import requests

            # Format for Splunk HEC
            splunk_event = {"event": log_entry, "sourcetype": "_json", "source": "histocore-api"}

            headers = {
                "Authorization": f"Splunk {splunk_token}",
                "Content-Type": "application/json",
            }

            response = requests.post(splunk_url, json=splunk_event, headers=headers, timeout=5)
            response.raise_for_status()

        except ImportError:
            logger.debug("requests library not available for Splunk integration")
        except Exception as e:
            # Don't fail application if audit system unavailable
            logger.debug(f"Failed to send audit log to Splunk: {e}")


def _send_to_ids(
    event_type: str,
    username: Optional[str],
    ip_address: Optional[str],
    details: Optional[str],
    success: bool,
) -> None:
    """Send security event to IDS for intrusion detection.

    Args:
        event_type: Type of security event
        username: Username if available
        ip_address: Source IP address
        details: Event details
        success: Whether event was successful
    """
    try:
        from src.platform.monitoring.ids import create_ids_event_from_security_log, get_ids_engine

        # Only send failed events to IDS (potential attacks)
        if not success:
            ids_engine = get_ids_engine()
            ids_event = create_ids_event_from_security_log(
                event_type=event_type,
                username=username,
                ip_address=ip_address,
                details=details,
            )
            ids_engine.process_event(ids_event)

    except ImportError:
        logger.debug("IDS module not available")
    except Exception as e:
        # Don't fail application if IDS unavailable
        logger.debug(f"Failed to send event to IDS: {e}")
